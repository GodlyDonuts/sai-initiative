#!/bin/bash
#SBATCH --job-name=sai-launch-100m-250m
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/sai_launch_100m_250m_%j.out
#SBATCH --error=logs/sai_launch_100m_250m_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${GEOMETRY:?GEOMETRY is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"
: "${DEVELOPMENT_STREAM:?DEVELOPMENT_STREAM is required}"
: "${ENVIRONMENT_RECEIPT:?ENVIRONMENT_RECEIPT is required}"
: "${SCREEN_SCOPE:?SCREEN_SCOPE is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -f "$GEOMETRY"
test -d "$TRAIN_STREAM"
test -d "$DEVELOPMENT_STREAM"
test -f "$ENVIRONMENT_RECEIPT"
test ! -L "$ENVIRONMENT_RECEIPT"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test -w "$LOG_ROOT"
test -d "$(dirname "$OUTPUT_ROOT")"
test ! -L "$(dirname "$OUTPUT_ROOT")"
test ! -e "$OUTPUT_ROOT"

case "$SCREEN_SCOPE" in
  gqa_only)
    families=(gated_gqa)
    comparison_class="single_family_reference_baseline_not_tournament"
    purpose="establish_250m_token_gqa_reference_baseline"
    ;;
  three_family)
    : "${FULL_MODEL_PARITY_RECEIPT:?FULL_MODEL_PARITY_RECEIPT is required for three_family}"
    : "${FULL_MODEL_PARITY_SHA256:?FULL_MODEL_PARITY_SHA256 is required for three_family}"
    test -f "$FULL_MODEL_PARITY_RECEIPT"
    test ! -L "$FULL_MODEL_PARITY_RECEIPT"
    test "$(sha256sum "$FULL_MODEL_PARITY_RECEIPT" | cut -d ' ' -f 1)" = "$FULL_MODEL_PARITY_SHA256"
    families=(gated_gqa gdn_hybrid kda_mla_hybrid)
    comparison_class="iso_data_250m_token_screen_not_iso_flop"
    purpose="distinguish_data_starvation_from_mixer_failure"
    ;;
  *)
    printf '%s\n' "unsupported SCREEN_SCOPE: $SCREEN_SCOPE" >&2
    exit 2
    ;;
esac

if [[ "$SCREEN_SCOPE" = three_family ]]; then
  "$PYTHON" -P -s -B - "$FULL_MODEL_PARITY_RECEIPT" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert receipt["schema"] == "sai-full-delta-mixer-fla-parity-v1"
assert receipt["status"] == "production_cuda_qualified"
assert receipt["production_cuda_qualified"] is True
assert receipt["fla_version"] == "0.4.2"
assert receipt["optimizer_steps"] == 0
assert receipt["training_authorized"] is False
assert receipt["architecture_promoted"] is False
cases = receipt["cases"]
assert len(cases) == 8
assert {(row["family"], row["sequence_length"]) for row in cases} == {
    (family, length)
    for family in ("gdn", "kda")
    for length in (1, 63, 64, 65)
}
assert all(row["passed"] is True for row in cases)
assert receipt["checks"]["every_parameter_gradient_compared"] is True
assert receipt["checks"]["packed_row_boundaries_verified"] is True
assert receipt["checks"]["explicit_scale_one_verified"] is True
PY
fi

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

read -r train_identity development_identity < <(
  "$PYTHON" -P -s -B - "$TRAIN_STREAM" "$DEVELOPMENT_STREAM" <<'PY'
import pathlib
import sys

from sai.data.token_stream import validate_frozen_stream

train = validate_frozen_stream(pathlib.Path(sys.argv[1]), verify_sources=True)
development = validate_frozen_stream(pathlib.Path(sys.argv[2]), verify_sources=True)
assert train["sequence_length"] == 2_048
assert development["sequence_length"] == 2_048
assert train["sequences"] >= 122_070
assert development["sequences"] >= 1_024
assert train["prefix_utf8_bytes"]["122070"] > 0
assert development["prefix_utf8_bytes"]["1024"] > 0
assert train["tokenizer_identity_sha256"] == development["tokenizer_identity_sha256"]
print(
    train["ordered_stream_identity_sha256"],
    development["ordered_stream_identity_sha256"],
)
PY
)
test "${#train_identity}" = 64
test "${#development_identity}" = 64
test "$train_identity" != "$development_identity"

mkdir "$OUTPUT_ROOT"
job_ids=()

cancel_partial_graph() {
  local job_id
  for job_id in "${job_ids[@]}"; do
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
      scancel "$job_id" || true
    fi
  done
}
trap cancel_partial_graph ERR

for family in "${families[@]}"; do
  checkpoint="$OUTPUT_ROOT/$family.checkpoint.pt"
  output="$OUTPUT_ROOT/$family.result.json"
  test ! -e "$checkpoint"
  test ! -e "$output"
  job_export="SAI_ROOT=$SAI_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT,PYTHON=$PYTHON,FLA_ROOT=$FLA_ROOT,GEOMETRY=$GEOMETRY,FAMILY=$family,TRAIN_STREAM=$TRAIN_STREAM,TRAIN_IDENTITY=$train_identity,DEVELOPMENT_STREAM=$DEVELOPMENT_STREAM,DEVELOPMENT_IDENTITY=$development_identity,DEVELOPMENT_SEQUENCES=1024,OPTIMIZER_STEPS=477,TRAINING_SEQUENCES=122070,SEQUENCES_PER_UPDATE=256,BATCH_SIZE=8,DEVELOPMENT_BATCH_SIZE=8,CHECKPOINT_INTERVAL=10,SEED=20260821,ENVIRONMENT_RECEIPT=$ENVIRONMENT_RECEIPT,CHECKPOINT=$checkpoint,OUTPUT=$output"
  if [[ "$family" != gated_gqa ]]; then
    job_export+=",FULL_MODEL_PARITY_RECEIPT=$FULL_MODEL_PARITY_RECEIPT,FULL_MODEL_PARITY_SHA256=$FULL_MODEL_PARITY_SHA256"
  fi
  job_id="$(
    env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
      --chdir="$SAI_ROOT" \
      --output="$LOG_ROOT/sai_100m_250m_%j.out" \
      --error="$LOG_ROOT/sai_100m_250m_%j.err" \
      --export="$job_export" \
      "$SAI_ROOT/jobs/sai-short-screen-single-h100.sbatch"
  )"
  [[ "$job_id" =~ ^[0-9]+$ ]]
  job_ids+=("$job_id")
done

dispatch="$OUTPUT_ROOT/dispatch.json"
"$PYTHON" -P -s -B - "$dispatch" "$EXPECTED_COMMIT" "$train_identity" "$development_identity" "$LOG_ROOT" "$SCREEN_SCOPE" "$comparison_class" "$purpose" "${families[*]}" "${job_ids[*]}" <<'PY'
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sai-100m-250m-token-screen-dispatch-v1",
    "status": "submitted",
    "source_commit": sys.argv[2],
    "training_stream_identity_sha256": sys.argv[3],
    "development_stream_identity_sha256": sys.argv[4],
    "log_root": sys.argv[5],
    "screen_scope": sys.argv[6],
    "comparison_class": sys.argv[7],
    "purpose": sys.argv[8],
    "families": sys.argv[9].split(),
    "job_ids": [int(value) for value in sys.argv[10].split()],
    "gpu_jobs_submitted": len(sys.argv[10].split()),
    "maximum_concurrent_gpu_jobs": len(sys.argv[10].split()),
    "gpus_per_job": 1,
    "requeue": False,
    "training_sequences": 122_070,
    "sequence_length": 2_048,
    "training_tokens": 249_999_360,
    "sequences_per_update": 256,
    "optimizer_steps": 477,
    "development_sequences": 1_024,
    "scientific_promotion_authorized": False,
    "four_b_training_authorized": False,
}
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
PY

trap - ERR
printf '%s\n' "jobs=${job_ids[*]}"
