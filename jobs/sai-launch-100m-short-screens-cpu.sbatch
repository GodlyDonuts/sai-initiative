#!/bin/bash
#SBATCH --job-name=sai-launch-100m
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=logs/sai_launch_100m_%j.out
#SBATCH --error=logs/sai_launch_100m_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${GEOMETRY:?GEOMETRY is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"
: "${DEVELOPMENT_STREAM:?DEVELOPMENT_STREAM is required}"
: "${ENVIRONMENT_RECEIPT:?ENVIRONMENT_RECEIPT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

test -d "$SAI_ROOT/.git"
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -f "$GEOMETRY"
test -d "$TRAIN_STREAM"
test -d "$DEVELOPMENT_STREAM"
test -f "$ENVIRONMENT_RECEIPT"
test ! -L "$ENVIRONMENT_RECEIPT"
test -d "$(dirname "$OUTPUT_ROOT")"
test ! -L "$(dirname "$OUTPUT_ROOT")"
test ! -e "$OUTPUT_ROOT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

read -r train_identity development_identity < <(
  "$PYTHON" -P -s -B - "$TRAIN_STREAM" "$DEVELOPMENT_STREAM" <<'PY'
import pathlib
import sys

from sai.data.token_stream import validate_frozen_stream

train = validate_frozen_stream(pathlib.Path(sys.argv[1]), verify_sources=True)
development = validate_frozen_stream(pathlib.Path(sys.argv[2]), verify_sources=True)
assert train["sequences"] >= 48_828
assert development["sequences"] >= 4_096
assert train["prefix_utf8_bytes"]["256"] > 0
assert train["prefix_utf8_bytes"]["48828"] > 0
assert development["prefix_utf8_bytes"]["4096"] > 0
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
canary_job_ids=()
full_job_ids=()

cancel_partial_graph() {
  local job_id
  for job_id in "${canary_job_ids[@]}" "${full_job_ids[@]}"; do
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
      scancel "$job_id" || true
    fi
  done
}
trap cancel_partial_graph ERR

for family in gated_gqa gdn_hybrid kda_mla_hybrid; do
  canary_checkpoint="$OUTPUT_ROOT/$family.canary.checkpoint.pt"
  canary_output="$OUTPUT_ROOT/$family.canary.result.json"
  checkpoint="$OUTPUT_ROOT/$family.checkpoint.pt"
  output="$OUTPUT_ROOT/$family.result.json"
  test ! -e "$canary_checkpoint"
  test ! -e "$canary_output"
  test ! -e "$checkpoint"
  test ! -e "$output"
  canary_job_id="$(
    env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
      --chdir="$SAI_ROOT" \
      --time=00:45:00 \
      --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FLA_ROOT="$FLA_ROOT",GEOMETRY="$GEOMETRY",FAMILY="$family",TRAIN_STREAM="$TRAIN_STREAM",TRAIN_IDENTITY="$train_identity",DEVELOPMENT_STREAM="$DEVELOPMENT_STREAM",DEVELOPMENT_IDENTITY="$development_identity",DEVELOPMENT_SEQUENCES=1,OPTIMIZER_STEPS=1,TRAINING_SEQUENCES=256,SEQUENCES_PER_UPDATE=256,BATCH_SIZE=8,DEVELOPMENT_BATCH_SIZE=1,CHECKPOINT_INTERVAL=1,SEED=20260821,MECHANICS_ONLY=1,ENVIRONMENT_RECEIPT="$ENVIRONMENT_RECEIPT",CHECKPOINT="$canary_checkpoint",OUTPUT="$canary_output" \
      "$SAI_ROOT/jobs/sai-short-screen-single-h100.sbatch"
  )"
  [[ "$canary_job_id" =~ ^[0-9]+$ ]]
  canary_job_ids+=("$canary_job_id")
  full_job_id="$(
    env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
      --chdir="$SAI_ROOT" \
      --dependency="afterok:$canary_job_id" \
      --kill-on-invalid-dep=yes \
      --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FLA_ROOT="$FLA_ROOT",GEOMETRY="$GEOMETRY",FAMILY="$family",TRAIN_STREAM="$TRAIN_STREAM",TRAIN_IDENTITY="$train_identity",DEVELOPMENT_STREAM="$DEVELOPMENT_STREAM",DEVELOPMENT_IDENTITY="$development_identity",DEVELOPMENT_SEQUENCES=4096,OPTIMIZER_STEPS=191,TRAINING_SEQUENCES=48828,SEQUENCES_PER_UPDATE=256,BATCH_SIZE=8,DEVELOPMENT_BATCH_SIZE=8,CHECKPOINT_INTERVAL=10,SEED=20260821,ENVIRONMENT_RECEIPT="$ENVIRONMENT_RECEIPT",CHECKPOINT="$checkpoint",OUTPUT="$output" \
      "$SAI_ROOT/jobs/sai-short-screen-single-h100.sbatch"
  )"
  [[ "$full_job_id" =~ ^[0-9]+$ ]]
  full_job_ids+=("$full_job_id")
done

dispatch="$OUTPUT_ROOT/dispatch.json"
"$PYTHON" -P -s -B - "$dispatch" "$EXPECTED_COMMIT" "$train_identity" "$development_identity" "${canary_job_ids[*]}" "${full_job_ids[*]}" <<'PY'
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sai-100m-short-screen-dispatch-v1",
    "status": "submitted",
    "source_commit": sys.argv[2],
    "training_stream_identity_sha256": sys.argv[3],
    "development_stream_identity_sha256": sys.argv[4],
    "families": ["gated_gqa", "gdn_hybrid", "kda_mla_hybrid"],
    "canary_job_ids": [int(value) for value in sys.argv[5].split()],
    "full_job_ids": [int(value) for value in sys.argv[6].split()],
    "gpu_jobs_submitted": 6,
    "maximum_concurrent_gpu_jobs": 3,
    "gpus_per_job": 1,
    "requeue": False,
    "training_sequences": 48_828,
    "sequences_per_update": 256,
    "optimizer_steps": 191,
    "development_sequences": 4_096,
    "comparison_class": "iso_data_short_screen_not_iso_flop",
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

printf '%s\n' "canary=${canary_job_ids[*]} full=${full_job_ids[*]}"
