#!/bin/bash
#SBATCH --job-name=sai-stage-100m-dev-mc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${ENVIRONMENT_RECEIPT:?ENVIRONMENT_RECEIPT is required}"
: "${GEOMETRY:?GEOMETRY is required}"
: "${SCREEN_LAUNCH_JOB_ID:?SCREEN_LAUNCH_JOB_ID is required}"
: "${SCREEN_SOURCE_COMMIT:?SCREEN_SOURCE_COMMIT is required}"
: "${SCREEN_ROOT:?SCREEN_ROOT is required}"
: "${TRAINING_STREAM:?TRAINING_STREAM is required}"
: "${TOKENIZER_ROOT:?TOKENIZER_ROOT is required}"
: "${POPULATION_JOB_ID:?POPULATION_JOB_ID is required}"
: "${POPULATION_AGGREGATE:?POPULATION_AGGREGATE is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

FAMILY_LAUNCHER="$SAI_ROOT/jobs/sai-launch-short-screen-development-mc-cpu.sbatch"
COMPARISON_STAGER="$SAI_ROOT/jobs/sai-stage-short-screen-development-mc-comparison-cpu.sbatch"
SCREEN_DISPATCH="$SCREEN_ROOT/dispatch.json"
DISPATCH="$OUTPUT_ROOT/dispatch.json"

require_absolute() {
  case "$1" in
    /*) ;;
    *) echo "path must be absolute: $1" >&2; exit 2 ;;
  esac
}

for path in \
  "$SAI_ROOT" "$PYTHON" "$FLA_ROOT" "$ENVIRONMENT_RECEIPT" "$GEOMETRY" \
  "$SCREEN_ROOT" "$SCREEN_DISPATCH" "$TRAINING_STREAM" "$TOKENIZER_ROOT" \
  "$POPULATION_AGGREGATE" "$OUTPUT_ROOT" "$LOG_ROOT" "$FAMILY_LAUNCHER" \
  "$COMPARISON_STAGER" "$DISPATCH"; do
  require_absolute "$path"
done

test -n "${SLURM_JOB_ID:-}"
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU benchmark stager was exposed to a GPU" >&2; exit 2 ;;
esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -d "$SCREEN_ROOT"
test ! -L "$SCREEN_ROOT"
test -d "$TRAINING_STREAM"
test ! -L "$TRAINING_STREAM"
test -d "$TOKENIZER_ROOT"
test ! -L "$TOKENIZER_ROOT"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test -w "$LOG_ROOT"
for path in \
  "$ENVIRONMENT_RECEIPT" "$GEOMETRY" "$SCREEN_DISPATCH" \
  "$POPULATION_AGGREGATE" "$FAMILY_LAUNCHER" "$COMPARISON_STAGER"; do
  test -f "$path"
  test ! -L "$path"
done
test -d "$(dirname "$OUTPUT_ROOT")"
test ! -e "$OUTPUT_ROOT"
test ! -L "$OUTPUT_ROOT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

screen_state="$(sacct -j "$SCREEN_LAUNCH_JOB_ID" -X -n -P -o State | sed -n '1p')"
population_state="$(sacct -j "$POPULATION_JOB_ID" -X -n -P -o State | sed -n '1p')"
test "$screen_state" = "COMPLETED"
test "$population_state" = "COMPLETED"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

validated_lines="$("$PYTHON" -P -s -B - \
  "$SCREEN_DISPATCH" "$SCREEN_SOURCE_COMMIT" "$SCREEN_ROOT" \
  "$POPULATION_AGGREGATE" "$TRAINING_STREAM" "$TOKENIZER_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

from sai.data.token_stream import canonical_sha256, sha256_tree, validate_frozen_stream


def sha256_file(path):
    assert path.is_file() and not path.is_symlink()
    return hashlib.sha256(path.read_bytes()).hexdigest()


dispatch_path = pathlib.Path(sys.argv[1])
expected_commit = sys.argv[2]
screen_root = pathlib.Path(sys.argv[3])
population_path = pathlib.Path(sys.argv[4])
training_stream_path = pathlib.Path(sys.argv[5])
tokenizer_root = pathlib.Path(sys.argv[6])

dispatch = json.loads(dispatch_path.read_text())
assert set(dispatch) == {
    "schema", "status", "source_commit", "training_stream_identity_sha256",
    "development_stream_identity_sha256", "log_root", "families", "job_ids",
    "gpu_jobs_submitted", "maximum_concurrent_gpu_jobs", "gpus_per_job",
    "requeue", "training_sequences", "sequence_length", "training_tokens",
    "sequences_per_update", "optimizer_steps", "development_sequences",
    "comparison_class", "purpose", "scientific_promotion_authorized",
    "four_b_training_authorized",
}
families = ["gated_gqa", "gdn_hybrid", "kda_mla_hybrid"]
assert dispatch["schema"] == "sai-100m-250m-token-screen-dispatch-v1"
assert dispatch["status"] == "submitted"
assert dispatch["source_commit"] == expected_commit
assert dispatch["families"] == families
assert dispatch["gpu_jobs_submitted"] == 3
assert dispatch["gpus_per_job"] == 1
assert dispatch["requeue"] is False
assert dispatch["training_sequences"] == 122_070
assert dispatch["training_tokens"] == 249_999_360
assert dispatch["optimizer_steps"] == 477
assert dispatch["scientific_promotion_authorized"] is False
assert dispatch["four_b_training_authorized"] is False
job_ids = dispatch["job_ids"]
assert len(job_ids) == 3 and len(set(job_ids)) == 3
assert all(isinstance(job_id, int) and job_id > 0 for job_id in job_ids)

stream = validate_frozen_stream(training_stream_path, verify_sources=True)
tokenizer_sha256 = sha256_tree(tokenizer_root)
assert stream["ordered_stream_identity_sha256"] == dispatch[
    "training_stream_identity_sha256"
]
assert stream["tokenizer_identity_sha256"] == tokenizer_sha256
assert stream["sequence_length"] == 2_048
assert stream["sequences"] >= 122_070

population = json.loads(population_path.read_text())
unsigned = dict(population)
claimed = unsigned.pop("receipt_sha256")
assert population["schema"] == "sai-development-mc-populations-aggregate-v1"
assert population["status"] == "complete"
assert population["development_only"] is True
assert population["official_benchmark_result"] is False
assert population["total_rows"] == 12_788
assert claimed == canonical_sha256(unsigned)

print(sha256_file(dispatch_path))
print(stream["ordered_stream_identity_sha256"])
print(tokenizer_sha256)
print(sha256_file(population_path))
for family, job_id in zip(families, job_ids, strict=True):
    result = screen_root / f"{family}.result.json"
    checkpoint = screen_root / f"{family}.checkpoint.pt"
    manifest = pathlib.Path(f"{checkpoint}.manifest.json")
    print(f"{family}\t{job_id}\t{result}\t{checkpoint}\t{manifest}")
PY
)"

mapfile -t validated <<< "$validated_lines"
test "${#validated[@]}" = 7
SCREEN_DISPATCH_SHA256="${validated[0]}"
TRAINING_STREAM_IDENTITY="${validated[1]}"
TOKENIZER_SHA256="${validated[2]}"
POPULATION_AGGREGATE_SHA256="${validated[3]}"
test "${#SCREEN_DISPATCH_SHA256}" = 64
test "${#TRAINING_STREAM_IDENTITY}" = 64
test "${#TOKENIZER_SHA256}" = 64
test "${#POPULATION_AGGREGATE_SHA256}" = 64

mkdir "$OUTPUT_ROOT"
submitted=()
families=()
training_jobs=()
family_launchers=()

cancel_partial_graph() {
  local job_id
  for job_id in "${submitted[@]}"; do
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then
      scancel "$job_id" || true
    fi
  done
}
trap cancel_partial_graph EXIT

for row in "${validated[@]:4}"; do
  IFS=$'\t' read -r family training_job result checkpoint manifest <<< "$row"
  case "$family" in gated_gqa|gdn_hybrid|kda_mla_hybrid) ;; *) exit 2 ;; esac
  [[ "$training_job" =~ ^[0-9]+$ ]]
  family_root="$OUTPUT_ROOT/$family"
  mkdir "$family_root"
  family_dispatch="$family_root/dispatch.json"
  launcher_id="$(env -i PATH=/apps/slurm/current/bin:/usr/bin:/bin sbatch --parsable \
    --chdir="$SAI_ROOT" \
    --job-name="sai-stage-${family}-mc" \
    --dependency="afterok:$training_job" \
    --kill-on-invalid-dep=yes \
    --output="$LOG_ROOT/sai_stage_${family}_mc_%j.out" \
    --error="$LOG_ROOT/sai_stage_${family}_mc_%j.err" \
    --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FLA_ROOT="$FLA_ROOT",ENVIRONMENT_RECEIPT="$ENVIRONMENT_RECEIPT",LOG_ROOT="$LOG_ROOT",GEOMETRY="$GEOMETRY",FAMILY="$family",SHORT_SCREEN_JOB_ID="$training_job",SHORT_SCREEN_RESULT="$result",CHECKPOINT="$checkpoint",CHECKPOINT_MANIFEST="$manifest",TRAINING_STREAM="$TRAINING_STREAM",TOKENIZER_ROOT="$TOKENIZER_ROOT",POPULATION_JOB_ID="$POPULATION_JOB_ID",POPULATION_AGGREGATE="$POPULATION_AGGREGATE",OUTPUT_ROOT="$family_root",DISPATCH="$family_dispatch" \
    "$FAMILY_LAUNCHER")"
  [[ "$launcher_id" =~ ^[0-9]+$ ]]
  submitted+=("$launcher_id")
  families+=("$family")
  training_jobs+=("$training_job")
  family_launchers+=("$launcher_id")
done

test "${#family_launchers[@]}" = 3
launcher_dependency="$(IFS=:; printf '%s' "${family_launchers[*]}")"
comparison_stage_id="$(env -i PATH=/apps/slurm/current/bin:/usr/bin:/bin sbatch --parsable \
  --chdir="$SAI_ROOT" \
  --dependency="afterok:$launcher_dependency" \
  --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/sai_stage_100m_comparison_%j.out" \
  --error="$LOG_ROOT/sai_stage_100m_comparison_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",LOG_ROOT="$LOG_ROOT",OUTPUT_ROOT="$OUTPUT_ROOT",GATED_GQA_LAUNCH_JOB_ID="${family_launchers[0]}",GDN_HYBRID_LAUNCH_JOB_ID="${family_launchers[1]}",KDA_MLA_HYBRID_LAUNCH_JOB_ID="${family_launchers[2]}" \
  "$COMPARISON_STAGER")"
[[ "$comparison_stage_id" =~ ^[0-9]+$ ]]
submitted+=("$comparison_stage_id")

"$PYTHON" -P -s -B - \
  "$DISPATCH" "$EXPECTED_COMMIT" "$SCREEN_LAUNCH_JOB_ID" \
  "$SCREEN_SOURCE_COMMIT" "$SCREEN_DISPATCH" "$SCREEN_DISPATCH_SHA256" \
  "$POPULATION_JOB_ID" \
  "$POPULATION_AGGREGATE" "$POPULATION_AGGREGATE_SHA256" \
  "$TRAINING_STREAM" "$TRAINING_STREAM_IDENTITY" "$TOKENIZER_ROOT" \
  "$TOKENIZER_SHA256" "$OUTPUT_ROOT" "$comparison_stage_id" \
  "${families[*]}" "${training_jobs[*]}" "${family_launchers[*]}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import uuid


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


values = sys.argv[1:]
output = pathlib.Path(values[0])
families = values[15].split()
training_jobs = [int(value) for value in values[16].split()]
launchers = [int(value) for value in values[17].split()]
assert families == ["gated_gqa", "gdn_hybrid", "kda_mla_hybrid"]
assert len(training_jobs) == len(launchers) == 3
payload = {
    "schema": "sai-100m-250m-development-mc-stage-v1",
    "status": "submitted",
    "source_commit": values[1],
    "screen": {
        "launcher_job_id": int(values[2]),
        "source_commit": values[3],
        "dispatch": {"path": values[4], "sha256": values[5]},
    },
    "population": {
        "job_id": int(values[6]),
        "aggregate": {"path": values[7], "sha256": values[8]},
    },
    "training_stream": {"path": values[9], "identity_sha256": values[10]},
    "tokenizer": {"path": values[11], "tree_sha256": values[12]},
    "output_root": values[13],
    "families": [
        {
            "family": family,
            "training_job_id": training_job,
            "evaluation_launcher_job_id": launcher,
            "benchmarks": ["mmlu_pro", "musr"],
        }
        for family, training_job, launcher in zip(
            families, training_jobs, launchers, strict=True
        )
    ],
    "comparison_stage_job_id": int(values[14]),
    "gpu_jobs_submitted_directly": 0,
    "expected_evaluation_gpu_jobs": 6,
    "gpus_per_evaluation_job": 1,
    "arrays": 0,
    "retry": False,
    "requeue": False,
    "development_only": True,
    "scientific_promotion_authorized": False,
    "four_b_training_authorized": False,
}
payload["receipt_sha256"] = canonical_sha256(payload)
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, output)
    temporary.unlink()
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
PY

trap - EXIT
printf '%s\n' "family_launchers=${family_launchers[*]} comparison_stage=$comparison_stage_id"
