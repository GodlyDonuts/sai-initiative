#!/bin/bash
#SBATCH --job-name=sai-stage-curriculum-continuation
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${ENVIRONMENT_RECEIPT:?ENVIRONMENT_RECEIPT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${GEOMETRY:?GEOMETRY is required}"
: "${SCREEN_LAUNCH_JOB_ID:?SCREEN_LAUNCH_JOB_ID is required}"
: "${SCREEN_ROOT:?SCREEN_ROOT is required}"
: "${CURRICULUM_STREAM:?CURRICULUM_STREAM is required}"
: "${CONTROL_STREAM:?CONTROL_STREAM is required}"
: "${TOKENIZER_ROOT:?TOKENIZER_ROOT is required}"
: "${POPULATION_JOB_ID:?POPULATION_JOB_ID is required}"
: "${POPULATION_AGGREGATE:?POPULATION_AGGREGATE is required}"
: "${CURRICULUM_SPLIT_JOB_ID:?CURRICULUM_SPLIT_JOB_ID is required}"
: "${CURRICULUM_SPLIT_RECEIPT:?CURRICULUM_SPLIT_RECEIPT is required}"
: "${CURRICULUM_SPLIT_RECEIPT_SHA256:?CURRICULUM_SPLIT_RECEIPT_SHA256 is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

STAGER="$SAI_ROOT/jobs/sai-stage-curriculum-development-mc-cpu.sbatch"
SCREEN_DISPATCH="$SCREEN_ROOT/dispatch.json"
DISPATCH="$OUTPUT_ROOT.continuation.dispatch.json"

for value in "$SCREEN_LAUNCH_JOB_ID" "$POPULATION_JOB_ID" \
  "$CURRICULUM_SPLIT_JOB_ID"; do [[ "$value" =~ ^[0-9]+$ ]]; done
[[ "$CURRICULUM_SPLIT_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]]
for path in "$SAI_ROOT" "$PYTHON" "$FLA_ROOT" "$ENVIRONMENT_RECEIPT" \
  "$LOG_ROOT" "$GEOMETRY" "$SCREEN_ROOT" "$SCREEN_DISPATCH" \
  "$CURRICULUM_STREAM" "$CONTROL_STREAM" "$TOKENIZER_ROOT" \
  "$POPULATION_AGGREGATE" "$CURRICULUM_SPLIT_RECEIPT" "$OUTPUT_ROOT" \
  "$STAGER" "$DISPATCH"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done

case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"
test -d "$SCREEN_ROOT" && test ! -L "$SCREEN_ROOT"
test -d "$CURRICULUM_STREAM" && test ! -L "$CURRICULUM_STREAM"
test -d "$CONTROL_STREAM" && test ! -L "$CONTROL_STREAM"
test -d "$TOKENIZER_ROOT" && test ! -L "$TOKENIZER_ROOT"
for path in "$ENVIRONMENT_RECEIPT" "$GEOMETRY" "$SCREEN_DISPATCH" \
  "$POPULATION_AGGREGATE" "$CURRICULUM_SPLIT_RECEIPT" "$STAGER"; do
  test -f "$path" && test ! -L "$path"
done
test ! -e "$OUTPUT_ROOT" && test ! -L "$OUTPUT_ROOT"
test ! -e "$DISPATCH" && test ! -L "$DISPATCH"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
for job_id in "$SCREEN_LAUNCH_JOB_ID" "$POPULATION_JOB_ID" \
  "$CURRICULUM_SPLIT_JOB_ID"; do
  row="$(sacct -X -j "$job_id" -n -P -o State,ExitCode,Restarts | sed -n '1p')"
  test "$row" = "COMPLETED|0:0|0"
done

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

validated_lines="$("$PYTHON" -P -s -B - "$SCREEN_DISPATCH" "$SCREEN_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

dispatch_path = pathlib.Path(sys.argv[1]); root = pathlib.Path(sys.argv[2])
payload = json.loads(dispatch_path.read_text())
assert set(payload) == {
    "schema", "status", "source_commit", "curriculum_stream_identity_sha256",
    "control_stream_identity_sha256", "development_stream_identity_sha256",
    "canary_job_id", "curriculum_job_id", "control_job_id", "comparison_job_id",
    "gpu_jobs_submitted", "maximum_concurrent_gpu_jobs", "gpus_per_job",
    "same_sequence_multiset", "only_training_order_changed",
    "training_sequences_per_arm", "training_tokens_per_arm",
    "optimizer_steps_per_arm", "seed", "scientific_promotion_authorized",
    "four_b_training_authorized", "requeue",
}
assert payload["schema"] == "sai-curriculum-order-screen-dispatch-v1"
assert payload["status"] == "submitted"
assert payload["gpu_jobs_submitted"] == 3
assert payload["maximum_concurrent_gpu_jobs"] == 2
assert payload["same_sequence_multiset"] is True
assert payload["only_training_order_changed"] is True
assert payload["training_sequences_per_arm"] == 244_140
assert payload["training_tokens_per_arm"] == 499_998_720
assert payload["optimizer_steps_per_arm"] == 954
assert payload["scientific_promotion_authorized"] is False
assert payload["four_b_training_authorized"] is False
assert payload["requeue"] is False
for key in ("canary_job_id", "curriculum_job_id", "control_job_id", "comparison_job_id"):
    assert isinstance(payload[key], int) and payload[key] > 0
paths = (
    root / "curriculum.result.json", root / "order_control.result.json",
    root / "curriculum.checkpoint.pt", root / "order_control.checkpoint.pt",
    root / "comparison.json",
)
for path in paths:
    assert not path.is_symlink()
print(payload["curriculum_job_id"])
print(payload["control_job_id"])
print(payload["comparison_job_id"])
print(*(str(path) for path in paths), sep="\n")
print(hashlib.sha256(dispatch_path.read_bytes()).hexdigest())
PY
)"
mapfile -t values <<< "$validated_lines"
test "${#values[@]}" = 9
curriculum_job="${values[0]}"
control_job="${values[1]}"
comparison_job="${values[2]}"
curriculum_result="${values[3]}"
control_result="${values[4]}"
curriculum_checkpoint="${values[5]}"
control_checkpoint="${values[6]}"
comparison_result="${values[7]}"
screen_dispatch_sha256="${values[8]}"
for value in "$curriculum_job" "$control_job" "$comparison_job"; do
  [[ "$value" =~ ^[0-9]+$ ]]
done
[[ "$screen_dispatch_sha256" =~ ^[0-9a-f]{64}$ ]]

dependency="$comparison_job:$POPULATION_JOB_ID"
stage_job_id="$(env -i PATH=/apps/slurm/current/bin:/usr/bin:/bin sbatch --parsable \
  --chdir="$SAI_ROOT" --dependency="afterok:$dependency" \
  --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/sai_stage_curriculum_benchmarks_%j.out" \
  --error="$LOG_ROOT/sai_stage_curriculum_benchmarks_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FLA_ROOT="$FLA_ROOT",ENVIRONMENT_RECEIPT="$ENVIRONMENT_RECEIPT",LOG_ROOT="$LOG_ROOT",GEOMETRY="$GEOMETRY",ORDER_COMPARISON_JOB_ID="$comparison_job",ORDER_COMPARISON="$comparison_result",CURRICULUM_JOB_ID="$curriculum_job",CONTROL_JOB_ID="$control_job",CURRICULUM_RESULT="$curriculum_result",CONTROL_RESULT="$control_result",CURRICULUM_CHECKPOINT="$curriculum_checkpoint",CONTROL_CHECKPOINT="$control_checkpoint",CURRICULUM_STREAM="$CURRICULUM_STREAM",CONTROL_STREAM="$CONTROL_STREAM",TOKENIZER_ROOT="$TOKENIZER_ROOT",POPULATION_JOB_ID="$POPULATION_JOB_ID",POPULATION_AGGREGATE="$POPULATION_AGGREGATE",CURRICULUM_SPLIT_JOB_ID="$CURRICULUM_SPLIT_JOB_ID",CURRICULUM_SPLIT_RECEIPT="$CURRICULUM_SPLIT_RECEIPT",CURRICULUM_SPLIT_RECEIPT_SHA256="$CURRICULUM_SPLIT_RECEIPT_SHA256",OUTPUT_ROOT="$OUTPUT_ROOT" \
  "$STAGER")"
[[ "$stage_job_id" =~ ^[0-9]+$ ]]

"$PYTHON" -P -s -B - "$DISPATCH" "$EXPECTED_COMMIT" \
  "$SCREEN_LAUNCH_JOB_ID" "$SCREEN_DISPATCH" "$screen_dispatch_sha256" \
  "$comparison_job" "$POPULATION_JOB_ID" "$stage_job_id" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import uuid

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sai-curriculum-benchmark-continuation-dispatch-v1",
    "status": "submitted",
    "source_commit": sys.argv[2],
    "screen_launcher_job_id": int(sys.argv[3]),
    "screen_dispatch": {"path": sys.argv[4], "sha256": sys.argv[5]},
    "order_comparison_job_id": int(sys.argv[6]),
    "population_job_id": int(sys.argv[7]),
    "benchmark_stage_job_id": int(sys.argv[8]),
    "benchmark_work_condition": "heldout_nll_and_every_phase_nonregression",
    "gpu_jobs_submitted": 0,
    "development_only": True,
    "architecture_promotion_authorized": False,
    "four_b_training_authorized": False,
    "retry": False,
    "requeue": False,
}
payload["receipt_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    os.link(temporary, path); temporary.unlink()
except BaseException:
    temporary.unlink(missing_ok=True); raise
PY

printf 'benchmark_stage=%s\n' "$stage_job_id"
