#!/bin/bash
#SBATCH --job-name=sai-stage-curriculum-mc
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
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${GEOMETRY:?GEOMETRY is required}"
: "${ORDER_COMPARISON_JOB_ID:?ORDER_COMPARISON_JOB_ID is required}"
: "${ORDER_COMPARISON:?ORDER_COMPARISON is required}"
: "${CURRICULUM_JOB_ID:?CURRICULUM_JOB_ID is required}"
: "${CONTROL_JOB_ID:?CONTROL_JOB_ID is required}"
: "${CURRICULUM_RESULT:?CURRICULUM_RESULT is required}"
: "${CONTROL_RESULT:?CONTROL_RESULT is required}"
: "${CURRICULUM_CHECKPOINT:?CURRICULUM_CHECKPOINT is required}"
: "${CONTROL_CHECKPOINT:?CONTROL_CHECKPOINT is required}"
: "${CURRICULUM_STREAM:?CURRICULUM_STREAM is required}"
: "${CONTROL_STREAM:?CONTROL_STREAM is required}"
: "${TOKENIZER_ROOT:?TOKENIZER_ROOT is required}"
: "${POPULATION_JOB_ID:?POPULATION_JOB_ID is required}"
: "${POPULATION_AGGREGATE:?POPULATION_AGGREGATE is required}"
: "${CURRICULUM_SPLIT_JOB_ID:?CURRICULUM_SPLIT_JOB_ID is required}"
: "${CURRICULUM_SPLIT_RECEIPT:?CURRICULUM_SPLIT_RECEIPT is required}"
: "${CURRICULUM_SPLIT_RECEIPT_SHA256:?CURRICULUM_SPLIT_RECEIPT_SHA256 is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

FAMILY_LAUNCHER="$SAI_ROOT/jobs/sai-launch-short-screen-development-mc-cpu.sbatch"
COMPARISON_STAGER="$SAI_ROOT/jobs/sai-stage-curriculum-development-mc-comparison-cpu.sbatch"
DISPATCH="$OUTPUT_ROOT/dispatch.json"

require_absolute() {
  case "$1" in /*) ;; *) echo "path must be absolute: $1" >&2; exit 2 ;; esac
}

for value in "$ORDER_COMPARISON_JOB_ID" "$CURRICULUM_JOB_ID" \
  "$CONTROL_JOB_ID" "$POPULATION_JOB_ID" "$CURRICULUM_SPLIT_JOB_ID"; do
  [[ "$value" =~ ^[0-9]+$ ]]
done
[[ "$CURRICULUM_SPLIT_RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]]
for path in "$SAI_ROOT" "$PYTHON" "$FLA_ROOT" "$ENVIRONMENT_RECEIPT" \
  "$LOG_ROOT" "$GEOMETRY" "$ORDER_COMPARISON" "$CURRICULUM_RESULT" \
  "$CONTROL_RESULT" "$CURRICULUM_CHECKPOINT" "$CONTROL_CHECKPOINT" \
  "$CURRICULUM_STREAM" "$CONTROL_STREAM" "$TOKENIZER_ROOT" \
  "$POPULATION_AGGREGATE" "$CURRICULUM_SPLIT_RECEIPT" "$OUTPUT_ROOT" \
  "$FAMILY_LAUNCHER" "$COMPARISON_STAGER" "$DISPATCH"; do
  require_absolute "$path"
done

test -n "${SLURM_JOB_ID:-}"
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU curriculum benchmark stager was exposed to a GPU" >&2; exit 2 ;;
esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"
test -d "$CURRICULUM_STREAM" && test ! -L "$CURRICULUM_STREAM"
test -d "$CONTROL_STREAM" && test ! -L "$CONTROL_STREAM"
test -d "$TOKENIZER_ROOT" && test ! -L "$TOKENIZER_ROOT"
for path in "$ENVIRONMENT_RECEIPT" "$GEOMETRY" "$ORDER_COMPARISON" \
  "$CURRICULUM_RESULT" "$CONTROL_RESULT" "$CURRICULUM_CHECKPOINT" \
  "$CURRICULUM_CHECKPOINT.manifest.json" "$CONTROL_CHECKPOINT" \
  "$CONTROL_CHECKPOINT.manifest.json" "$POPULATION_AGGREGATE" \
  "$CURRICULUM_SPLIT_RECEIPT" "$FAMILY_LAUNCHER" "$COMPARISON_STAGER"; do
  test -f "$path" && test ! -L "$path"
done
test -d "$(dirname "$OUTPUT_ROOT")"
test ! -e "$OUTPUT_ROOT" && test ! -L "$OUTPUT_ROOT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

for job_id in "$ORDER_COMPARISON_JOB_ID" "$CURRICULUM_JOB_ID" \
  "$CONTROL_JOB_ID" "$POPULATION_JOB_ID" "$CURRICULUM_SPLIT_JOB_ID"; do
  row="$(sacct -X -j "$job_id" -n -P -o State,ExitCode,Restarts | sed -n '1p')"
  test "$row" = "COMPLETED|0:0|0"
done

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

ORDER_COMPARISON_SHA256="$($PYTHON -P -s -B - \
  "$ORDER_COMPARISON" "$CURRICULUM_RESULT" "$CONTROL_RESULT" \
  "$CURRICULUM_CHECKPOINT" "$CONTROL_CHECKPOINT" <<'PY'
import hashlib
import json
import pathlib
import sys

comparison_path, curriculum_result, control_result, curriculum_checkpoint, control_checkpoint = map(pathlib.Path, sys.argv[1:])
payload = json.loads(comparison_path.read_text())
unsigned = dict(payload)
claimed = unsigned.pop("receipt_sha256")
canonical = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
assert payload["schema"] == "sai-curriculum-order-training-comparison-v1"
assert payload["status"] == "complete"
assert claimed == canonical
assert payload["curriculum_order_supported_by_heldout_nll"] is True
assert payload["heldout_phase_no_regression"] is True
assert payload["real_benchmark_gate_required"] is True
assert payload["scientific_promotion_authorized"] is False
assert payload["four_b_training_authorized"] is False
for arm, result_path, checkpoint_path in (
    ("curriculum", curriculum_result, curriculum_checkpoint),
    ("order_control", control_result, control_checkpoint),
):
    assert result_path.is_file() and not result_path.is_symlink()
    assert checkpoint_path.is_file() and not checkpoint_path.is_symlink()
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert payload["arms"][arm]["result_file_sha256"] == result_sha
    assert payload["arms"][arm]["checkpoint_sha256"] == checkpoint_sha
print(hashlib.sha256(comparison_path.read_bytes()).hexdigest())
PY
)"
[[ "$ORDER_COMPARISON_SHA256" =~ ^[0-9a-f]{64}$ ]]

mkdir "$OUTPUT_ROOT"
submitted=()
cancel_partial_graph() {
  for job_id in "${submitted[@]}"; do scancel "$job_id" || true; done
}
trap cancel_partial_graph EXIT

arm_launchers=()
for arm in curriculum order_control; do
  if test "$arm" = curriculum; then
    training_job="$CURRICULUM_JOB_ID"
    result="$CURRICULUM_RESULT"
    checkpoint="$CURRICULUM_CHECKPOINT"
    stream="$CURRICULUM_STREAM"
  else
    training_job="$CONTROL_JOB_ID"
    result="$CONTROL_RESULT"
    checkpoint="$CONTROL_CHECKPOINT"
    stream="$CONTROL_STREAM"
  fi
  arm_root="$OUTPUT_ROOT/$arm/gated_gqa"
  mkdir -p "$arm_root"
  launcher_id="$(env -i PATH=/apps/slurm/current/bin:/usr/bin:/bin sbatch --parsable \
    --chdir="$SAI_ROOT" --job-name="sai-${arm}-dev-mc" \
    --output="$LOG_ROOT/sai_${arm}_dev_mc_stage_%j.out" \
    --error="$LOG_ROOT/sai_${arm}_dev_mc_stage_%j.err" \
    --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FLA_ROOT="$FLA_ROOT",ENVIRONMENT_RECEIPT="$ENVIRONMENT_RECEIPT",LOG_ROOT="$LOG_ROOT",GEOMETRY="$GEOMETRY",FAMILY=gated_gqa,SHORT_SCREEN_JOB_ID="$training_job",SHORT_SCREEN_RESULT="$result",CHECKPOINT="$checkpoint",CHECKPOINT_MANIFEST="$checkpoint.manifest.json",TRAINING_STREAM="$stream",TOKENIZER_ROOT="$TOKENIZER_ROOT",POPULATION_JOB_ID="$POPULATION_JOB_ID",POPULATION_AGGREGATE="$POPULATION_AGGREGATE",CURRICULUM_SPLIT_JOB_ID="$CURRICULUM_SPLIT_JOB_ID",CURRICULUM_SPLIT_RECEIPT="$CURRICULUM_SPLIT_RECEIPT",CURRICULUM_SPLIT_RECEIPT_SHA256="$CURRICULUM_SPLIT_RECEIPT_SHA256",OUTPUT_ROOT="$arm_root",DISPATCH="$arm_root/dispatch.json" \
    "$FAMILY_LAUNCHER")"
  [[ "$launcher_id" =~ ^[0-9]+$ ]]
  submitted+=("$launcher_id")
  arm_launchers+=("$launcher_id")
done

launcher_dependency="$(IFS=:; printf '%s' "${arm_launchers[*]}")"
comparison_stage_id="$(env -i PATH=/apps/slurm/current/bin:/usr/bin:/bin sbatch --parsable \
  --chdir="$SAI_ROOT" --dependency="afterok:$launcher_dependency" \
  --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/sai_stage_curriculum_board_compare_%j.out" \
  --error="$LOG_ROOT/sai_stage_curriculum_board_compare_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",LOG_ROOT="$LOG_ROOT",OUTPUT_ROOT="$OUTPUT_ROOT",ORDER_COMPARISON_JOB_ID="$ORDER_COMPARISON_JOB_ID",ORDER_COMPARISON="$ORDER_COMPARISON",CURRICULUM_LAUNCH_JOB_ID="${arm_launchers[0]}",CONTROL_LAUNCH_JOB_ID="${arm_launchers[1]}" \
  "$COMPARISON_STAGER")"
[[ "$comparison_stage_id" =~ ^[0-9]+$ ]]
submitted+=("$comparison_stage_id")

"$PYTHON" -P -s -B - "$DISPATCH" "$EXPECTED_COMMIT" \
  "$ORDER_COMPARISON_JOB_ID" "$ORDER_COMPARISON" "$ORDER_COMPARISON_SHA256" \
  "$CURRICULUM_JOB_ID" "$CONTROL_JOB_ID" "$POPULATION_JOB_ID" \
  "$CURRICULUM_SPLIT_JOB_ID" "${arm_launchers[0]}" "${arm_launchers[1]}" \
  "$comparison_stage_id" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import uuid

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sai-curriculum-development-mc-stage-v1",
    "status": "submitted",
    "source_commit": sys.argv[2],
    "order_comparison": {"job_id": int(sys.argv[3]), "path": sys.argv[4], "sha256": sys.argv[5]},
    "training_job_ids": {"curriculum": int(sys.argv[6]), "order_control": int(sys.argv[7])},
    "population_job_id": int(sys.argv[8]),
    "curriculum_split_job_id": int(sys.argv[9]),
    "evaluation_launcher_job_ids": {"curriculum": int(sys.argv[10]), "order_control": int(sys.argv[11])},
    "comparison_stage_job_id": int(sys.argv[12]),
    "expected_evaluation_gpu_jobs": 18,
    "expected_evaluation_cpu_merges": 2,
    "maximum_concurrent_single_h100_jobs": 18,
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

trap - EXIT
printf 'curriculum_launcher=%s control_launcher=%s comparison_stage=%s\n' \
  "${arm_launchers[0]}" "${arm_launchers[1]}" "$comparison_stage_id"
