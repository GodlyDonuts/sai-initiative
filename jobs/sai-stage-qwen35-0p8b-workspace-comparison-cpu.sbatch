#!/usr/bin/env bash
#SBATCH --job-name=sai-stage-q08-compare
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${PARENT_LAUNCHER_JOB_ID:?PARENT_LAUNCHER_JOB_ID is required}"
: "${WORKSPACE_STAGE_JOB_ID:?WORKSPACE_STAGE_JOB_ID is required}"
: "${PARENT_ROOT:?PARENT_ROOT is required}"
: "${WORKSPACE_RESULT_ROOT:?WORKSPACE_RESULT_ROOT is required}"
: "${TRAINING_ROOT:?TRAINING_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test "$(sacct -j "$PARENT_LAUNCHER_JOB_ID" -X -n -P -o State | sed -n '1p')" = COMPLETED
test "$(sacct -j "$WORKSPACE_STAGE_JOB_ID" -X -n -P -o State | sed -n '1p')" = COMPLETED
test -f "$WORKSPACE_RESULT_ROOT/dispatch.json"
test ! -e "$OUTPUT.dispatch.json"
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac

read -r recurrent_launcher reset_launcher < <(
  "$PYTHON" -P -s -B - <<'PY' "$WORKSPACE_RESULT_ROOT/dispatch.json"
import hashlib, json, sys
value = json.load(open(sys.argv[1]))
unsigned = dict(value); claimed = unsigned.pop("receipt_sha256", None)
observed = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if (
    value.get("schema") != "sai-qwen35-0p8b-workspace-development-mc-stage-v1"
    or value.get("status") != "staged" or claimed != observed
    or value.get("eventual_h100_jobs") != 18
    or value.get("four_b_training_executed") is not False
):
    raise SystemExit("workspace evaluation stage differs")
print(value["evaluation_launcher_jobs"]["recurrent"], value["evaluation_launcher_jobs"]["reset_average"])
PY
)

release_job="$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --dependency="afterok:$PARENT_LAUNCHER_JOB_ID:$recurrent_launcher:$reset_launcher" \
  --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/sai_release_q08_workspace_compare_%j.out" \
  --error="$LOG_ROOT/sai_release_q08_workspace_compare_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",PARENT_ROOT="$PARENT_ROOT",WORKSPACE_RESULT_ROOT="$WORKSPACE_RESULT_ROOT",TRAINING_ROOT="$TRAINING_ROOT",LOG_ROOT="$LOG_ROOT",OUTPUT="$OUTPUT" \
  "$SAI_ROOT/jobs/sai-release-qwen35-0p8b-workspace-comparison-cpu.sbatch")"
printf 'release_job=%s recurrent_launcher=%s reset_launcher=%s\n' "$release_job" "$recurrent_launcher" "$reset_launcher"
