#!/usr/bin/env bash
#SBATCH --job-name=sai-release-q08-compare
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
: "${PARENT_ROOT:?PARENT_ROOT is required}"
: "${WORKSPACE_RESULT_ROOT:?WORKSPACE_RESULT_ROOT is required}"
: "${TRAINING_ROOT:?TRAINING_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -f "$PARENT_ROOT/dispatch.json"
test -f "$WORKSPACE_RESULT_ROOT/recurrent/dispatch.json"
test -f "$WORKSPACE_RESULT_ROOT/reset_average/dispatch.json"
test -d "$LOG_ROOT"
test ! -e "$OUTPUT"
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac

read -r parent_musr parent_merge recurrent_musr recurrent_merge reset_musr reset_merge < <(
  "$PYTHON" -P -s -B - <<'PY' \
    "$PARENT_ROOT/dispatch.json" \
    "$WORKSPACE_RESULT_ROOT/recurrent/dispatch.json" \
    "$WORKSPACE_RESULT_ROOT/reset_average/dispatch.json"
import hashlib, json, sys

def load(path, schema, mode=None):
    value = json.load(open(path))
    unsigned = dict(value); claimed = unsigned.pop("receipt_sha256", None)
    observed = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if value.get("schema") != schema or value.get("status") != "submitted" or claimed != observed:
        raise SystemExit("development dispatch differs")
    if mode is not None and value.get("state_mode") != mode:
        raise SystemExit("workspace development mode differs")
    return value

parent = load(sys.argv[1], "sai-qwen35-0p8b-development-mc-dispatch-v1")
recurrent = load(sys.argv[2], "sai-qwen35-0p8b-workspace-development-mc-dispatch-v1", "recurrent")
reset = load(sys.argv[3], "sai-qwen35-0p8b-workspace-development-mc-dispatch-v1", "reset_average")
print(parent["musr_job"], parent["mmlu_merge_job"], recurrent["musr_job"], recurrent["mmlu_merge_job"], reset["musr_job"], reset["mmlu_merge_job"])
PY
)

dependency="$parent_musr:$parent_merge:$recurrent_musr:$recurrent_merge:$reset_musr:$reset_merge"
comparison_job="$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --dependency="afterok:$dependency" --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/sai_compare_q08_workspace_%j.out" \
  --error="$LOG_ROOT/sai_compare_q08_workspace_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",PARENT_MMLU="$PARENT_ROOT/mmlu_pro.json",PARENT_MUSR="$PARENT_ROOT/musr.json",RECURRENT_MMLU="$WORKSPACE_RESULT_ROOT/recurrent/mmlu_pro.json",RECURRENT_MUSR="$WORKSPACE_RESULT_ROOT/recurrent/musr.json",RESET_MMLU="$WORKSPACE_RESULT_ROOT/reset_average/mmlu_pro.json",RESET_MUSR="$WORKSPACE_RESULT_ROOT/reset_average/musr.json",RECURRENT_TRAINING_RESULT="$TRAINING_ROOT/full/recurrent/result.json",RESET_TRAINING_RESULT="$TRAINING_ROOT/full/reset_average/result.json",OUTPUT="$OUTPUT" \
  "$SAI_ROOT/jobs/sai-compare-qwen35-0p8b-workspace-cpu.sbatch")"

"$PYTHON" -P -s -B - <<'PY' "$OUTPUT.dispatch.json" "$EXPECTED_COMMIT" "$comparison_job" "$dependency"
import hashlib, json, os, pathlib, sys, uuid
output = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sai-qwen35-0p8b-workspace-comparison-dispatch-v1",
    "status": "submitted", "source_commit": sys.argv[2],
    "comparison_job": int(sys.argv[3]),
    "result_dependencies": [int(value) for value in sys.argv[4].split(":")],
    "gpu_jobs_submitted": 0, "requeue": False, "four_b_training_executed": False,
}
payload["receipt_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
os.link(temporary, output); temporary.unlink()
PY
printf 'comparison_job=%s dependency=%s\n' "$comparison_job" "$dependency"
