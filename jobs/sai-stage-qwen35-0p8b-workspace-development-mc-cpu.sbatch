#!/usr/bin/env bash
#SBATCH --job-name=sai-stage-q08-workspace-mc
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
: "${FAST_KERNELS_ROOT:?FAST_KERNELS_ROOT is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MECHANICS_RECEIPT:?MECHANICS_RECEIPT is required}"
: "${TRAINING_DISPATCH_JOB_ID:?TRAINING_DISPATCH_JOB_ID is required}"
: "${TRAINING_ROOT:?TRAINING_ROOT is required}"
: "${POPULATION_JOB_ID:?POPULATION_JOB_ID is required}"
: "${POPULATION_ROOT:?POPULATION_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test "$(sacct -j "$TRAINING_DISPATCH_JOB_ID" -X -n -P -o State | sed -n '1p')" = COMPLETED
test -f "$TRAINING_ROOT/dispatch.json"
test -d "$LOG_ROOT"
test ! -e "$RESULT_ROOT"
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac

read -r recurrent_job reset_job < <(
  "$PYTHON" -P -s -B - <<'PY' "$TRAINING_ROOT/dispatch.json" "$TRAINING_DISPATCH_JOB_ID"
import hashlib, json, sys
payload = json.load(open(sys.argv[1]))
unsigned = dict(payload); claimed = unsigned.pop("receipt_sha256", None)
observed = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if (
    payload.get("schema") != "sai-qwen35-0p8b-matched-workspace-dispatch-v1"
    or payload.get("status") != "submitted"
    or claimed != observed
    or payload.get("h100_jobs") != 4
    or payload.get("full_jobs_require_both_canaries") is not True
    or payload.get("four_b_training_executed") is not False
):
    raise SystemExit("workspace training dispatch differs")
print(payload["full_jobs"]["recurrent"], payload["full_jobs"]["reset_average"])
PY
)

mkdir -m 700 "$RESULT_ROOT"
admitted_jobs=()
cleanup_partial_submission() {
  if ((${#admitted_jobs[@]})); then scancel "${admitted_jobs[@]}" || true; fi
}
trap cleanup_partial_submission ERR

submit_arm_launcher() {
  local mode="$1" training_job="$2"
  env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
    --dependency="afterok:$training_job:$POPULATION_JOB_ID" \
    --kill-on-invalid-dep=yes \
    --job-name="sai-launch-q08-${mode}-mc" \
    --output="$LOG_ROOT/sai_launch_q08_${mode}_mc_%j.out" \
    --error="$LOG_ROOT/sai_launch_q08_${mode}_mc_%j.err" \
    --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FAST_KERNELS_ROOT="$FAST_KERNELS_ROOT",MODEL_ROOT="$MODEL_ROOT",MECHANICS_RECEIPT="$MECHANICS_RECEIPT",TRAINING_JOB_ID="$training_job",TRAINING_RESULT="$TRAINING_ROOT/full/$mode/result.json",CHECKPOINT="$TRAINING_ROOT/full/$mode/checkpoint.pt",POPULATION_JOB_ID="$POPULATION_JOB_ID",POPULATION_ROOT="$POPULATION_ROOT",STATE_MODE="$mode",LOG_ROOT="$LOG_ROOT",RESULT_ROOT="$RESULT_ROOT/$mode" \
    "$SAI_ROOT/jobs/sai-launch-qwen35-0p8b-workspace-development-mc-arm-cpu.sbatch"
}

recurrent_launcher="$(submit_arm_launcher recurrent "$recurrent_job")"
admitted_jobs+=("$recurrent_launcher")
reset_launcher="$(submit_arm_launcher reset_average "$reset_job")"
admitted_jobs+=("$reset_launcher")

"$PYTHON" -P -s -B - <<'PY' "$RESULT_ROOT/dispatch.json" "$EXPECTED_COMMIT" "$TRAINING_DISPATCH_JOB_ID" "$POPULATION_JOB_ID" "$recurrent_job" "$reset_job" "$recurrent_launcher" "$reset_launcher"
import hashlib, json, os, pathlib, sys, uuid
output = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sai-qwen35-0p8b-workspace-development-mc-stage-v1",
    "status": "staged", "source_commit": sys.argv[2],
    "training_dispatch_job": int(sys.argv[3]), "population_job": int(sys.argv[4]),
    "training_jobs": {"recurrent": int(sys.argv[5]), "reset_average": int(sys.argv[6])},
    "evaluation_launcher_jobs": {"recurrent": int(sys.argv[7]), "reset_average": int(sys.argv[8])},
    "eventual_h100_jobs": 18, "maximum_concurrent_h100_jobs": 18,
    "one_h100_per_job": True, "requeue": False, "four_b_training_executed": False,
}
payload["receipt_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
os.link(temporary, output); temporary.unlink()
PY

trap - ERR
printf 'recurrent_launcher=%s reset_launcher=%s\n' "$recurrent_launcher" "$reset_launcher"
