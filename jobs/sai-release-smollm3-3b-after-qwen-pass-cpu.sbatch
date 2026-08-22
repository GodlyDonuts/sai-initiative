#!/usr/bin/env bash
#SBATCH --job-name=sai-release-smol3-after-qwen
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FAST_KERNELS_ROOT:?FAST_KERNELS_ROOT is required}"
: "${QWEN_COMPARISON_JOB_ID:?QWEN_COMPARISON_JOB_ID is required}"
: "${QWEN_FACTOR_RECEIPT:?QWEN_FACTOR_RECEIPT is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_MANIFEST:?MODEL_MANIFEST is required}"
: "${RESTORATION_RECEIPT:?RESTORATION_RECEIPT is required}"
: "${STREAM_JOB_ID:?STREAM_JOB_ID is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"
: "${POPULATION_JOB_ID:?POPULATION_JOB_ID is required}"
: "${POPULATION_ROOT:?POPULATION_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test "$(sacct -j "$QWEN_COMPARISON_JOB_ID" -X -n -P -o State | sed -n '1p')" = COMPLETED
test "$(sacct -j "$STREAM_JOB_ID" -X -n -P -o State | sed -n '1p')" = COMPLETED
test "$(sacct -j "$POPULATION_JOB_ID" -X -n -P -o State | sed -n '1p')" = COMPLETED
test -f "$QWEN_FACTOR_RECEIPT"
test ! -L "$QWEN_FACTOR_RECEIPT"
test -d "$MODEL_ROOT"
test -f "$MODEL_MANIFEST"
test ! -L "$MODEL_MANIFEST"
test -f "$RESTORATION_RECEIPT"
test ! -L "$RESTORATION_RECEIPT"
test -d "$TRAIN_STREAM"
test -d "$POPULATION_ROOT"
test -d "$FAST_KERNELS_ROOT"
test -d "$LOG_ROOT"
test ! -e "$RUN_ROOT"
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
read -r qwen_file_sha qwen_receipt_sha < <(
  "$PYTHON" -P -s -B - <<'PY' "$QWEN_FACTOR_RECEIPT"
import pathlib
import sys

from sai.evaluation.hf_smol_workspace_compare import _load_qwen_factor_receipt

receipt, file_sha = _load_qwen_factor_receipt(pathlib.Path(sys.argv[1]))
print(file_sha, receipt["receipt_sha256"])
PY
)

mkdir -m 700 "$RUN_ROOT"
mechanics_receipt="$RUN_ROOT/mechanics.json"
training_root="$RUN_ROOT/training"
parent_root="$RUN_ROOT/parent_evaluation"
workspace_result_root="$RUN_ROOT/workspace_evaluation"
comparison_output="$RUN_ROOT/comparison.json"
for path in \
  "$mechanics_receipt" "$training_root" "$parent_root" \
  "$workspace_result_root" "$comparison_output"; do
  test ! -e "$path"
  test ! -L "$path"
done

admitted_jobs=()
cleanup_partial_submission() {
  if ((${#admitted_jobs[@]})); then scancel "${admitted_jobs[@]}" || true; fi
}
trap cleanup_partial_submission ERR

mechanics_job="$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --output="$LOG_ROOT/sai_smol3_mechanics_%j.out" \
  --error="$LOG_ROOT/sai_smol3_mechanics_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FAST_KERNELS_ROOT="$FAST_KERNELS_ROOT",MODEL_ROOT="$MODEL_ROOT",MODEL_MANIFEST="$MODEL_MANIFEST",RESTORATION_RECEIPT="$RESTORATION_RECEIPT",OUTPUT="$mechanics_receipt" \
  "$SAI_ROOT/jobs/sai-smollm3-3b-mechanics-single-h100.sbatch")"
admitted_jobs+=("$mechanics_job")

parent_launcher="$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --dependency="afterok:$mechanics_job:$POPULATION_JOB_ID" \
  --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/sai_launch_smol3_parent_mc_%j.out" \
  --error="$LOG_ROOT/sai_launch_smol3_parent_mc_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FAST_KERNELS_ROOT="$FAST_KERNELS_ROOT",MODEL_ROOT="$MODEL_ROOT",MODEL_MANIFEST="$MODEL_MANIFEST",RESTORATION_RECEIPT="$RESTORATION_RECEIPT",MECHANICS_JOB_ID="$mechanics_job",MECHANICS_RECEIPT="$mechanics_receipt",POPULATION_JOB_ID="$POPULATION_JOB_ID",POPULATION_ROOT="$POPULATION_ROOT",LOG_ROOT="$LOG_ROOT",RESULT_ROOT="$parent_root" \
  "$SAI_ROOT/jobs/sai-launch-smollm3-3b-development-mc-cpu.sbatch")"
admitted_jobs+=("$parent_launcher")

training_launcher="$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --dependency="afterok:$mechanics_job:$STREAM_JOB_ID" \
  --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/sai_launch_smol3_workspace_%j.out" \
  --error="$LOG_ROOT/sai_launch_smol3_workspace_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FAST_KERNELS_ROOT="$FAST_KERNELS_ROOT",MODEL_ROOT="$MODEL_ROOT",MODEL_MANIFEST="$MODEL_MANIFEST",RESTORATION_RECEIPT="$RESTORATION_RECEIPT",MECHANICS_JOB_ID="$mechanics_job",MECHANICS_RECEIPT="$mechanics_receipt",STREAM_JOB_ID="$STREAM_JOB_ID",TRAIN_STREAM="$TRAIN_STREAM",LOG_ROOT="$LOG_ROOT",RUN_ROOT="$training_root" \
  "$SAI_ROOT/jobs/sai-launch-smollm3-3b-workspace-screen-cpu.sbatch")"
admitted_jobs+=("$training_launcher")

workspace_stage="$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --dependency="afterok:$training_launcher" --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/sai_stage_smol3_workspace_mc_%j.out" \
  --error="$LOG_ROOT/sai_stage_smol3_workspace_mc_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FAST_KERNELS_ROOT="$FAST_KERNELS_ROOT",MODEL_ROOT="$MODEL_ROOT",MODEL_MANIFEST="$MODEL_MANIFEST",RESTORATION_RECEIPT="$RESTORATION_RECEIPT",MECHANICS_RECEIPT="$mechanics_receipt",TRAINING_DISPATCH_JOB_ID="$training_launcher",TRAINING_ROOT="$training_root",POPULATION_JOB_ID="$POPULATION_JOB_ID",POPULATION_ROOT="$POPULATION_ROOT",LOG_ROOT="$LOG_ROOT",RESULT_ROOT="$workspace_result_root" \
  "$SAI_ROOT/jobs/sai-stage-smollm3-3b-workspace-development-mc-cpu.sbatch")"
admitted_jobs+=("$workspace_stage")

comparison_stage="$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --dependency="afterok:$parent_launcher:$workspace_stage" \
  --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/sai_stage_smol3_compare_%j.out" \
  --error="$LOG_ROOT/sai_stage_smol3_compare_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",PARENT_LAUNCHER_JOB_ID="$parent_launcher",WORKSPACE_STAGE_JOB_ID="$workspace_stage",PARENT_ROOT="$parent_root",WORKSPACE_RESULT_ROOT="$workspace_result_root",TRAINING_ROOT="$training_root",QWEN_FACTOR_RECEIPT="$QWEN_FACTOR_RECEIPT",LOG_ROOT="$LOG_ROOT",OUTPUT="$comparison_output" \
  "$SAI_ROOT/jobs/sai-stage-smollm3-3b-workspace-comparison-cpu.sbatch")"
admitted_jobs+=("$comparison_stage")

"$PYTHON" -P -s -B - <<'PY' \
  "$RUN_ROOT/dispatch.json" "$EXPECTED_COMMIT" "$QWEN_COMPARISON_JOB_ID" \
  "$qwen_file_sha" "$qwen_receipt_sha" "$STREAM_JOB_ID" "$POPULATION_JOB_ID" \
  "$mechanics_job" "$parent_launcher" "$training_launcher" \
  "$workspace_stage" "$comparison_stage"
import hashlib
import json
import os
import pathlib
import sys
import uuid

output = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sai-smollm3-3b-cross-family-release-dispatch-v1",
    "status": "submitted_after_qwen_pass",
    "source_commit": sys.argv[2],
    "qwen_comparison_job": int(sys.argv[3]),
    "qwen_factor_receipt_file_sha256": sys.argv[4],
    "qwen_factor_receipt_sha256": sys.argv[5],
    "stream_job": int(sys.argv[6]),
    "population_job": int(sys.argv[7]),
    "top_level_jobs": {
        "mechanics": int(sys.argv[8]),
        "parent_evaluation_launcher": int(sys.argv[9]),
        "workspace_training_launcher": int(sys.argv[10]),
        "workspace_evaluation_stage": int(sys.argv[11]),
        "comparison_stage": int(sys.argv[12]),
    },
    "eventual_h100_jobs": 32,
    "maximum_concurrent_h100_jobs": 27,
    "one_h100_per_job": True,
    "cross_family_confirmation": True,
    "requeue": False,
    "four_b_training_executed": False,
    "four_b_training_authorized": False,
}
payload["receipt_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
os.link(temporary, output)
temporary.unlink()
PY

trap - ERR
printf 'mechanics=%s parent=%s training=%s workspace_eval=%s compare=%s\n' \
  "$mechanics_job" "$parent_launcher" "$training_launcher" \
  "$workspace_stage" "$comparison_stage"
