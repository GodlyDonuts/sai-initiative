#!/bin/bash
#SBATCH --job-name=sai-curriculum-control
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${PARENT_STREAM_JOB_ID:?PARENT_STREAM_JOB_ID is required}"
: "${PARENT_STREAM:?PARENT_STREAM is required}"
: "${CONTROL_STREAM:?CONTROL_STREAM is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

for path in "$SAI_ROOT" "$PYTHON" "$PARENT_STREAM" "$CONTROL_STREAM" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU curriculum control was exposed to a GPU" >&2; exit 2 ;;
esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$PARENT_STREAM"
test ! -L "$PARENT_STREAM"
test -f "$PARENT_STREAM/stream_receipt.json"
test ! -L "$PARENT_STREAM/stream_receipt.json"
test ! -e "$CONTROL_STREAM"
test ! -L "$CONTROL_STREAM"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
state="$(sacct -X -j "$PARENT_STREAM_JOB_ID" -n -P -o State | sed -n '1p')"
test "$state" = "COMPLETED"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$PYTHON" -P -s -B -m sai.data.curriculum_control build \
  --source "$PARENT_STREAM" \
  --output "$CONTROL_STREAM" \
  --seed 2026082201

"$PYTHON" -P -s -B -m sai.data.curriculum_control validate \
  --output "$CONTROL_STREAM"
