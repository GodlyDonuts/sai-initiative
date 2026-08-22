#!/bin/bash
#SBATCH --job-name=sai-compare-curr-order
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${CURRICULUM_JOB_ID:?CURRICULUM_JOB_ID is required}"
: "${CONTROL_JOB_ID:?CONTROL_JOB_ID is required}"
: "${CURRICULUM_RESULT:?CURRICULUM_RESULT is required}"
: "${CONTROL_RESULT:?CONTROL_RESULT is required}"
: "${CURRICULUM_STREAM:?CURRICULUM_STREAM is required}"
: "${CONTROL_STREAM:?CONTROL_STREAM is required}"
: "${DEVELOPMENT_STREAM:?DEVELOPMENT_STREAM is required}"
: "${SPLIT_RECEIPT:?SPLIT_RECEIPT is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

for path in "$SAI_ROOT" "$PYTHON" "$CURRICULUM_RESULT" "$CONTROL_RESULT" \
  "$CURRICULUM_STREAM" "$CONTROL_STREAM" "$DEVELOPMENT_STREAM" \
  "$SPLIT_RECEIPT" "$OUTPUT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
for path in "$CURRICULUM_RESULT" "$CONTROL_RESULT" "$SPLIT_RECEIPT"; do
  test -f "$path"
  test ! -L "$path"
done
for path in "$CURRICULUM_STREAM" "$CONTROL_STREAM" "$DEVELOPMENT_STREAM"; do
  test -d "$path"
  test ! -L "$path"
done
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test ! -e "$OUTPUT"

for job_id in "$CURRICULUM_JOB_ID" "$CONTROL_JOB_ID"; do
  row="$(sacct -n -X -j "$job_id" -o State,ExitCode,Restarts -P | sed -n '1p')"
  test "$row" = "COMPLETED|0:0|0"
done
job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

exec "$PYTHON" -P -s -B -m sai.evaluation.curriculum_order_compare \
  --curriculum-result "$CURRICULUM_RESULT" \
  --control-result "$CONTROL_RESULT" \
  --curriculum-stream "$CURRICULUM_STREAM" \
  --control-stream "$CONTROL_STREAM" \
  --development-stream "$DEVELOPMENT_STREAM" \
  --split-receipt "$SPLIT_RECEIPT" \
  --curriculum-validation-workers "$SLURM_CPUS_PER_TASK" \
  --output "$OUTPUT"
