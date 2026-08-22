#!/bin/bash
#SBATCH --job-name=sai-curr-split-500m
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
: "${CURRICULUM_RECEIPT:?CURRICULUM_RECEIPT is required}"
: "${TRAIN_CORPUS:?TRAIN_CORPUS is required}"
: "${DEVELOPMENT_CORPUS:?DEVELOPMENT_CORPUS is required}"
: "${SPLIT_RECEIPT:?SPLIT_RECEIPT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

for path in "$SAI_ROOT" "$PYTHON" "$CURRICULUM_RECEIPT" "$TRAIN_CORPUS" \
  "$DEVELOPMENT_CORPUS" "$SPLIT_RECEIPT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$CURRICULUM_RECEIPT"
test ! -L "$CURRICULUM_RECEIPT"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test ! -e "$TRAIN_CORPUS"
test ! -e "$DEVELOPMENT_CORPUS"
test ! -e "$SPLIT_RECEIPT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -P -s -B -m sai.data.curriculum_split build \
  --curriculum-receipt "$CURRICULUM_RECEIPT" \
  --train "$TRAIN_CORPUS" \
  --development "$DEVELOPMENT_CORPUS" \
  --receipt "$SPLIT_RECEIPT"

"$PYTHON" -P -s -B -m sai.data.curriculum_split validate \
  --receipt "$SPLIT_RECEIPT"
