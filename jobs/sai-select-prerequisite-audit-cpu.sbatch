#!/bin/bash
#SBATCH --job-name=sai-prereq-audit-pop
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${CURRICULUM_RECEIPT:?CURRICULUM_RECEIPT is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${RECEIPT:?RECEIPT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in "$SAI_ROOT" "$PYTHON" "$CURRICULUM_RECEIPT" "$OUTPUT" "$RECEIPT" "$LOG_ROOT"; do
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
test ! -e "$OUTPUT"
test ! -e "$RECEIPT"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -P -s -B -m sai.data.prerequisite_sample build \
  --curriculum-receipt "$CURRICULUM_RECEIPT" \
  --output "$OUTPUT" \
  --receipt "$RECEIPT" \
  --per-stratum 8 \
  --curriculum-workers "$SLURM_CPUS_PER_TASK"

"$PYTHON" -P -s -B -m sai.data.prerequisite_sample validate \
  --receipt "$RECEIPT" \
  --curriculum-workers "$SLURM_CPUS_PER_TASK"

test -f "$OUTPUT"
test -f "$RECEIPT"
test ! -L "$OUTPUT"
test ! -L "$RECEIPT"
