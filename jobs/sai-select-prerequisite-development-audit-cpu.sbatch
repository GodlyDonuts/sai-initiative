#!/bin/bash
#SBATCH --job-name=sai-prereq-dev-audit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${SPLIT_RECEIPT:?SPLIT_RECEIPT is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${RECEIPT:?RECEIPT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in "$SAI_ROOT" "$PYTHON" "$SPLIT_RECEIPT" "$OUTPUT" "$RECEIPT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$SPLIT_RECEIPT" && test ! -L "$SPLIT_RECEIPT"
test ! -e "$OUTPUT" && test ! -L "$OUTPUT"
test ! -e "$RECEIPT" && test ! -L "$RECEIPT"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -P -s -B -m sai.data.prerequisite_development_sample build \
  --split-receipt "$SPLIT_RECEIPT" \
  --output "$OUTPUT" \
  --receipt "$RECEIPT" \
  --per-stratum 8

"$PYTHON" -P -s -B -m sai.data.prerequisite_development_sample validate \
  --receipt "$RECEIPT"

test -f "$OUTPUT" && test ! -L "$OUTPUT"
test -f "$RECEIPT" && test ! -L "$RECEIPT"
