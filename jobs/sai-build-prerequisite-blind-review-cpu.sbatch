#!/bin/bash
#SBATCH --job-name=sai-prereq-blind-review
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${POPULATION_RECEIPT:?POPULATION_RECEIPT is required}"
: "${REVIEW_OUTPUT:?REVIEW_OUTPUT is required}"
: "${KEY_OUTPUT:?KEY_OUTPUT is required}"
: "${RECEIPT_OUTPUT:?RECEIPT_OUTPUT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in \
  "$SAI_ROOT" "$PYTHON" "$POPULATION_RECEIPT" "$REVIEW_OUTPUT" \
  "$KEY_OUTPUT" "$RECEIPT_OUTPUT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$POPULATION_RECEIPT" && test ! -L "$POPULATION_RECEIPT"
test ! -e "$REVIEW_OUTPUT" && test ! -L "$REVIEW_OUTPUT"
test ! -e "$KEY_OUTPUT" && test ! -L "$KEY_OUTPUT"
test ! -e "$RECEIPT_OUTPUT" && test ! -L "$RECEIPT_OUTPUT"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON" -P -s -B -m sai.data.prerequisite_review_packet build \
  --population-receipt "$POPULATION_RECEIPT" \
  --review-output "$REVIEW_OUTPUT" \
  --key-output "$KEY_OUTPUT" \
  --receipt-output "$RECEIPT_OUTPUT"

"$PYTHON" -P -s -B -m sai.data.prerequisite_review_packet validate \
  --population-receipt "$POPULATION_RECEIPT" \
  --review-output "$REVIEW_OUTPUT" \
  --key-output "$KEY_OUTPUT" \
  --receipt-output "$RECEIPT_OUTPUT"

for path in "$REVIEW_OUTPUT" "$KEY_OUTPUT" "$RECEIPT_OUTPUT"; do
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
done
