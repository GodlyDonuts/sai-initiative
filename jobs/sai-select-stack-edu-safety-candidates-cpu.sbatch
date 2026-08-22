#!/bin/bash
#SBATCH --job-name=sai-stack-edu-safety-select
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${SAFETY_RECEIPT:?SAFETY_RECEIPT is required}"
: "${REVIEWS:?REVIEWS is required}"
: "${SELECTED_OUTPUT:?SELECTED_OUTPUT is required}"
: "${RECEIPT_OUTPUT:?RECEIPT_OUTPUT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in "$SAI_ROOT" "$PYTHON" "$SAFETY_RECEIPT" "$REVIEWS" "$SELECTED_OUTPUT" "$RECEIPT_OUTPUT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
for path in "$SAFETY_RECEIPT" "$REVIEWS"; do
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
  case "$(stat -c %a "$path")" in 400|440|444) ;; *) echo "input must be sealed: $path" >&2; exit 2 ;; esac
done
test ! -e "$SELECTED_OUTPUT" && test ! -L "$SELECTED_OUTPUT"
test ! -e "$RECEIPT_OUTPUT" && test ! -L "$RECEIPT_OUTPUT"
test "$(dirname "$SELECTED_OUTPUT")" = "$(dirname "$RECEIPT_OUTPUT")"
test -d "$(dirname "$SELECTED_OUTPUT")" && test ! -L "$(dirname "$SELECTED_OUTPUT")"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -P -s -B -m sai.data.stack_edu_safety_select select \
  --safety-receipt "$SAFETY_RECEIPT" \
  --reviews "$REVIEWS" \
  --selected-output "$SELECTED_OUTPUT" \
  --receipt-output "$RECEIPT_OUTPUT"
"$PYTHON" -P -s -B -m sai.data.stack_edu_safety_select validate \
  --receipt "$RECEIPT_OUTPUT"
chmod 0444 "$SELECTED_OUTPUT" "$RECEIPT_OUTPUT"
for path in "$SELECTED_OUTPUT" "$RECEIPT_OUTPUT"; do
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
  test "$(stat -c %a "$path")" = 444
done
