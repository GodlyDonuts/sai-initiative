#!/bin/bash
#SBATCH --job-name=sai-stack-edu-content-verify
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${ALIGNMENT_RECEIPT:?ALIGNMENT_RECEIPT is required}"
: "${CONTENT_BUNDLE:?CONTENT_BUNDLE is required}"
: "${CONTENT_INDEX:?CONTENT_INDEX is required}"
: "${RECEIPT_OUTPUT:?RECEIPT_OUTPUT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in "$SAI_ROOT" "$PYTHON" "$ALIGNMENT_RECEIPT" "$CONTENT_BUNDLE" "$CONTENT_INDEX" "$RECEIPT_OUTPUT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
for path in "$ALIGNMENT_RECEIPT" "$CONTENT_BUNDLE" "$CONTENT_INDEX"; do
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
  case "$(stat -c %a "$path")" in 400|440|444) ;; *) echo "input must be sealed: $path" >&2; exit 2 ;; esac
done
test ! -e "$RECEIPT_OUTPUT" && test ! -L "$RECEIPT_OUTPUT"
test -d "$(dirname "$RECEIPT_OUTPUT")" && test ! -L "$(dirname "$RECEIPT_OUTPUT")"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -P -s -B -m sai.data.stack_edu_content verify \
  --alignment-receipt "$ALIGNMENT_RECEIPT" \
  --bundle "$CONTENT_BUNDLE" \
  --index "$CONTENT_INDEX" \
  --receipt-output "$RECEIPT_OUTPUT"
chmod 0444 "$RECEIPT_OUTPUT"
"$PYTHON" -P -s -B -m sai.data.stack_edu_content validate \
  --receipt "$RECEIPT_OUTPUT"
test -f "$RECEIPT_OUTPUT" && test ! -L "$RECEIPT_OUTPUT" && test "$(stat -c %h "$RECEIPT_OUTPUT")" -eq 1
test "$(stat -c %a "$RECEIPT_OUTPUT")" = 444
