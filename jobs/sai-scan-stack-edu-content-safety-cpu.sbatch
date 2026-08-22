#!/bin/bash
#SBATCH --job-name=sai-stack-edu-safety-scan
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
: "${CONTENT_RECEIPT:?CONTENT_RECEIPT is required}"
: "${FINDINGS_OUTPUT:?FINDINGS_OUTPUT is required}"
: "${RECEIPT_OUTPUT:?RECEIPT_OUTPUT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in "$SAI_ROOT" "$PYTHON" "$CONTENT_RECEIPT" "$FINDINGS_OUTPUT" "$RECEIPT_OUTPUT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$CONTENT_RECEIPT" && test ! -L "$CONTENT_RECEIPT" && test "$(stat -c %h "$CONTENT_RECEIPT")" -eq 1
case "$(stat -c %a "$CONTENT_RECEIPT")" in 400|440|444) ;; *) echo "content receipt must be sealed" >&2; exit 2 ;; esac
test ! -e "$FINDINGS_OUTPUT" && test ! -L "$FINDINGS_OUTPUT"
test ! -e "$RECEIPT_OUTPUT" && test ! -L "$RECEIPT_OUTPUT"
test "$(dirname "$FINDINGS_OUTPUT")" = "$(dirname "$RECEIPT_OUTPUT")"
test -d "$(dirname "$FINDINGS_OUTPUT")" && test ! -L "$(dirname "$FINDINGS_OUTPUT")"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -P -s -B -m sai.data.stack_edu_safety scan \
  --content-receipt "$CONTENT_RECEIPT" \
  --findings-output "$FINDINGS_OUTPUT" \
  --receipt-output "$RECEIPT_OUTPUT"
"$PYTHON" -P -s -B -m sai.data.stack_edu_safety validate \
  --receipt "$RECEIPT_OUTPUT"
chmod 0444 "$FINDINGS_OUTPUT" "$RECEIPT_OUTPUT"
for path in "$FINDINGS_OUTPUT" "$RECEIPT_OUTPUT"; do
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
  test "$(stat -c %a "$path")" = 444
done
