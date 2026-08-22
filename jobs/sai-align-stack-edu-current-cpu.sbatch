#!/bin/bash
#SBATCH --job-name=sai-stack-edu-current-align
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${CANDIDATE_RECEIPT:?CANDIDATE_RECEIPT is required}"
: "${SNAPSHOT_RECEIPT:?SNAPSHOT_RECEIPT is required}"
: "${ALIGNED_OUTPUT:?ALIGNED_OUTPUT is required}"
: "${RECEIPT_OUTPUT:?RECEIPT_OUTPUT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in "$SAI_ROOT" "$PYTHON" "$CANDIDATE_RECEIPT" "$SNAPSHOT_RECEIPT" "$ALIGNED_OUTPUT" "$RECEIPT_OUTPUT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
for path in "$CANDIDATE_RECEIPT" "$SNAPSHOT_RECEIPT"; do
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
done
test ! -e "$ALIGNED_OUTPUT" && test ! -L "$ALIGNED_OUTPUT"
test ! -e "$RECEIPT_OUTPUT" && test ! -L "$RECEIPT_OUTPUT"
test "$(dirname "$ALIGNED_OUTPUT")" = "$(dirname "$RECEIPT_OUTPUT")"
test -d "$(dirname "$ALIGNED_OUTPUT")" && test ! -L "$(dirname "$ALIGNED_OUTPUT")"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -P -s -B -m sai.data.stack_v2_alignment align \
  --candidate-receipt "$CANDIDATE_RECEIPT" \
  --snapshot-receipt "$SNAPSHOT_RECEIPT" \
  --aligned-output "$ALIGNED_OUTPUT" \
  --receipt-output "$RECEIPT_OUTPUT"
chmod 0444 "$ALIGNED_OUTPUT" "$RECEIPT_OUTPUT"
"$PYTHON" -P -s -B -m sai.data.stack_v2_alignment validate-alignment \
  --receipt "$RECEIPT_OUTPUT"
for path in "$ALIGNED_OUTPUT" "$RECEIPT_OUTPUT"; do
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
  test "$(stat -c %a "$path")" = 444
done
