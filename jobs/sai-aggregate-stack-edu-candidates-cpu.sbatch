#!/bin/bash
#SBATCH --job-name=sai-stack-edu-candidate-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${METADATA_AGGREGATE:?METADATA_AGGREGATE is required}"
: "${SHARD_RECEIPTS:?SHARD_RECEIPTS is required}"
: "${CANDIDATES_OUTPUT:?CANDIDATES_OUTPUT is required}"
: "${RECEIPT_OUTPUT:?RECEIPT_OUTPUT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in "$SAI_ROOT" "$PYTHON" "$METADATA_AGGREGATE" "$CANDIDATES_OUTPUT" "$RECEIPT_OUTPUT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$METADATA_AGGREGATE" && test ! -L "$METADATA_AGGREGATE" && test "$(stat -c %h "$METADATA_AGGREGATE")" -eq 1
test ! -e "$CANDIDATES_OUTPUT" && test ! -L "$CANDIDATES_OUTPUT"
test ! -e "$RECEIPT_OUTPUT" && test ! -L "$RECEIPT_OUTPUT"
test "$(dirname "$CANDIDATES_OUTPUT")" = "$(dirname "$RECEIPT_OUTPUT")"
test -d "$(dirname "$CANDIDATES_OUTPUT")" && test ! -L "$(dirname "$CANDIDATES_OUTPUT")"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"

IFS=: read -r -a receipt_paths <<< "$SHARD_RECEIPTS"
test "${#receipt_paths[@]}" -ge 2
arguments=()
for path in "${receipt_paths[@]}"; do
  case "$path" in /*) ;; *) echo "receipt path must be absolute: $path" >&2; exit 2 ;; esac
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
  arguments+=(--shard-receipt "$path")
done

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -P -s -B -m sai.data.stack_edu_candidates aggregate \
  --metadata-aggregate "$METADATA_AGGREGATE" \
  "${arguments[@]}" \
  --candidates-output "$CANDIDATES_OUTPUT" \
  --receipt-output "$RECEIPT_OUTPUT"
chmod 0444 "$CANDIDATES_OUTPUT" "$RECEIPT_OUTPUT"
"$PYTHON" -P -s -B -m sai.data.stack_edu_candidates validate-aggregate \
  --receipt "$RECEIPT_OUTPUT"
for path in "$CANDIDATES_OUTPUT" "$RECEIPT_OUTPUT"; do
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
  test "$(stat -c %a "$path")" = 444
done
