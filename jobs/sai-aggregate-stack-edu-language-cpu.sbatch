#!/bin/bash
#SBATCH --job-name=sai-stack-edu-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${RECEIPTS:?RECEIPTS is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in "$SAI_ROOT" "$PYTHON" "$OUTPUT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test ! -e "$OUTPUT" && test ! -L "$OUTPUT"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"

IFS=: read -r -a receipt_paths <<< "$RECEIPTS"
test "${#receipt_paths[@]}" -ge 2
arguments=()
for path in "${receipt_paths[@]}"; do
  case "$path" in /*) ;; *) echo "receipt path must be absolute: $path" >&2; exit 2 ;; esac
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
  arguments+=(--receipt "$path")
done

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -P -s -B -m sai.data.stack_edu_aggregate aggregate \
  "${arguments[@]}" \
  --output "$OUTPUT"
"$PYTHON" -P -s -B -m sai.data.stack_edu_aggregate validate \
  --receipt "$OUTPUT"
test -f "$OUTPUT" && test ! -L "$OUTPUT" && test "$(stat -c %h "$OUTPUT")" -eq 1
