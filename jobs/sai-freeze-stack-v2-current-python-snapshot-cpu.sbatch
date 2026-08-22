#!/bin/bash
#SBATCH --job-name=sai-stack-v2-current-snapshot
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
: "${DATASET_CARD:?DATASET_CARD is required}"
: "${ACCESS_EVIDENCE:?ACCESS_EVIDENCE is required}"
: "${SOURCES:?SOURCES is required}"
: "${RECEIPT_OUTPUT:?RECEIPT_OUTPUT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in "$SAI_ROOT" "$PYTHON" "$DATASET_CARD" "$ACCESS_EVIDENCE" "$RECEIPT_OUTPUT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
for path in "$DATASET_CARD" "$ACCESS_EVIDENCE"; do
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
done
test ! -e "$RECEIPT_OUTPUT" && test ! -L "$RECEIPT_OUTPUT"
test -d "$(dirname "$RECEIPT_OUTPUT")" && test ! -L "$(dirname "$RECEIPT_OUTPUT")"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"

IFS=: read -r -a source_specs <<< "$SOURCES"
test "${#source_specs[@]}" -ge 1
arguments=()
for spec in "${source_specs[@]}"; do
  source_file="${spec%%=*}"
  source_path="${spec#*=}"
  test "$source_file" != "$spec"
  case "$source_path" in /*) ;; *) echo "source path must be absolute: $source_path" >&2; exit 2 ;; esac
  test -f "$source_path" && test ! -L "$source_path" && test "$(stat -c %h "$source_path")" -eq 1
  arguments+=(--source "$source_file=$source_path")
done

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -P -s -B -m sai.data.stack_v2_alignment freeze-snapshot \
  --dataset-card "$DATASET_CARD" \
  --access-evidence "$ACCESS_EVIDENCE" \
  "${arguments[@]}" \
  --receipt-output "$RECEIPT_OUTPUT"
chmod 0444 "$RECEIPT_OUTPUT"
"$PYTHON" -P -s -B -m sai.data.stack_v2_alignment validate-snapshot \
  --receipt "$RECEIPT_OUTPUT"
test -f "$RECEIPT_OUTPUT" && test ! -L "$RECEIPT_OUTPUT" && test "$(stat -c %h "$RECEIPT_OUTPUT")" -eq 1
test "$(stat -c %a "$RECEIPT_OUTPUT")" = 444
