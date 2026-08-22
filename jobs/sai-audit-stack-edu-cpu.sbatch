#!/bin/bash
#SBATCH --job-name=sai-stack-edu-audit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${SOURCE:?SOURCE is required}"
: "${SOURCE_FILE:?SOURCE_FILE is required}"
: "${EXPECTED_BYTES:?EXPECTED_BYTES is required}"
: "${EXPECTED_SHA256:?EXPECTED_SHA256 is required}"
: "${SAMPLE_OUTPUT:?SAMPLE_OUTPUT is required}"
: "${RECEIPT_OUTPUT:?RECEIPT_OUTPUT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in "$SAI_ROOT" "$PYTHON" "$SOURCE" "$SAMPLE_OUTPUT" "$RECEIPT_OUTPUT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$SOURCE" && test ! -L "$SOURCE"
test ! -e "$SAMPLE_OUTPUT" && test ! -L "$SAMPLE_OUTPUT"
test ! -e "$RECEIPT_OUTPUT" && test ! -L "$RECEIPT_OUTPUT"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$PYTHON" -P -s -B -m sai.data.stack_edu_audit audit \
  --source "$SOURCE" \
  --source-file "$SOURCE_FILE" \
  --expected-bytes "$EXPECTED_BYTES" \
  --expected-sha256 "$EXPECTED_SHA256" \
  --sample-output "$SAMPLE_OUTPUT" \
  --receipt-output "$RECEIPT_OUTPUT"

"$PYTHON" -P -s -B -m sai.data.stack_edu_audit validate \
  --receipt "$RECEIPT_OUTPUT"

for path in "$SAMPLE_OUTPUT" "$RECEIPT_OUTPUT"; do
  test -f "$path" && test ! -L "$path" && test "$(stat -c %h "$path")" -eq 1
done
