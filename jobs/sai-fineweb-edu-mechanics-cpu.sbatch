#!/bin/bash
#SBATCH --job-name=sai-fineweb-cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/sai_fineweb_%j.out
#SBATCH --error=logs/sai_fineweb_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${MANIFEST:?MANIFEST is required}"
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${RECEIPT:?RECEIPT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$MANIFEST"
test ! -e "$SOURCE_ROOT"
test ! -e "$OUTPUT"
test ! -e "$RECEIPT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" -P -s -B -m sai.data.fineweb_edu \
  --manifest "$MANIFEST" \
  --source-root "$SOURCE_ROOT" \
  --output "$OUTPUT" \
  --receipt "$RECEIPT" \
  --acquire
