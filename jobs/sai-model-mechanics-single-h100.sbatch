#!/bin/bash
#SBATCH --job-name=sai-100m-mechanics
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc27,evc28,evc29,evc30,evc31,evc35,evc36,evc38,evc43,evc50
#SBATCH --output=logs/sai_model_mechanics_%j.out
#SBATCH --error=logs/sai_model_mechanics_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${GEOMETRY:?GEOMETRY is required}"
: "${PARITY_RECEIPT:?PARITY_RECEIPT is required}"
: "${OUTPUT:?OUTPUT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -f "$GEOMETRY"
test -f "$PARITY_RECEIPT"
test ! -e "$OUTPUT"

gpu_count="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')"
test "$gpu_count" = 1

export PYTHONPATH="$FLA_ROOT:$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" -P -s -B -m sai.training.model_mechanics \
  --geometry "$GEOMETRY" \
  --parity-receipt "$PARITY_RECEIPT" \
  --source-commit "$EXPECTED_COMMIT" \
  --output "$OUTPUT"
