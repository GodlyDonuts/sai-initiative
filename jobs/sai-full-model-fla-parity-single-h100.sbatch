#!/bin/bash
#SBATCH --job-name=sai-full-fla-parity
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc27,evc28,evc29,evc30,evc31,evc36,evc38,evc43,evc50
#SBATCH --output=logs/sai_full_fla_parity_%j.out
#SBATCH --error=logs/sai_full_fla_parity_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"

test -d "$SAI_ROOT/.git"
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -d "$(dirname "$OUTPUT")"
test ! -L "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT"
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')" = 1

export PYTHONPATH="$FLA_ROOT:$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

exec "$PYTHON" -P -s -B -m sai.training.full_model_fla_parity \
  --output "$OUTPUT" \
  --seed 20260821
