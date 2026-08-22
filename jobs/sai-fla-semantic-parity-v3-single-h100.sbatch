#!/bin/bash
#SBATCH --job-name=sai-fla-semantic-v3
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc27,evc28,evc29,evc30,evc31,evc35,evc36,evc38,evc43,evc50
#SBATCH --output=logs/sai_fla_semantic_v3_%j.out
#SBATCH --error=logs/sai_fla_semantic_v3_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${SEED:?SEED is required}"
: "${OUTPUT:?OUTPUT is required}"

case "$SEED" in
  20260827|20260828|20260829) ;;
  *) exit 64 ;;
esac

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test "$(git -C "$FLA_ROOT" rev-parse --short=7 HEAD)" = "ca910f8"
test -z "$(git -C "$FLA_ROOT" status --short)"
test -d "$(dirname "$OUTPUT")"
test ! -L "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT"
test "$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')" = 1

export PYTHONPATH="$FLA_ROOT:$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

exec "$PYTHON" -P -s -B -m sai.training.fla_semantic_parity_v3 \
  --seed "$SEED" \
  --fla-root "$FLA_ROOT" \
  --output "$OUTPUT"
