#!/bin/bash
#SBATCH --job-name=sai-short-screen
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc27,evc28,evc29,evc30,evc31,evc36,evc38,evc43,evc50
#SBATCH --output=logs/sai_short_screen_%j.out
#SBATCH --error=logs/sai_short_screen_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${GEOMETRY:?GEOMETRY is required}"
: "${FAMILY:?FAMILY is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"
: "${TRAIN_IDENTITY:?TRAIN_IDENTITY is required}"
: "${DEVELOPMENT_STREAM:?DEVELOPMENT_STREAM is required}"
: "${DEVELOPMENT_IDENTITY:?DEVELOPMENT_IDENTITY is required}"
: "${DEVELOPMENT_SEQUENCES:?DEVELOPMENT_SEQUENCES is required}"
: "${OPTIMIZER_STEPS:?OPTIMIZER_STEPS is required}"
: "${CODE_SHA256:?CODE_SHA256 is required}"
: "${ENVIRONMENT_SHA256:?ENVIRONMENT_SHA256 is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${OUTPUT:?OUTPUT is required}"

test -d "$SAI_ROOT/.git"
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -f "$GEOMETRY"
test -d "$TRAIN_STREAM"
test -d "$DEVELOPMENT_STREAM"
test -d "$(dirname "$CHECKPOINT")"
test -d "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT"

gpu_count="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')"
test "$gpu_count" = 1

export PYTHONPATH="$FLA_ROOT:$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

arguments=(
  --geometry "$GEOMETRY"
  --family "$FAMILY"
  --train-stream "$TRAIN_STREAM"
  --train-identity "$TRAIN_IDENTITY"
  --development-stream "$DEVELOPMENT_STREAM"
  --development-identity "$DEVELOPMENT_IDENTITY"
  --development-sequences "$DEVELOPMENT_SEQUENCES"
  --optimizer-steps "$OPTIMIZER_STEPS"
  --code-sha256 "$CODE_SHA256"
  --environment-sha256 "$ENVIRONMENT_SHA256"
  --checkpoint "$CHECKPOINT"
  --output "$OUTPUT"
)

if [[ "${RESUME:-0}" = 1 ]]; then
  arguments+=(--resume)
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  arguments+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${DEVELOPMENT_BATCH_SIZE:-}" ]]; then
  arguments+=(--development-batch-size "$DEVELOPMENT_BATCH_SIZE")
fi
if [[ -n "${CHECKPOINT_INTERVAL:-}" ]]; then
  arguments+=(--checkpoint-interval "$CHECKPOINT_INTERVAL")
fi
if [[ -n "${SEED:-}" ]]; then
  arguments+=(--seed "$SEED")
fi

exec "$PYTHON" -P -s -B -m sai.training.short_screen "${arguments[@]}"
