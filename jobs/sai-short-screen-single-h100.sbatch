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
: "${TRAINING_SEQUENCES:?TRAINING_SEQUENCES is required}"
: "${ENVIRONMENT_RECEIPT:?ENVIRONMENT_RECEIPT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${OUTPUT:?OUTPUT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -f "$GEOMETRY"
test -d "$TRAIN_STREAM"
test -d "$DEVELOPMENT_STREAM"
test -f "$ENVIRONMENT_RECEIPT"
test ! -L "$ENVIRONMENT_RECEIPT"
test -d "$(dirname "$CHECKPOINT")"
test -d "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT"

code_sha256="$(git -C "$SAI_ROOT" archive --format=tar "$EXPECTED_COMMIT" | sha256sum | cut -d ' ' -f 1)"
environment_sha256="$(sha256sum "$ENVIRONMENT_RECEIPT" | cut -d ' ' -f 1)"
test "${#code_sha256}" = 64
test "${#environment_sha256}" = 64

"$PYTHON" -P -s -B - "$ENVIRONMENT_RECEIPT" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert receipt["schema"] == "sai-fla-parity-receipt-v1"
assert receipt["status"] == "cuda_fla_parity_qualified"
assert receipt["parity_qualified"] is True
assert receipt["production_cuda_qualified"] is True
assert receipt["environment"]["device_type"] == "cuda"
assert receipt["environment"]["cuda_bf16_supported"] is True
assert receipt["operators"]["version"] == "0.4.2"
assert receipt["operators"]["mock"] is False
PY

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
  --training-sequences "$TRAINING_SEQUENCES"
  --code-sha256 "$code_sha256"
  --environment-sha256 "$environment_sha256"
  --checkpoint "$CHECKPOINT"
  --output "$OUTPUT"
)

if [[ "${RESUME:-0}" = 1 ]]; then
  arguments+=(--resume)
fi
if [[ "${MECHANICS_ONLY:-0}" = 1 ]]; then
  arguments+=(--mechanics-only)
fi
if [[ -n "${BATCH_SIZE:-}" ]]; then
  arguments+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "${SEQUENCES_PER_UPDATE:-}" ]]; then
  arguments+=(--sequences-per-update "$SEQUENCES_PER_UPDATE")
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
