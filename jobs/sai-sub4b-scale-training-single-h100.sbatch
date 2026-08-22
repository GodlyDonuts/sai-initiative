#!/bin/bash
#SBATCH --job-name=sai-sub4b-scale
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=24:00:00
#SBATCH --no-requeue
#SBATCH --exclude=evc26,evc27,evc28,evc29,evc30,evc31,evc35,evc36,evc38,evc43,evc50
#SBATCH --output=logs/sai_sub4b_scale_%j.out
#SBATCH --error=logs/sai_sub4b_scale_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${SCALE:?SCALE is required}"
: "${FAMILY:?FAMILY is required}"
: "${ADMISSION_RECEIPT:?ADMISSION_RECEIPT is required}"
: "${GEOMETRY:?GEOMETRY is required}"
: "${GEOMETRY_SHA256:?GEOMETRY_SHA256 is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"
: "${TRAIN_IDENTITY:?TRAIN_IDENTITY is required}"
: "${DEVELOPMENT_STREAM:?DEVELOPMENT_STREAM is required}"
: "${DEVELOPMENT_IDENTITY:?DEVELOPMENT_IDENTITY is required}"
: "${DEVELOPMENT_SEQUENCES:?DEVELOPMENT_SEQUENCES is required}"
: "${DEVELOPMENT_BATCH_SIZE:?DEVELOPMENT_BATCH_SIZE is required}"
: "${OPTIMIZER_STEPS:?OPTIMIZER_STEPS is required}"
: "${TRAINING_SEQUENCES:?TRAINING_SEQUENCES is required}"
: "${SEQUENCES_PER_UPDATE:?SEQUENCES_PER_UPDATE is required}"
: "${BATCH_SIZE:?BATCH_SIZE is required}"
: "${CHECKPOINT_INTERVAL:?CHECKPOINT_INTERVAL is required}"
: "${LEARNING_RATE:?LEARNING_RATE is required}"
: "${WARMUP_STEPS:?WARMUP_STEPS is required}"
: "${MINIMUM_LEARNING_RATE_RATIO:?MINIMUM_LEARNING_RATE_RATIO is required}"
: "${WEIGHT_DECAY:?WEIGHT_DECAY is required}"
: "${GRADIENT_CLIP_NORM:?GRADIENT_CLIP_NORM is required}"
: "${SEED:?SEED is required}"
: "${KERNEL_PARITY_RECEIPT:?KERNEL_PARITY_RECEIPT is required}"
: "${KERNEL_PARITY_SHA256:?KERNEL_PARITY_SHA256 is required}"
: "${FULL_MODEL_PARITY_RECEIPT:?FULL_MODEL_PARITY_RECEIPT is required}"
: "${FULL_MODEL_PARITY_SHA256:?FULL_MODEL_PARITY_SHA256 is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${OUTPUT:?OUTPUT is required}"

case "$SCALE" in
  300m|1b) ;;
  *) exit 64 ;;
esac
case "$FAMILY" in
  gated_gqa|gdn_hybrid|kda_mla_hybrid) ;;
  *) exit 64 ;;
esac

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -f "$ADMISSION_RECEIPT"
test ! -L "$ADMISSION_RECEIPT"
test -f "$GEOMETRY"
test ! -L "$GEOMETRY"
test "$(sha256sum "$GEOMETRY" | cut -d ' ' -f 1)" = "$GEOMETRY_SHA256"
test -d "$TRAIN_STREAM"
test -d "$DEVELOPMENT_STREAM"
test -f "$KERNEL_PARITY_RECEIPT"
test ! -L "$KERNEL_PARITY_RECEIPT"
test "$(sha256sum "$KERNEL_PARITY_RECEIPT" | cut -d ' ' -f 1)" = "$KERNEL_PARITY_SHA256"
test -f "$FULL_MODEL_PARITY_RECEIPT"
test ! -L "$FULL_MODEL_PARITY_RECEIPT"
test "$(sha256sum "$FULL_MODEL_PARITY_RECEIPT" | cut -d ' ' -f 1)" = "$FULL_MODEL_PARITY_SHA256"
test -d "$(dirname "$CHECKPOINT")"
test -d "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT"

"$PYTHON" -P -s -B - "$KERNEL_PARITY_RECEIPT" "$FULL_MODEL_PARITY_RECEIPT" <<'PY'
import json
import pathlib
import sys

kernel = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert kernel["schema"] == "sai-fla-parity-receipt-v1"
assert kernel["status"] == "cuda_fla_parity_qualified"
assert kernel["production_cuda_qualified"] is True
assert kernel["operators"]["version"] == "0.4.2"
assert kernel["operators"]["mock"] is False

full = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert full["schema"] == "sai-full-delta-mixer-fla-parity-v1"
assert full["status"] == "production_cuda_qualified"
assert full["production_cuda_qualified"] is True
assert full["fla_version"] == "0.4.2"
assert len(full["cases"]) == 8
assert all(case["passed"] for case in full["cases"])
PY

gpu_count="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')"
test "$gpu_count" = 1

code_sha256="$(git -C "$SAI_ROOT" archive --format=tar "$EXPECTED_COMMIT" | sha256sum | cut -d ' ' -f 1)"
environment_sha256="$(printf '%s\n%s\n' "$KERNEL_PARITY_SHA256" "$FULL_MODEL_PARITY_SHA256" | sha256sum | cut -d ' ' -f 1)"
test "${#code_sha256}" = 64
test "${#environment_sha256}" = 64

export PYTHONPATH="$FLA_ROOT:$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false

arguments=(
  --scale "$SCALE"
  --family "$FAMILY"
  --admission-receipt "$ADMISSION_RECEIPT"
  --geometry "$GEOMETRY"
  --train-stream "$TRAIN_STREAM"
  --train-identity "$TRAIN_IDENTITY"
  --development-stream "$DEVELOPMENT_STREAM"
  --development-identity "$DEVELOPMENT_IDENTITY"
  --development-sequences "$DEVELOPMENT_SEQUENCES"
  --development-batch-size "$DEVELOPMENT_BATCH_SIZE"
  --optimizer-steps "$OPTIMIZER_STEPS"
  --training-sequences "$TRAINING_SEQUENCES"
  --sequences-per-update "$SEQUENCES_PER_UPDATE"
  --batch-size "$BATCH_SIZE"
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --learning-rate "$LEARNING_RATE"
  --warmup-steps "$WARMUP_STEPS"
  --minimum-learning-rate-ratio "$MINIMUM_LEARNING_RATE_RATIO"
  --weight-decay "$WEIGHT_DECAY"
  --gradient-clip-norm "$GRADIENT_CLIP_NORM"
  --seed "$SEED"
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

exec "$PYTHON" -P -s -B -m sai.training.scale_training "${arguments[@]}"
