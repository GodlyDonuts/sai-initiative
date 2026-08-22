#!/usr/bin/env bash
#SBATCH --job-name=sai-q08-workspace
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc27,evc28,evc29,evc30,evc31,evc36,evc43,evc50
#SBATCH --no-requeue

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FAST_KERNELS_ROOT:?FAST_KERNELS_ROOT is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MECHANICS_RECEIPT:?MECHANICS_RECEIPT is required}"
: "${MECHANICS_RECEIPT_SHA256:?MECHANICS_RECEIPT_SHA256 is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"
: "${TRAIN_IDENTITY:?TRAIN_IDENTITY is required}"
: "${STATE_MODE:?STATE_MODE is required}"
: "${TRAINING_SEQUENCES:?TRAINING_SEQUENCES is required}"
: "${CODE_SHA256:?CODE_SHA256 is required}"
: "${ENVIRONMENT_SHA256:?ENVIRONMENT_SHA256 is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${OUTPUT:?OUTPUT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -d "$FAST_KERNELS_ROOT"
test -d "$MODEL_ROOT"
test -d "$TRAIN_STREAM"
test -f "$MECHANICS_RECEIPT"
test "$(sha256sum "$MECHANICS_RECEIPT" | cut -d ' ' -f 1)" = "$MECHANICS_RECEIPT_SHA256"
test -d "$(dirname "$CHECKPOINT")"
test -d "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT"
case "$STATE_MODE" in recurrent|reset_average) ;; *) exit 2 ;; esac
case "$TRAINING_SEQUENCES" in 256|61035) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src:$FAST_KERNELS_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export CUBLAS_WORKSPACE_CONFIG=:4096:8

nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader

resume=()
if [[ "${RESUME:-0}" = 1 ]]; then
  resume=(--resume)
elif [[ "${RESUME:-0}" != 0 ]]; then
  exit 2
fi

exec "$PYTHON" -P -s -B -m sai.training.hf_workspace_screen \
  --model-root "$MODEL_ROOT" \
  --mechanics-receipt "$MECHANICS_RECEIPT" \
  --mechanics-receipt-sha256 "$MECHANICS_RECEIPT_SHA256" \
  --train-stream "$TRAIN_STREAM" \
  --train-identity "$TRAIN_IDENTITY" \
  --state-mode "$STATE_MODE" \
  --training-sequences "$TRAINING_SEQUENCES" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-100}" \
  --code-sha256 "$CODE_SHA256" \
  --environment-sha256 "$ENVIRONMENT_SHA256" \
  --checkpoint "$CHECKPOINT" \
  "${resume[@]}" \
  --output "$OUTPUT"
