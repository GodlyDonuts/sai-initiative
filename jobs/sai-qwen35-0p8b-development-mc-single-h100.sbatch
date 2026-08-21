#!/usr/bin/env bash
#SBATCH --job-name=sai-qwen08-devmc
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
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
: "${BENCHMARK:?BENCHMARK is required}"
: "${BENCHMARK_SOURCE:?BENCHMARK_SOURCE is required}"
: "${DISJOINT_RECEIPT:?DISJOINT_RECEIPT is required}"
: "${TRAINING_SOURCE_SHA256:?TRAINING_SOURCE_SHA256 is required}"
: "${EXPECTED_ROWS:?EXPECTED_ROWS is required}"
: "${EXPECTED_IDENTITY_ORDER_SHA256:?EXPECTED_IDENTITY_ORDER_SHA256 is required}"
: "${OUTPUT:?OUTPUT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -d "$FAST_KERNELS_ROOT"
test -d "$MODEL_ROOT"
test -f "$MECHANICS_RECEIPT"
test -f "$BENCHMARK_SOURCE"
test -f "$DISJOINT_RECEIPT"
test ! -e "$OUTPUT"

export PYTHONPATH="$SAI_ROOT/src:$FAST_KERNELS_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
exec "$PYTHON" -P -s -B -m sai.evaluation.hf_parent_mc \
  --model-root "$MODEL_ROOT" \
  --mechanics-receipt "$MECHANICS_RECEIPT" \
  --mechanics-receipt-sha256 "$MECHANICS_RECEIPT_SHA256" \
  --benchmark "$BENCHMARK" \
  --benchmark-source "$BENCHMARK_SOURCE" \
  --disjoint-receipt "$DISJOINT_RECEIPT" \
  --training-source-sha256 "$TRAINING_SOURCE_SHA256" \
  --expected-rows "$EXPECTED_ROWS" \
  --expected-identity-order-sha256 "$EXPECTED_IDENTITY_ORDER_SHA256" \
  --output "$OUTPUT"
