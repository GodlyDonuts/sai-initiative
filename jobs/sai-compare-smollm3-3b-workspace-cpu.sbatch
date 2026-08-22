#!/usr/bin/env bash
#SBATCH --job-name=sai-compare-smol3-workspace
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${PARENT_MMLU:?PARENT_MMLU is required}"
: "${PARENT_MUSR:?PARENT_MUSR is required}"
: "${RECURRENT_MMLU:?RECURRENT_MMLU is required}"
: "${RECURRENT_MUSR:?RECURRENT_MUSR is required}"
: "${RESET_MMLU:?RESET_MMLU is required}"
: "${RESET_MUSR:?RESET_MUSR is required}"
: "${RECURRENT_TRAINING_RESULT:?RECURRENT_TRAINING_RESULT is required}"
: "${RESET_TRAINING_RESULT:?RESET_TRAINING_RESULT is required}"
: "${QWEN_FACTOR_RECEIPT:?QWEN_FACTOR_RECEIPT is required}"
: "${OUTPUT:?OUTPUT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
for input in \
  "$PARENT_MMLU" "$PARENT_MUSR" \
  "$RECURRENT_MMLU" "$RECURRENT_MUSR" \
  "$RESET_MMLU" "$RESET_MUSR" \
  "$RECURRENT_TRAINING_RESULT" "$RESET_TRAINING_RESULT" \
  "$QWEN_FACTOR_RECEIPT"; do
  test -f "$input"
  test ! -L "$input"
done
test -d "$(dirname "$OUTPUT")"
test ! -e "$OUTPUT"
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

exec "$PYTHON" -P -s -B -m sai.evaluation.hf_smol_workspace_compare \
  --parent-mmlu-pro "$PARENT_MMLU" \
  --parent-musr "$PARENT_MUSR" \
  --recurrent-mmlu-pro "$RECURRENT_MMLU" \
  --recurrent-musr "$RECURRENT_MUSR" \
  --reset-mmlu-pro "$RESET_MMLU" \
  --reset-musr "$RESET_MUSR" \
  --recurrent-training-result "$RECURRENT_TRAINING_RESULT" \
  --reset-training-result "$RESET_TRAINING_RESULT" \
  --qwen-factor-receipt "$QWEN_FACTOR_RECEIPT" \
  --output "$OUTPUT"
