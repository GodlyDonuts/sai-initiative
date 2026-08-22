#!/usr/bin/env bash
#SBATCH --job-name=sai-refresh-dev-mc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${DECONTAMINATION_JOB_ID:?DECONTAMINATION_JOB_ID is required}"
: "${DECONTAMINATION_RECEIPT:?DECONTAMINATION_RECEIPT is required}"
: "${MMLU_QUESTIONS:?MMLU_QUESTIONS is required}"
: "${MMLU_ASSESSORS:?MMLU_ASSESSORS is required}"
: "${MMLU_QUESTIONS_SHA256:?MMLU_QUESTIONS_SHA256 is required}"
: "${MMLU_ASSESSORS_SHA256:?MMLU_ASSESSORS_SHA256 is required}"
: "${MMLU_IDENTITY_ORDER_SHA256:?MMLU_IDENTITY_ORDER_SHA256 is required}"
: "${MUSR_QUESTIONS:?MUSR_QUESTIONS is required}"
: "${MUSR_ASSESSORS:?MUSR_ASSESSORS is required}"
: "${MUSR_QUESTIONS_SHA256:?MUSR_QUESTIONS_SHA256 is required}"
: "${MUSR_ASSESSORS_SHA256:?MUSR_ASSESSORS_SHA256 is required}"
: "${MUSR_IDENTITY_ORDER_SHA256:?MUSR_IDENTITY_ORDER_SHA256 is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test ! -e "$OUTPUT_ROOT"
test "$(sacct -j "$DECONTAMINATION_JOB_ID" -X -n -P -o State | sed -n '1p')" = "COMPLETED"

case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU population refresh was exposed to a GPU" >&2; exit 2 ;;
esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

exec "$PYTHON" -P -s -B -m sai.evaluation.population_refresh \
  --source-commit "$EXPECTED_COMMIT" \
  --decontamination-receipt "$DECONTAMINATION_RECEIPT" \
  --mmlu-questions "$MMLU_QUESTIONS" \
  --mmlu-assessors "$MMLU_ASSESSORS" \
  --mmlu-questions-sha256 "$MMLU_QUESTIONS_SHA256" \
  --mmlu-assessors-sha256 "$MMLU_ASSESSORS_SHA256" \
  --mmlu-identity-order-sha256 "$MMLU_IDENTITY_ORDER_SHA256" \
  --musr-questions "$MUSR_QUESTIONS" \
  --musr-assessors "$MUSR_ASSESSORS" \
  --musr-questions-sha256 "$MUSR_QUESTIONS_SHA256" \
  --musr-assessors-sha256 "$MUSR_ASSESSORS_SHA256" \
  --musr-identity-order-sha256 "$MUSR_IDENTITY_ORDER_SHA256" \
  --output-root "$OUTPUT_ROOT"
