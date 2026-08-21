#!/bin/bash
#SBATCH --job-name=sai-tokenizers-cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/sai_tokenizers_%j.out
#SBATCH --error=logs/sai_tokenizers_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${SOURCE_CORPUS:?SOURCE_CORPUS is required}"
: "${DECONTAMINATION_RECEIPT:?DECONTAMINATION_RECEIPT is required}"
: "${TRAIN_CORPUS:?TRAIN_CORPUS is required}"
: "${DEVELOPMENT_CORPUS:?DEVELOPMENT_CORPUS is required}"
: "${SPLIT_RECEIPT:?SPLIT_RECEIPT is required}"
: "${CANDIDATE_ROOT:?CANDIDATE_ROOT is required}"
: "${REPORT:?REPORT is required}"
: "${SELECTED_RECEIPT:?SELECTED_RECEIPT is required}"

test -d "$SAI_ROOT/.git"
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$SOURCE_CORPUS"
test -f "$DECONTAMINATION_RECEIPT"
test ! -e "$TRAIN_CORPUS"
test ! -e "$DEVELOPMENT_CORPUS"
test ! -e "$SPLIT_RECEIPT"
test ! -e "$CANDIDATE_ROOT"
test ! -e "$REPORT"
test ! -e "$SELECTED_RECEIPT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -P -s -B -m sai.data.decontamination validate \
  --receipt "$DECONTAMINATION_RECEIPT"

"$PYTHON" -P -s -B -m sai.data.split \
  --source "$SOURCE_CORPUS" \
  --train "$TRAIN_CORPUS" \
  --development "$DEVELOPMENT_CORPUS" \
  --receipt "$SPLIT_RECEIPT"

"$PYTHON" -P -s -B -m sai.tokenizer.build \
  --corpus "$TRAIN_CORPUS" \
  --output-root "$CANDIDATE_ROOT"

exec "$PYTHON" -P -s -B -m sai.tokenizer.qualification \
  --candidate "32k=$CANDIDATE_ROOT/32k" \
  --candidate "48k=$CANDIDATE_ROOT/48k" \
  --candidate "64k=$CANDIDATE_ROOT/64k" \
  --corpus "$TRAIN_CORPUS" \
  --protected-suite "$SAI_ROOT/docs/SAI_TOKENIZER_PROTECTED_SUITE.jsonl" \
  --output "$REPORT" \
  --selected-48k-output "$SELECTED_RECEIPT"
