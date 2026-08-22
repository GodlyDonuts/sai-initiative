#!/bin/bash
#SBATCH --job-name=sai-numeric-tokenizer
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/sai_numeric_tokenizer_%j.out
#SBATCH --error=logs/sai_numeric_tokenizer_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${CORPUS:?CORPUS is required}"
: "${CORPUS_SHA256:?CORPUS_SHA256 is required}"
: "${PROTECTED_SUITE:?PROTECTED_SUITE is required}"
: "${PROTECTED_SUITE_SHA256:?PROTECTED_SUITE_SHA256 is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$CORPUS"
test ! -L "$CORPUS"
test "$(stat -c %h "$CORPUS")" = "1"
test "$(sha256sum "$CORPUS" | cut -d' ' -f1)" = "$CORPUS_SHA256"
test -f "$PROTECTED_SUITE"
test ! -L "$PROTECTED_SUITE"
test "$(stat -c %h "$PROTECTED_SUITE")" = "1"
test "$(sha256sum "$PROTECTED_SUITE" | cut -d' ' -f1)" = "$PROTECTED_SUITE_SHA256"
test ! -e "$OUTPUT_ROOT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false

exec "$PYTHON" -P -s -B -m sai.tokenizer.numeric_pretokenization \
  --corpus "$CORPUS" \
  --protected-suite "$PROTECTED_SUITE" \
  --output-root "$OUTPUT_ROOT" \
  --vocab-size 48000
