#!/bin/bash
#SBATCH --job-name=sai-streams-cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/sai_streams_%j.out
#SBATCH --error=logs/sai_streams_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${TOKENIZER_ROOT:?TOKENIZER_ROOT is required}"
: "${SELECTED_RECEIPT:?SELECTED_RECEIPT is required}"
: "${TRAIN_CORPUS:?TRAIN_CORPUS is required}"
: "${DEVELOPMENT_CORPUS:?DEVELOPMENT_CORPUS is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"
: "${DEVELOPMENT_STREAM:?DEVELOPMENT_STREAM is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$TOKENIZER_ROOT"
test -f "$SELECTED_RECEIPT"
test -f "$TRAIN_CORPUS"
test -f "$DEVELOPMENT_CORPUS"
test ! -e "$TRAIN_STREAM"
test ! -e "$DEVELOPMENT_STREAM"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false

tokenizer_sha256="$($PYTHON -P -s -B - "$SELECTED_RECEIPT" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert payload["schema"] == "sai-selected-tokenizer-receipt-v1"
assert payload["status"] == "qualified"
assert payload["vocab_size"] == 48_000
print(payload["tokenizer_identity_sha256"])
PY
)"
test "${#tokenizer_sha256}" = 64

"$PYTHON" -P -s -B -m sai.data.token_stream \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --expected-tokenizer-sha256 "$tokenizer_sha256" \
  --source "$TRAIN_CORPUS" \
  --output "$TRAIN_STREAM" \
  --sequence-length 2048 \
  --prefix-sequences 256 \
  --prefix-sequences 12207 \
  --prefix-sequences 24414 \
  --prefix-sequences 48828

exec "$PYTHON" -P -s -B -m sai.data.token_stream \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --expected-tokenizer-sha256 "$tokenizer_sha256" \
  --source "$DEVELOPMENT_CORPUS" \
  --output "$DEVELOPMENT_STREAM" \
  --sequence-length 2048 \
  --prefix-sequences 4096
