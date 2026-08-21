#!/bin/bash
#SBATCH --job-name=sai-dev-stream-cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/sai_dev_stream_%j.out
#SBATCH --error=logs/sai_dev_stream_%j.err

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${TOKENIZER_ROOT:?TOKENIZER_ROOT is required}"
: "${SELECTED_RECEIPT:?SELECTED_RECEIPT is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"
: "${DEVELOPMENT_CORPUS:?DEVELOPMENT_CORPUS is required}"
: "${DEVELOPMENT_STREAM:?DEVELOPMENT_STREAM is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$TOKENIZER_ROOT"
test -f "$SELECTED_RECEIPT"
test -d "$TRAIN_STREAM"
test -f "$DEVELOPMENT_CORPUS"
test ! -e "$DEVELOPMENT_STREAM"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false

tokenizer_sha256="$($PYTHON -P -s -B - \
  "$SELECTED_RECEIPT" "$TRAIN_STREAM" <<'PY'
import json
import pathlib
import sys

from sai.data.token_stream import validate_frozen_stream

selected = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert selected["schema"] == "sai-tokenizer-qualification-receipt-v1"
assert selected["status"] == "qualified"
assert selected["vocab_size"] == 48_000
train = validate_frozen_stream(pathlib.Path(sys.argv[2]), verify_sources=True)
assert train["status"] == "complete"
assert train["sequences"] >= 48_828
assert train["prefix_utf8_bytes"]["256"] > 0
assert train["prefix_utf8_bytes"]["48828"] > 0
assert train["tokenizer_identity_sha256"] == selected["tokenizer_identity_sha256"]
print(selected["tokenizer_identity_sha256"])
PY
)"
test "${#tokenizer_sha256}" = 64

exec "$PYTHON" -P -s -B -m sai.data.token_stream \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --expected-tokenizer-sha256 "$tokenizer_sha256" \
  --source "$DEVELOPMENT_CORPUS" \
  --output "$DEVELOPMENT_STREAM" \
  --sequence-length 2048 \
  --prefix-sequences 1024
