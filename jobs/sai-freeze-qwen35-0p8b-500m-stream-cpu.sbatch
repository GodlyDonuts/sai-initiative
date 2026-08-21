#!/usr/bin/env bash
#SBATCH --job-name=sai-qwen08-stream
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${TRAIN_CORPUS:?TRAIN_CORPUS is required}"
: "${TRAIN_CORPUS_RECEIPT:?TRAIN_CORPUS_RECEIPT is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$MODEL_ROOT"
test -f "$TRAIN_CORPUS"
test -f "$TRAIN_CORPUS_RECEIPT"
test ! -e "$TRAIN_STREAM"

case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU Qwen stream freezer was exposed to a GPU" >&2; exit 2 ;;
esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

tokenizer_identity="$($PYTHON -P -s -B - <<'PY' "$MODEL_ROOT" "$TRAIN_CORPUS" "$TRAIN_CORPUS_RECEIPT"
import hashlib
import json
import pathlib
import sys

from sai.data.hf_model_snapshot import validate_snapshot
from sai.data.token_stream import sha256_tree

model_root, corpus, receipt_path = map(pathlib.Path, sys.argv[1:])
validate_snapshot(model_root)
receipt = json.loads(receipt_path.read_text())
assert receipt["schema"] == "sai-decontamination-receipt-v1"
assert receipt["status"] == "passed"
assert receipt["output"]["path"] == str(corpus.resolve())
assert receipt["output"]["bytes"] == corpus.stat().st_size
assert receipt["output"]["sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
print(sha256_tree(model_root))
PY
)"
test "${#tokenizer_identity}" = 64

"$PYTHON" -P -s -B -m sai.data.token_stream \
  --tokenizer-root "$MODEL_ROOT" \
  --expected-tokenizer-sha256 "$tokenizer_identity" \
  --source "$TRAIN_CORPUS" \
  --output "$TRAIN_STREAM" \
  --sequence-length 2048 \
  --prefix-sequences 256 \
  --prefix-sequences 61035 \
  --prefix-sequences 122070 \
  --prefix-sequences 183105 \
  --prefix-sequences 244140

"$PYTHON" -P -s -B - <<'PY' "$TRAIN_STREAM" "$tokenizer_identity"
import pathlib
import sys

from sai.data.token_stream import validate_frozen_stream

stream = validate_frozen_stream(pathlib.Path(sys.argv[1]), verify_sources=True)
assert stream["tokenizer_identity_sha256"] == sys.argv[2]
assert stream["sequence_length"] == 2048
assert stream["sequences"] == 244140
assert stream["valid_tokens"] == 499998720
assert stream["vocab_size"] == 248320
assert stream["eos_token_id"] == 248044
assert set(stream["prefix_utf8_bytes"]) == {"256", "61035", "122070", "183105", "244140"}
assert stream["benchmark_disjoint"] is True
assert stream["cross_document_targets_masked"] is True
PY
