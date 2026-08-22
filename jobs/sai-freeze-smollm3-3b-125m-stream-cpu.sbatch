#!/usr/bin/env bash
#SBATCH --job-name=sai-smollm3-stream-125m
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_MANIFEST:?MODEL_MANIFEST is required}"
: "${RESTORATION_RECEIPT:?RESTORATION_RECEIPT is required}"
: "${TRAIN_CORPUS:?TRAIN_CORPUS is required}"
: "${TRAIN_CORPUS_RECEIPT:?TRAIN_CORPUS_RECEIPT is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$MODEL_ROOT"
test -f "$MODEL_MANIFEST"
test -f "$RESTORATION_RECEIPT"
test -f "$TRAIN_CORPUS"
test -f "$TRAIN_CORPUS_RECEIPT"
test ! -e "$TRAIN_STREAM"

case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU Smol stream freezer was exposed to a GPU" >&2; exit 2 ;;
esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

tokenizer_identity="$($PYTHON -P -s -B - <<'PY' \
  "$MODEL_ROOT" "$MODEL_MANIFEST" "$RESTORATION_RECEIPT" \
  "$TRAIN_CORPUS" "$TRAIN_CORPUS_RECEIPT"
import json
import pathlib
import sys

from sai.data.external_hf_snapshot import validate_external_snapshot
from sai.data.token_stream import sha256_file, sha256_tree
from sai.evaluation.hf_smol_parent import SNAPSHOT_SPEC

model_root, manifest, restoration, corpus, corpus_receipt = map(pathlib.Path, sys.argv[1:])
validate_external_snapshot(
    model_root,
    manifest_path=manifest,
    receipt_path=restoration,
    spec=SNAPSHOT_SPEC,
)
receipt = json.loads(corpus_receipt.read_text())
assert receipt["schema"] == "sai-decontamination-receipt-v1"
assert receipt["status"] == "passed"
assert receipt["output"]["path"] == str(corpus.resolve())
assert receipt["output"]["bytes"] == corpus.stat().st_size
assert receipt["output"]["sha256"] == sha256_file(corpus)
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
  --prefix-sequences 61035

"$PYTHON" -P -s -B - <<'PY' "$TRAIN_STREAM" "$tokenizer_identity"
import pathlib
import sys

from sai.data.token_stream import validate_frozen_stream

stream = validate_frozen_stream(pathlib.Path(sys.argv[1]), verify_sources=True)
assert stream["tokenizer_identity_sha256"] == sys.argv[2]
assert stream["sequence_length"] == 2048
assert stream["sequences"] == 61035
assert stream["valid_tokens"] == 124999680
assert stream["vocab_size"] == 128256
assert stream["eos_token_id"] == 128012
assert set(stream["prefix_utf8_bytes"]) == {"256", "61035"}
assert stream["benchmark_disjoint"] is True
assert stream["cross_document_targets_masked"] is True
PY
