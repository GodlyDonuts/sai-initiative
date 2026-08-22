#!/bin/bash
#SBATCH --job-name=sai-stream-500m-cpu
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${TOKENIZER_ROOT:?TOKENIZER_ROOT is required}"
: "${SELECTED_RECEIPT:?SELECTED_RECEIPT is required}"
: "${TRAIN_CORPUS:?TRAIN_CORPUS is required}"
: "${CURRICULUM_RECEIPT:?CURRICULUM_RECEIPT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

require_absolute() {
  case "$1" in
    /*) ;;
    *) echo "path must be absolute: $1" >&2; exit 2 ;;
  esac
}

for path in "$SAI_ROOT" "$PYTHON" "$TOKENIZER_ROOT" "$SELECTED_RECEIPT" \
  "$TRAIN_CORPUS" "$CURRICULUM_RECEIPT" "$LOG_ROOT" "$TRAIN_STREAM"; do
  require_absolute "$path"
done

case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU stream freezer was exposed to a GPU" >&2; exit 2 ;;
esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$TOKENIZER_ROOT"
test ! -L "$TOKENIZER_ROOT"
for path in "$SELECTED_RECEIPT" "$TRAIN_CORPUS" "$CURRICULUM_RECEIPT"; do
  test -f "$path"
  test ! -L "$path"
done
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test ! -e "$TRAIN_STREAM"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

tokenizer_sha256="$("$PYTHON" -P -s -B - \
  "$SELECTED_RECEIPT" "$TRAIN_CORPUS" "$CURRICULUM_RECEIPT" <<'PY'
import json
import pathlib
import sys

from sai.data.curriculum_split import validate_curriculum_split
from sai.data.token_stream import sha256_file

selected_path, corpus_path, corpus_receipt_path = map(pathlib.Path, sys.argv[1:])
selected = json.loads(selected_path.read_text())
assert selected["schema"] == "sai-tokenizer-qualification-receipt-v1"
assert selected["status"] == "qualified"
assert selected["vocab_size"] == 48_000
assert selected["byte_fallback"] is True
assert selected["roundtrip_failures"] == 0
assert selected["special_tokens_preserved"] is True
receipt = validate_curriculum_split(corpus_receipt_path)
train = receipt["train"]
assert receipt["status"] == "qualified"
assert receipt["split_qualified"] is True
assert receipt["training_authorized"] is False
assert receipt["four_b_training_authorized"] is False
assert train["documents"] >= 40_000
assert train["curriculum_qualified"] is True
assert all(train["progression_checks"].values())
assert train["path"] == str(corpus_path.resolve())
assert train["bytes"] == corpus_path.stat().st_size
assert train["sha256"] == sha256_file(corpus_path)
print(selected["tokenizer_identity_sha256"])
PY
)"
test "${#tokenizer_sha256}" = 64
curriculum_sha256="$(sha256sum "$CURRICULUM_RECEIPT" | cut -d ' ' -f 1)"
test "${#curriculum_sha256}" = 64

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false

"$PYTHON" -P -s -B -m sai.data.token_stream \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --expected-tokenizer-sha256 "$tokenizer_sha256" \
  --source "$TRAIN_CORPUS" \
  --source-qualification-sha256 "$curriculum_sha256" \
  --curriculum-receipt "$CURRICULUM_RECEIPT" \
  --output "$TRAIN_STREAM" \
  --sequence-length 2048 \
  --prefix-sequences 256 \
  --prefix-sequences 61035 \
  --prefix-sequences 122070 \
  --prefix-sequences 183105 \
  --prefix-sequences 244140 \
  --require-all-curriculum-phases-at-prefix 61035 \
  --require-all-curriculum-phases-at-prefix 122070 \
  --require-all-curriculum-phases-at-prefix 183105 \
  --require-all-curriculum-phases-at-prefix 244140

"$PYTHON" -P -s -B - "$TRAIN_STREAM" "$tokenizer_sha256" "$curriculum_sha256" <<'PY'
import pathlib
import sys

from sai.data.token_stream import validate_frozen_stream

stream = validate_frozen_stream(pathlib.Path(sys.argv[1]), verify_sources=True)
assert stream["tokenizer_identity_sha256"] == sys.argv[2]
assert stream["source_qualification_sha256"] == sys.argv[3]
assert stream["sequence_length"] == 2_048
assert stream["sequences"] == 244_140
assert stream["valid_tokens"] == 499_998_720
assert set(stream["prefix_utf8_bytes"]) == {
    "256",
    "61035",
    "122070",
    "183105",
    "244140",
}
assert stream["benchmark_disjoint"] is True
assert stream["cross_document_targets_masked"] is True
assert stream["curriculum"]["required_all_phase_prefixes"] == [
    61_035,
    122_070,
    183_105,
    244_140,
]
assert stream["curriculum"]["all_required_prefixes_cover_every_phase"] is True
for prefix in stream["curriculum"]["required_all_phase_prefixes"]:
    assert all(
        row["tokens"] > 0 and row["utf8_bytes"] > 0
        for row in stream["curriculum"]["prefixes"][str(prefix)].values()
    )
PY
