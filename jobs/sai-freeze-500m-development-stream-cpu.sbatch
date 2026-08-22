#!/bin/bash
#SBATCH --job-name=sai-dev-stream-500m
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${TOKENIZER_ROOT:?TOKENIZER_ROOT is required}"
: "${SELECTED_RECEIPT:?SELECTED_RECEIPT is required}"
: "${DEVELOPMENT_CORPUS:?DEVELOPMENT_CORPUS is required}"
: "${SPLIT_RECEIPT:?SPLIT_RECEIPT is required}"
: "${DEVELOPMENT_STREAM:?DEVELOPMENT_STREAM is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

for path in "$SAI_ROOT" "$PYTHON" "$TOKENIZER_ROOT" "$SELECTED_RECEIPT" \
  "$DEVELOPMENT_CORPUS" "$SPLIT_RECEIPT" "$DEVELOPMENT_STREAM" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$TOKENIZER_ROOT"
test ! -L "$TOKENIZER_ROOT"
for path in "$SELECTED_RECEIPT" "$DEVELOPMENT_CORPUS" "$SPLIT_RECEIPT"; do
  test -f "$path"
  test ! -L "$path"
done
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test ! -e "$DEVELOPMENT_STREAM"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false

read -r tokenizer_sha256 split_sha256 < <(
  "$PYTHON" -P -s -B - \
    "$SELECTED_RECEIPT" "$SPLIT_RECEIPT" "$DEVELOPMENT_CORPUS" \
    "$SLURM_CPUS_PER_TASK" <<'PY'
import json
import pathlib
import sys

from sai.data.curriculum_split import validate_curriculum_split
from sai.data.token_stream import sha256_file

selected_path, split_path, development = map(pathlib.Path, sys.argv[1:4])
curriculum_workers = int(sys.argv[4])
selected = json.loads(selected_path.read_text())
assert selected["schema"] == "sai-tokenizer-qualification-receipt-v1"
assert selected["status"] == "qualified"
assert selected["vocab_size"] == 48_000
assert selected["byte_fallback"] is True
assert selected["roundtrip_failures"] == 0
split = validate_curriculum_split(
    split_path, curriculum_workers=curriculum_workers
)
row = split["development"]
assert row["path"] == str(development.resolve())
assert row["bytes"] == development.stat().st_size
assert row["sha256"] == sha256_file(development)
assert all(phase["documents"] > 0 for phase in row["phases"].values())
print(selected["tokenizer_identity_sha256"], sha256_file(split_path))
PY
)
test "${#tokenizer_sha256}" = 64
test "${#split_sha256}" = 64

"$PYTHON" -P -s -B -m sai.data.token_stream \
  --tokenizer-root "$TOKENIZER_ROOT" \
  --expected-tokenizer-sha256 "$tokenizer_sha256" \
  --source "$DEVELOPMENT_CORPUS" \
  --source-qualification-sha256 "$split_sha256" \
  --curriculum-receipt "$SPLIT_RECEIPT" \
  --curriculum-validation-workers "$SLURM_CPUS_PER_TASK" \
  --curriculum-population development \
  --output "$DEVELOPMENT_STREAM" \
  --sequence-length 2048 \
  --prefix-sequences 1024 \
  --curriculum-phase-sequences grounding=256 \
  --curriculum-phase-sequences integration=256 \
  --curriculum-phase-sequences reasoning=256 \
  --curriculum-phase-sequences specialization=256 \
  --require-all-curriculum-phases-at-prefix 1024

"$PYTHON" -P -s -B - "$DEVELOPMENT_STREAM" "$tokenizer_sha256" "$split_sha256" <<'PY'
import pathlib
import sys

from sai.data.token_stream import validate_frozen_stream

stream = validate_frozen_stream(pathlib.Path(sys.argv[1]), verify_sources=True)
assert stream["tokenizer_identity_sha256"] == sys.argv[2]
assert stream["source_qualification_sha256"] == sys.argv[3]
assert stream["sequence_length"] == 2_048
assert stream["sequences"] == 1_024
assert stream["benchmark_disjoint"] is True
assert stream["cross_document_targets_masked"] is True
curriculum = stream["curriculum"]
expected_phases = {
    "grounding": 256,
    "integration": 256,
    "reasoning": 256,
    "specialization": 256,
}
assert curriculum["phase_order"] == list(expected_phases)
assert curriculum["phase_token_budget_enforced"] is True
assert curriculum["phase_sequence_targets"] == expected_phases
assert curriculum["phase_sequences_emitted"] == expected_phases
assert curriculum["required_all_phase_prefixes"] == [1_024]
assert curriculum["all_required_prefixes_cover_every_phase"] is True
assert curriculum["phase_order_contiguous_at_every_prefix"] is True
assert all(
    curriculum["prefixes"]["1024"][phase]["tokens"] == 256 * 2_048
    and curriculum["prefixes"]["1024"][phase]["utf8_bytes"] > 0
    for phase in expected_phases
)
PY
