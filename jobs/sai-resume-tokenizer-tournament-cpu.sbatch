#!/bin/bash
#SBATCH --job-name=sai-resume-tokenizers
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${PRIOR_JOB_ID:?PRIOR_JOB_ID is required}"
: "${PRIOR_STDOUT:?PRIOR_STDOUT is required}"
: "${SOURCE_CORPUS:?SOURCE_CORPUS is required}"
: "${DECONTAMINATION_RECEIPT:?DECONTAMINATION_RECEIPT is required}"
: "${TRAIN_CORPUS:?TRAIN_CORPUS is required}"
: "${DEVELOPMENT_CORPUS:?DEVELOPMENT_CORPUS is required}"
: "${SPLIT_RECEIPT:?SPLIT_RECEIPT is required}"
: "${CANDIDATE_ROOT:?CANDIDATE_ROOT is required}"
: "${REPORT:?REPORT is required}"
: "${SELECTED_RECEIPT:?SELECTED_RECEIPT is required}"

for path in \
  "$SAI_ROOT" "$PYTHON" "$LOG_ROOT" "$PRIOR_STDOUT" "$SOURCE_CORPUS" \
  "$DECONTAMINATION_RECEIPT" "$TRAIN_CORPUS" "$DEVELOPMENT_CORPUS" \
  "$SPLIT_RECEIPT" "$CANDIDATE_ROOT" "$REPORT" "$SELECTED_RECEIPT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done

test -n "${SLURM_JOB_ID:-}"
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU tokenizer continuation was exposed to a GPU" >&2; exit 2 ;;
esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"

prior_state="$(sacct -j "$PRIOR_JOB_ID" -X -n -P -o State | sed -n '1p')"
case "$prior_state" in TIMEOUT|FAILED) ;; *) exit 2 ;; esac

for path in \
  "$PRIOR_STDOUT" "$SOURCE_CORPUS" "$DECONTAMINATION_RECEIPT" \
  "$TRAIN_CORPUS" "$DEVELOPMENT_CORPUS" "$SPLIT_RECEIPT"; do
  test -f "$path"
  test ! -L "$path"
done
test ! -e "$REPORT"
test ! -L "$REPORT"
test ! -e "$SELECTED_RECEIPT"
test ! -L "$SELECTED_RECEIPT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false

candidate_state="$($PYTHON -P -s -B - \
  "$PRIOR_STDOUT" "$SOURCE_CORPUS" "$DECONTAMINATION_RECEIPT" \
  "$TRAIN_CORPUS" "$DEVELOPMENT_CORPUS" "$SPLIT_RECEIPT" \
  "$CANDIDATE_ROOT" <<'PY'
import json
import pathlib
import sys

from sai.data.token_stream import canonical_sha256, sha256_file, sha256_tree


def safe_file(path):
    assert path.is_file() and not path.is_symlink()


(
    prior_stdout,
    source,
    decontamination_receipt,
    train,
    development,
    split_receipt,
    candidate_root,
) = map(pathlib.Path, sys.argv[1:])
for path in (
    prior_stdout,
    source,
    decontamination_receipt,
    train,
    development,
    split_receipt,
):
    safe_file(path)

decontamination = json.loads(decontamination_receipt.read_text())
decontamination_unsigned = {
    key: value
    for key, value in decontamination.items()
    if key != "receipt_sha256"
}
assert decontamination["schema"] == "sai-decontamination-receipt-v1"
assert decontamination["status"] == "passed"
assert decontamination["receipt_sha256"] == canonical_sha256(
    decontamination_unsigned
)
assert decontamination["output"]["path"] == str(source.resolve())
assert decontamination["output"]["bytes"] == source.stat().st_size
assert decontamination["output"]["sha256"] == sha256_file(source)

split = json.loads(split_receipt.read_text())
split_unsigned = {key: value for key, value in split.items() if key != "receipt_sha256"}
assert split["schema"] == "sai-document-split-receipt-v1"
assert split["status"] == "complete"
assert split["receipt_sha256"] == canonical_sha256(split_unsigned)
assert split["scientific_promotion_allowed"] is False
assert split["near_duplicate_cluster_split_qualified"] is False
for label, path in (("source", source), ("train", train), ("development", development)):
    entry = split[label]
    assert entry["path"] == str(path.resolve())
    assert entry["bytes"] == path.stat().st_size
    assert entry["sha256"] == sha256_file(path)

observed = [json.loads(line) for line in prior_stdout.read_text().splitlines() if line]
assert {
    "status": "passed",
    "receipt_sha256": decontamination["receipt_sha256"],
} in observed
assert {"status": "complete", "receipt_sha256": split["receipt_sha256"]} in observed

if not candidate_root.exists():
    print("build_required")
    raise SystemExit(0)
assert candidate_root.is_dir() and not candidate_root.is_symlink()
manifest_path = candidate_root / "manifest.json"
safe_file(manifest_path)
manifest = json.loads(manifest_path.read_text())
unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
assert manifest["schema"] == "sai-tokenizer-build-manifest-v1"
assert manifest["status"] == "complete"
assert manifest["training_authorized"] is False
assert manifest["manifest_sha256"] == canonical_sha256(unsigned)
assert len(manifest["source_receipts"]) == 1
source_entry = manifest["source_receipts"][0]
assert source_entry["path"] == str(train.resolve())
assert source_entry["bytes"] == train.stat().st_size
assert source_entry["sha256"] == sha256_file(train)
expected_sizes = {"32k": 32_000, "48k": 48_000, "64k": 64_000}
assert set(manifest["candidates"]) == set(expected_sizes)
for name, size in expected_sizes.items():
    entry = manifest["candidates"][name]
    root = candidate_root / name
    assert entry["root"] == name
    assert entry["vocab_size"] == size
    assert root.is_dir() and not root.is_symlink()
    assert entry["tree_sha256"] == sha256_tree(root)
print("qualification_required")
PY
)"

case "$candidate_state" in
  build_required)
    "$PYTHON" -P -s -B -m sai.tokenizer.build \
      --corpus "$TRAIN_CORPUS" \
      --output-root "$CANDIDATE_ROOT"
    ;;
  qualification_required) ;;
  *) exit 2 ;;
esac

exec "$PYTHON" -P -s -B -m sai.tokenizer.qualification \
  --candidate "32k=$CANDIDATE_ROOT/32k" \
  --candidate "48k=$CANDIDATE_ROOT/48k" \
  --candidate "64k=$CANDIDATE_ROOT/64k" \
  --corpus "$TRAIN_CORPUS" \
  --protected-suite "$SAI_ROOT/docs/SAI_TOKENIZER_PROTECTED_SUITE.jsonl" \
  --output "$REPORT" \
  --selected-48k-output "$SELECTED_RECEIPT"
