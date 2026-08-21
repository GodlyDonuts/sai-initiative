#!/bin/bash
#SBATCH --job-name=sai-fineweb-500m-cpu
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
: "${MANIFEST:?MANIFEST is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${OUTPUT:?OUTPUT is required}"
: "${RECEIPT:?RECEIPT is required}"
: "${SLURM_JOB_ID:?SLURM_JOB_ID is required}"

require_absolute() {
  case "$1" in
    /*) ;;
    *) echo "path must be absolute: $1" >&2; exit 2 ;;
  esac
}

for path in "$SAI_ROOT" "$PYTHON" "$MANIFEST" "$LOG_ROOT" "$OUTPUT" \
  "$RECEIPT"; do
  require_absolute "$path"
done

case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU source acquisition was exposed to a GPU" >&2; exit 2 ;;
esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -f "$MANIFEST"
test ! -L "$MANIFEST"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test ! -e "$OUTPUT"
test ! -e "$RECEIPT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

manifest_values="$("$PYTHON" -P -s -B - "$MANIFEST" <<'PY'
import json
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
files = payload["files"]
assert payload["schema"] == "sai-pinned-hf-file-selection-v1"
assert payload["dataset"] == "HuggingFaceFW/fineweb-edu"
assert payload["revision"] == "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
assert payload["selection"] == {
    "method": "complete_prefix_sorted_tree",
    "selected_count": 30,
}
assert len(files) == 30
assert files == sorted(files, key=lambda row: row["path"])
assert [row["path"] for row in files] == [
    f"sample/100BT/{group:03d}_{index:05d}.parquet"
    for group in range(3)
    for index in range(10)
]
assert all(set(row) == {"path", "sha256", "size"} for row in files)
assert all(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) for row in files)
assert all(isinstance(row["size"], int) and row["size"] > 0 for row in files)
selected_bytes = sum(row["size"] for row in files)
assert selected_bytes == payload["selected_bytes"] == 64_562_434_300
print(selected_bytes)
PY
)"
test "$manifest_values" = "64562434300"

scratch_parent="${SLURM_TMPDIR:-/tmp}"
require_absolute "$scratch_parent"
test -d "$scratch_parent"
test ! -L "$scratch_parent"
available_bytes="$(( $(df -Pk "$scratch_parent" | awk 'NR==2 {print $4}') * 1024 ))"
required_bytes=75100000000
if (( available_bytes < required_bytes )); then
  echo "node-local scratch cannot hold the pinned FineWeb prefix" >&2
  exit 2
fi

umask 077
SCRATCH_ROOT="$(mktemp -d "$scratch_parent/sai-fineweb-edu-500m-$SLURM_JOB_ID.XXXXXX")"
case "$SCRATCH_ROOT" in
  "$scratch_parent"/sai-fineweb-edu-500m-"$SLURM_JOB_ID".??????) ;;
  *) echo "unsafe node-local scratch path" >&2; exit 2 ;;
esac
test -d "$SCRATCH_ROOT"
test ! -L "$SCRATCH_ROOT"
test "$(stat -c %u "$SCRATCH_ROOT")" = "$(id -u)"
test "$(stat -c %a "$SCRATCH_ROOT")" = "700"
SOURCE_ROOT="$SCRATCH_ROOT/source"
cleanup_source() {
  case "$SCRATCH_ROOT" in
    "$scratch_parent"/sai-fineweb-edu-500m-"$SLURM_JOB_ID".??????) ;;
    *) echo "unsafe node-local cleanup target" >&2; return 2 ;;
  esac
  if test -e "$SCRATCH_ROOT"; then
    test -d "$SCRATCH_ROOT"
    test ! -L "$SCRATCH_ROOT"
    test "$(stat -c %u "$SCRATCH_ROOT")" = "$(id -u)"
    find "$SCRATCH_ROOT" -xdev -depth -delete
  fi
}
trap cleanup_source EXIT

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

"$PYTHON" -P -s -B -m sai.data.fineweb_edu \
  --manifest "$MANIFEST" \
  --source-root "$SOURCE_ROOT" \
  --output "$OUTPUT" \
  --receipt "$RECEIPT" \
  --acquire

"$PYTHON" -P -s -B - "$OUTPUT" "$RECEIPT" <<'PY'
import hashlib
import json
import pathlib
import sys

output, receipt = map(pathlib.Path, sys.argv[1:])
assert output.is_file() and not output.is_symlink() and output.stat().st_size > 0
assert receipt.is_file() and not receipt.is_symlink()
payload = json.loads(receipt.read_text())
assert payload["schema"] == "sai-fineweb-edu-conversion-receipt-v1"
assert payload["status"] == "passed"
assert len(payload["source_receipts"]) == 30
assert payload["output"]["path"] == str(output.resolve())
assert payload["output"]["bytes"] == output.stat().st_size
assert payload["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
PY

cleanup_source
trap - EXIT
