#!/usr/bin/env bash
#SBATCH --job-name=sai-launch-smol3-workspace
#SBATCH --partition=normal
#SBATCH --account=skattel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FAST_KERNELS_ROOT:?FAST_KERNELS_ROOT is required}"
: "${MODEL_ROOT:?MODEL_ROOT is required}"
: "${MODEL_MANIFEST:?MODEL_MANIFEST is required}"
: "${RESTORATION_RECEIPT:?RESTORATION_RECEIPT is required}"
: "${MECHANICS_JOB_ID:?MECHANICS_JOB_ID is required}"
: "${MECHANICS_RECEIPT:?MECHANICS_RECEIPT is required}"
: "${STREAM_JOB_ID:?STREAM_JOB_ID is required}"
: "${TRAIN_STREAM:?TRAIN_STREAM is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test "$(sacct -j "$MECHANICS_JOB_ID" -X -n -P -o State | sed -n '1p')" = COMPLETED
test "$(sacct -j "$STREAM_JOB_ID" -X -n -P -o State | sed -n '1p')" = COMPLETED
test -f "$MECHANICS_RECEIPT"
test -d "$TRAIN_STREAM"
test -d "$MODEL_ROOT"
test -f "$MODEL_MANIFEST"
test ! -L "$MODEL_MANIFEST"
test -f "$RESTORATION_RECEIPT"
test ! -L "$RESTORATION_RECEIPT"
test -d "$FAST_KERNELS_ROOT"
test -d "$LOG_ROOT"
test ! -e "$RUN_ROOT"
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

read -r mechanics_sha train_identity code_sha environment_sha < <(
  "$PYTHON" -P -s -B - <<'PY' \
    "$MECHANICS_RECEIPT" "$TRAIN_STREAM" "$EXPECTED_COMMIT" "$PYTHON" \
    "$MODEL_ROOT" "$MODEL_MANIFEST" "$RESTORATION_RECEIPT"
import hashlib
import json
import os
import pathlib
import subprocess
import sys

from sai.data.token_stream import canonical_sha256, validate_frozen_stream
from sai.evaluation.hf_smol_parent import validate_smol_mechanics_receipt

mechanics_path = pathlib.Path(sys.argv[1])
stream_root = pathlib.Path(sys.argv[2])
commit = sys.argv[3]
python_path = pathlib.Path(os.path.realpath(sys.argv[4]))
model_root = pathlib.Path(sys.argv[5])
manifest = pathlib.Path(sys.argv[6])
restoration = pathlib.Path(sys.argv[7])
mechanics_sha = hashlib.sha256(mechanics_path.read_bytes()).hexdigest()
mechanics = validate_smol_mechanics_receipt(
    mechanics_path,
    expected_file_sha256=mechanics_sha,
    model_root=model_root,
    manifest_path=manifest,
    restoration_receipt_path=restoration,
)
stream = validate_frozen_stream(stream_root, verify_sources=True)
if (
    stream["sequence_length"] != 2048
    or stream["vocab_size"] != 128256
    or stream["eos_token_id"] != 128012
):
    raise SystemExit("Smol workspace stream geometry differs")
if not all(str(value) in stream["prefix_utf8_bytes"] for value in (256, 61035)):
    raise SystemExit("Smol workspace prefixes differ")
tree = subprocess.check_output(
    ["git", "-C", os.environ["SAI_ROOT"], "ls-tree", "-r", "--full-tree", commit]
)
code_sha = hashlib.sha256(tree).hexdigest()
python_sha = hashlib.sha256(python_path.read_bytes()).hexdigest()
environment_sha = canonical_sha256(
    {
        "mechanics_environment": mechanics["environment"],
        "mechanics_runtime": mechanics["runtime"],
        "python_resolved_path": str(python_path),
        "python_sha256": python_sha,
    }
)
print(mechanics_sha, stream["ordered_stream_identity_sha256"], code_sha, environment_sha)
PY
)

mkdir -m 700 "$RUN_ROOT"
admitted_jobs=()
cleanup_partial_submission() {
  if ((${#admitted_jobs[@]})); then
    scancel "${admitted_jobs[@]}" || true
  fi
}
trap cleanup_partial_submission ERR

submit_arm() {
  local mode="$1" sequences="$2" dependency="$3" time_limit="$4" label="$5"
  local root="$RUN_ROOT/$label/$mode"
  mkdir -p -m 700 "$root"
  local dependency_args=()
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
  fi
  env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
    --time="$time_limit" \
    "${dependency_args[@]}" \
    --job-name="sai-smol3-${label}-${mode}" \
    --output="$LOG_ROOT/sai_smol3_${label}_${mode}_%j.out" \
    --error="$LOG_ROOT/sai_smol3_${label}_${mode}_%j.err" \
    --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FAST_KERNELS_ROOT="$FAST_KERNELS_ROOT",MODEL_ROOT="$MODEL_ROOT",MODEL_MANIFEST="$MODEL_MANIFEST",RESTORATION_RECEIPT="$RESTORATION_RECEIPT",MECHANICS_RECEIPT="$MECHANICS_RECEIPT",MECHANICS_RECEIPT_SHA256="$mechanics_sha",TRAIN_STREAM="$TRAIN_STREAM",TRAIN_IDENTITY="$train_identity",STATE_MODE="$mode",TRAINING_SEQUENCES="$sequences",CODE_SHA256="$code_sha",ENVIRONMENT_SHA256="$environment_sha",CHECKPOINT="$root/checkpoint.pt",OUTPUT="$root/result.json",CHECKPOINT_INTERVAL=100,RESUME=0 \
    "$SAI_ROOT/jobs/sai-smollm3-3b-workspace-screen-single-h100.sbatch"
}

canary_recurrent="$(submit_arm recurrent 256 '' 01:00:00 canary)"
admitted_jobs+=("$canary_recurrent")
canary_reset="$(submit_arm reset_average 256 '' 01:00:00 canary)"
admitted_jobs+=("$canary_reset")
canary_dependency="$canary_recurrent:$canary_reset"
full_recurrent="$(submit_arm recurrent 61035 "$canary_dependency" 08:00:00 full)"
admitted_jobs+=("$full_recurrent")
full_reset="$(submit_arm reset_average 61035 "$canary_dependency" 08:00:00 full)"
admitted_jobs+=("$full_reset")

"$PYTHON" -P -s -B - <<'PY' \
  "$RUN_ROOT/dispatch.json" "$EXPECTED_COMMIT" "$MECHANICS_JOB_ID" \
  "$STREAM_JOB_ID" "$mechanics_sha" "$train_identity" "$code_sha" \
  "$environment_sha" "$canary_recurrent" "$canary_reset" \
  "$full_recurrent" "$full_reset"
import hashlib
import json
import os
import pathlib
import sys
import uuid

output = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sai-smollm3-3b-matched-workspace-dispatch-v1",
    "status": "submitted",
    "source_commit": sys.argv[2],
    "mechanics_job": int(sys.argv[3]),
    "stream_job": int(sys.argv[4]),
    "mechanics_receipt_file_sha256": sys.argv[5],
    "training_stream_identity_sha256": sys.argv[6],
    "code_sha256": sys.argv[7],
    "environment_sha256": sys.argv[8],
    "canary_jobs": {
        "recurrent": int(sys.argv[9]),
        "reset_average": int(sys.argv[10]),
    },
    "full_jobs": {
        "recurrent": int(sys.argv[11]),
        "reset_average": int(sys.argv[12]),
    },
    "h100_jobs": 4,
    "maximum_concurrent_h100_jobs": 2,
    "one_h100_per_job": True,
    "full_jobs_require_both_canaries": True,
    "cross_family_confirmation": True,
    "requeue": False,
    "four_b_training_executed": False,
}
payload["receipt_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(descriptor, "wb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
os.link(temporary, output)
temporary.unlink()
PY

trap - ERR
printf 'canary=%s:%s full=%s:%s\n' \
  "$canary_recurrent" "$canary_reset" "$full_recurrent" "$full_reset"
