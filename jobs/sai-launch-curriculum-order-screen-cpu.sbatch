#!/bin/bash
#SBATCH --job-name=sai-launch-curr-order
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${GEOMETRY:?GEOMETRY is required}"
: "${CURRICULUM_STREAM:?CURRICULUM_STREAM is required}"
: "${CONTROL_STREAM:?CONTROL_STREAM is required}"
: "${DEVELOPMENT_STREAM:?DEVELOPMENT_STREAM is required}"
: "${SPLIT_RECEIPT:?SPLIT_RECEIPT is required}"
: "${ENVIRONMENT_RECEIPT:?ENVIRONMENT_RECEIPT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"

for path in "$SAI_ROOT" "$PYTHON" "$FLA_ROOT" "$GEOMETRY" \
  "$CURRICULUM_STREAM" "$CONTROL_STREAM" "$DEVELOPMENT_STREAM" \
  "$SPLIT_RECEIPT" "$ENVIRONMENT_RECEIPT" "$OUTPUT_ROOT" "$LOG_ROOT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -f "$GEOMETRY"
for path in "$CURRICULUM_STREAM" "$CONTROL_STREAM" "$DEVELOPMENT_STREAM"; do
  test -d "$path"
  test ! -L "$path"
done
for path in "$SPLIT_RECEIPT" "$ENVIRONMENT_RECEIPT"; do
  test -f "$path"
  test ! -L "$path"
done
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test ! -e "$OUTPUT_ROOT"

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

MIN_QUOTA_HEADROOM_KIB=25165824
MIN_QUOTA_HEADROOM_FILES=10000
read -r quota_used_kib quota_hard_kib quota_files quota_hard_files < <(
  /usr/bin/lfs quota -u "$(id -u)" /lustre/fs1 |
    awk 'END {print $2, $4, $6, $8}'
)
for value in "$quota_used_kib" "$quota_hard_kib" \
  "$quota_files" "$quota_hard_files"; do
  [[ "$value" =~ ^[0-9]+$ ]]
done
quota_headroom_kib=$((quota_hard_kib - quota_used_kib))
quota_headroom_files=$((quota_hard_files - quota_files))
test "$quota_headroom_kib" -ge "$MIN_QUOTA_HEADROOM_KIB"
test "$quota_headroom_files" -ge "$MIN_QUOTA_HEADROOM_FILES"

read -r curriculum_identity control_identity development_identity < <(
  "$PYTHON" -P -s -B - \
    "$CURRICULUM_STREAM" "$CONTROL_STREAM" "$DEVELOPMENT_STREAM" \
    "$SPLIT_RECEIPT" "$SLURM_CPUS_PER_TASK" <<'PY'
import pathlib
import sys

from sai.data.curriculum_control import validate_curriculum_order_bundle

curriculum_root, control_root, development_root, split_path = map(
    pathlib.Path, sys.argv[1:5]
)
curriculum_workers = int(sys.argv[5])
bundle = validate_curriculum_order_bundle(
    curriculum_root,
    control_root,
    development_root,
    split_path,
    curriculum_workers=curriculum_workers,
)
split = bundle["split"]
curriculum = bundle["curriculum"]
control = bundle["control"]
development = bundle["development"]
assert curriculum["sequence_length"] == control["sequence_length"] == 2_048
assert development["sequence_length"] == 2_048
assert curriculum["sequences"] == control["sequences"] == 244_140
assert development["sequences"] == 1_024
assert curriculum["valid_tokens"] == control["valid_tokens"] == 499_998_720
assert curriculum["admitted_utf8_bytes"] == control["admitted_utf8_bytes"]
assert curriculum["prefix_utf8_bytes"]["256"] > 0
assert control["prefix_utf8_bytes"] == {
    "244140": curriculum["admitted_utf8_bytes"]
}
assert curriculum["tokenizer_identity_sha256"] == control["tokenizer_identity_sha256"]
assert curriculum["tokenizer_identity_sha256"] == development["tokenizer_identity_sha256"]
ordering = control["ordering_control"]
assert ordering["parent_stream"]["ordered_stream_identity_sha256"] == curriculum["ordered_stream_identity_sha256"]
assert ordering["same_tokens_and_boundary_masks"] is True
assert ordering["same_sequence_multiset"] is True
assert ordering["only_sequence_order_changed"] is True
assert curriculum["ordered_stream_identity_sha256"] != control["ordered_stream_identity_sha256"]
assert split["train"]["path"] == curriculum["source_receipts"][0]["path"]
assert split["development"]["path"] == development["source_receipts"][0]["path"]
print(
    curriculum["ordered_stream_identity_sha256"],
    control["ordered_stream_identity_sha256"],
    development["ordered_stream_identity_sha256"],
)
PY
)

mkdir "$OUTPUT_ROOT"
canary_checkpoint="$OUTPUT_ROOT/shared_gqa.canary.checkpoint.pt"
canary_result="$OUTPUT_ROOT/shared_gqa.canary.result.json"
curriculum_checkpoint="$OUTPUT_ROOT/curriculum.checkpoint.pt"
curriculum_result="$OUTPUT_ROOT/curriculum.result.json"
control_checkpoint="$OUTPUT_ROOT/order_control.checkpoint.pt"
control_result="$OUTPUT_ROOT/order_control.result.json"
comparison="$OUTPUT_ROOT/comparison.json"
job_ids=()
cancel_partial_graph() {
  local job_id
  for job_id in "${job_ids[@]}"; do
    if [[ "$job_id" =~ ^[0-9]+$ ]]; then scancel "$job_id" || true; fi
  done
}
trap cancel_partial_graph ERR

shared="SAI_ROOT=$SAI_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT,PYTHON=$PYTHON,FLA_ROOT=$FLA_ROOT,GEOMETRY=$GEOMETRY,FAMILY=gated_gqa,DEVELOPMENT_STREAM=$DEVELOPMENT_STREAM,DEVELOPMENT_IDENTITY=$development_identity,SEQUENCES_PER_UPDATE=256,BATCH_SIZE=8,SEED=20260821,ENVIRONMENT_RECEIPT=$ENVIRONMENT_RECEIPT,MIN_QUOTA_HEADROOM_KIB=$MIN_QUOTA_HEADROOM_KIB,MIN_QUOTA_HEADROOM_FILES=$MIN_QUOTA_HEADROOM_FILES"
canary_job="$({
  env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
    --chdir="$SAI_ROOT" --time=00:45:00 \
    --output="$LOG_ROOT/sai_curriculum_order_canary_%j.out" \
    --error="$LOG_ROOT/sai_curriculum_order_canary_%j.err" \
    --export="$shared,TRAIN_STREAM=$CURRICULUM_STREAM,TRAIN_IDENTITY=$curriculum_identity,DEVELOPMENT_SEQUENCES=1,OPTIMIZER_STEPS=1,TRAINING_SEQUENCES=256,DEVELOPMENT_BATCH_SIZE=1,CHECKPOINT_INTERVAL=1,MECHANICS_ONLY=1,CHECKPOINT=$canary_checkpoint,OUTPUT=$canary_result" \
    "$SAI_ROOT/jobs/sai-short-screen-single-h100.sbatch"
})"
[[ "$canary_job" =~ ^[0-9]+$ ]]
job_ids+=("$canary_job")

curriculum_job="$({
  env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
    --chdir="$SAI_ROOT" --dependency="afterok:$canary_job" \
    --time=18:00:00 \
    --kill-on-invalid-dep=yes \
    --output="$LOG_ROOT/sai_curriculum_500m_%j.out" \
    --error="$LOG_ROOT/sai_curriculum_500m_%j.err" \
    --export="$shared,TRAIN_STREAM=$CURRICULUM_STREAM,TRAIN_IDENTITY=$curriculum_identity,DEVELOPMENT_SEQUENCES=1024,OPTIMIZER_STEPS=954,TRAINING_SEQUENCES=244140,DEVELOPMENT_BATCH_SIZE=8,CHECKPOINT_INTERVAL=10,CHECKPOINT=$curriculum_checkpoint,OUTPUT=$curriculum_result" \
    "$SAI_ROOT/jobs/sai-short-screen-single-h100.sbatch"
})"
[[ "$curriculum_job" =~ ^[0-9]+$ ]]
job_ids+=("$curriculum_job")

control_job="$({
  env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
    --chdir="$SAI_ROOT" --dependency="afterok:$canary_job" \
    --time=18:00:00 \
    --kill-on-invalid-dep=yes \
    --output="$LOG_ROOT/sai_order_control_500m_%j.out" \
    --error="$LOG_ROOT/sai_order_control_500m_%j.err" \
    --export="$shared,TRAIN_STREAM=$CONTROL_STREAM,TRAIN_IDENTITY=$control_identity,DEVELOPMENT_SEQUENCES=1024,OPTIMIZER_STEPS=954,TRAINING_SEQUENCES=244140,DEVELOPMENT_BATCH_SIZE=8,CHECKPOINT_INTERVAL=10,CHECKPOINT=$control_checkpoint,OUTPUT=$control_result" \
    "$SAI_ROOT/jobs/sai-short-screen-single-h100.sbatch"
})"
[[ "$control_job" =~ ^[0-9]+$ ]]
job_ids+=("$control_job")

compare_job="$({
  env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable --time=04:00:00 \
    --chdir="$SAI_ROOT" --dependency="afterok:$curriculum_job:$control_job" \
    --kill-on-invalid-dep=yes \
    --output="$LOG_ROOT/sai_compare_curriculum_order_%j.out" \
    --error="$LOG_ROOT/sai_compare_curriculum_order_%j.err" \
    --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",CURRICULUM_JOB_ID="$curriculum_job",CONTROL_JOB_ID="$control_job",CURRICULUM_RESULT="$curriculum_result",CONTROL_RESULT="$control_result",CURRICULUM_STREAM="$CURRICULUM_STREAM",CONTROL_STREAM="$CONTROL_STREAM",DEVELOPMENT_STREAM="$DEVELOPMENT_STREAM",SPLIT_RECEIPT="$SPLIT_RECEIPT",OUTPUT="$comparison",LOG_ROOT="$LOG_ROOT" \
    "$SAI_ROOT/jobs/sai-compare-curriculum-order-cpu.sbatch"
})"
[[ "$compare_job" =~ ^[0-9]+$ ]]
job_ids+=("$compare_job")

dispatch="$OUTPUT_ROOT/dispatch.json"
"$PYTHON" -P -s -B - "$dispatch" "$EXPECTED_COMMIT" \
  "$curriculum_identity" "$control_identity" "$development_identity" \
  "$canary_job" "$curriculum_job" "$control_job" "$compare_job" \
  "$quota_used_kib" "$quota_hard_kib" "$quota_headroom_kib" \
  "$quota_files" "$quota_hard_files" "$quota_headroom_files" \
  "$MIN_QUOTA_HEADROOM_KIB" "$MIN_QUOTA_HEADROOM_FILES" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sai-curriculum-order-screen-dispatch-v1",
    "status": "submitted",
    "source_commit": sys.argv[2],
    "curriculum_stream_identity_sha256": sys.argv[3],
    "control_stream_identity_sha256": sys.argv[4],
    "development_stream_identity_sha256": sys.argv[5],
    "canary_job_id": int(sys.argv[6]),
    "curriculum_job_id": int(sys.argv[7]),
    "control_job_id": int(sys.argv[8]),
    "comparison_job_id": int(sys.argv[9]),
    "gpu_jobs_submitted": 3,
    "maximum_concurrent_gpu_jobs": 2,
    "gpus_per_job": 1,
    "same_sequence_multiset": True,
    "only_training_order_changed": True,
    "training_sequences_per_arm": 244_140,
    "training_tokens_per_arm": 499_998_720,
    "optimizer_steps_per_arm": 954,
    "seed": 20260821,
    "quota_preflight": {
        "used_kib": int(sys.argv[10]),
        "hard_kib": int(sys.argv[11]),
        "headroom_kib": int(sys.argv[12]),
        "used_files": int(sys.argv[13]),
        "hard_files": int(sys.argv[14]),
        "headroom_files": int(sys.argv[15]),
        "minimum_headroom_kib": int(sys.argv[16]),
        "minimum_headroom_files": int(sys.argv[17]),
    },
    "scientific_promotion_authorized": False,
    "four_b_training_authorized": False,
    "requeue": False,
}
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with os.fdopen(descriptor, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

trap - ERR
printf 'canary=%s curriculum=%s control=%s compare=%s\n' \
  "$canary_job" "$curriculum_job" "$control_job" "$compare_job"
