#!/usr/bin/env bash
#SBATCH --job-name=sai-launch-smol3-workspace-mc
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
: "${MECHANICS_RECEIPT:?MECHANICS_RECEIPT is required}"
: "${TRAINING_JOB_ID:?TRAINING_JOB_ID is required}"
: "${TRAINING_RESULT:?TRAINING_RESULT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${POPULATION_JOB_ID:?POPULATION_JOB_ID is required}"
: "${POPULATION_ROOT:?POPULATION_ROOT is required}"
: "${STATE_MODE:?STATE_MODE is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test "$(sacct -j "$TRAINING_JOB_ID" -X -n -P -o State | sed -n '1p')" = COMPLETED
test "$(sacct -j "$POPULATION_JOB_ID" -X -n -P -o State | sed -n '1p')" = COMPLETED
test -d "$MODEL_ROOT"
test -f "$MODEL_MANIFEST"
test ! -L "$MODEL_MANIFEST"
test -f "$RESTORATION_RECEIPT"
test ! -L "$RESTORATION_RECEIPT"
test -f "$MECHANICS_RECEIPT"
test -f "$TRAINING_RESULT"
test -f "$CHECKPOINT"
test -f "$CHECKPOINT.manifest.json"
test -d "$POPULATION_ROOT"
test -d "$LOG_ROOT"
test ! -e "$RESULT_ROOT"
case "$STATE_MODE" in recurrent|reset_average) ;; *) exit 2 ;; esac
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
mechanics_sha="$(sha256sum "$MECHANICS_RECEIPT" | cut -d ' ' -f 1)"
training_result_sha="$(sha256sum "$TRAINING_RESULT" | cut -d ' ' -f 1)"
read -r observed_mode training_source_sha < <(
  "$PYTHON" -P -s -B - <<'PY' \
    "$TRAINING_RESULT" "$training_result_sha" \
    "$POPULATION_ROOT/mmlu_pro.source_disjoint.receipt.json"
import json
import pathlib
import sys

from sai.evaluation.hf_smol_workspace_mc import _load_result

result = _load_result(pathlib.Path(sys.argv[1]), sys.argv[2])
population = json.load(open(sys.argv[3]))
print(result["state_mode"], population["training_source_sha256"])
PY
)
test "$observed_mode" = "$STATE_MODE"

mkdir -m 700 "$RESULT_ROOT"
mmlu_shards="$RESULT_ROOT/mmlu_pro_shards"
"$PYTHON" -P -s -B -m sai.evaluation.development_mc_shards build \
  --benchmark mmlu_pro \
  --source "$POPULATION_ROOT/mmlu_pro.jsonl" \
  --full-disjoint-receipt "$POPULATION_ROOT/mmlu_pro.source_disjoint.receipt.json" \
  --training-source-sha256 "$training_source_sha" \
  --shard-count 8 \
  --output-root "$mmlu_shards"

admitted_jobs=()
mmlu_jobs=()
mmlu_results=()
cleanup_partial_submission() {
  if ((${#admitted_jobs[@]})); then scancel "${admitted_jobs[@]}" || true; fi
}
trap cleanup_partial_submission ERR

while IFS=$'\t' read -r index source disjoint rows identity; do
  result="$RESULT_ROOT/mmlu_pro_shard_${index}.json"
  job_id="$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
    --job-name="sai-smol3-${STATE_MODE}-mmlu-${index}" \
    --output="$LOG_ROOT/sai_smol3_${STATE_MODE}_mmlu_${index}_%j.out" \
    --error="$LOG_ROOT/sai_smol3_${STATE_MODE}_mmlu_${index}_%j.err" \
    --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FAST_KERNELS_ROOT="$FAST_KERNELS_ROOT",MODEL_ROOT="$MODEL_ROOT",MODEL_MANIFEST="$MODEL_MANIFEST",RESTORATION_RECEIPT="$RESTORATION_RECEIPT",MECHANICS_RECEIPT="$MECHANICS_RECEIPT",MECHANICS_RECEIPT_SHA256="$mechanics_sha",TRAINING_RESULT="$TRAINING_RESULT",TRAINING_RESULT_SHA256="$training_result_sha",CHECKPOINT="$CHECKPOINT",BENCHMARK=mmlu_pro,BENCHMARK_SOURCE="$source",DISJOINT_RECEIPT="$disjoint",TRAINING_SOURCE_SHA256="$training_source_sha",EXPECTED_ROWS="$rows",EXPECTED_IDENTITY_ORDER_SHA256="$identity",OUTPUT="$result" \
    "$SAI_ROOT/jobs/sai-smollm3-3b-workspace-development-mc-single-h100.sbatch")"
  admitted_jobs+=("$job_id")
  mmlu_jobs+=("$job_id")
  mmlu_results+=("$result")
done < <("$PYTHON" -P -s -B - <<'PY' "$mmlu_shards/manifest.json"
import json
import sys

for row in json.load(open(sys.argv[1]))["shards"]:
    print(
        row["index"], row["source"], row["disjoint_receipt"], row["rows"],
        row["identity_order_sha256"], sep="\t"
    )
PY
)
test "${#mmlu_jobs[@]}" = 8

read -r musr_rows musr_identity < <("$PYTHON" -P -s -B - <<'PY' \
  "$POPULATION_ROOT/populations.aggregate.receipt.json"
import json
import sys

for row in json.load(open(sys.argv[1]))["populations"]:
    if row["benchmark"] == "musr":
        print(row["rows"], row["identity_order_sha256"])
PY
)
musr_result="$RESULT_ROOT/musr.json"
musr_job="$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --job-name="sai-smol3-${STATE_MODE}-musr" \
  --output="$LOG_ROOT/sai_smol3_${STATE_MODE}_musr_%j.out" \
  --error="$LOG_ROOT/sai_smol3_${STATE_MODE}_musr_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",FAST_KERNELS_ROOT="$FAST_KERNELS_ROOT",MODEL_ROOT="$MODEL_ROOT",MODEL_MANIFEST="$MODEL_MANIFEST",RESTORATION_RECEIPT="$RESTORATION_RECEIPT",MECHANICS_RECEIPT="$MECHANICS_RECEIPT",MECHANICS_RECEIPT_SHA256="$mechanics_sha",TRAINING_RESULT="$TRAINING_RESULT",TRAINING_RESULT_SHA256="$training_result_sha",CHECKPOINT="$CHECKPOINT",BENCHMARK=musr,BENCHMARK_SOURCE="$POPULATION_ROOT/musr.jsonl",DISJOINT_RECEIPT="$POPULATION_ROOT/musr.source_disjoint.receipt.json",TRAINING_SOURCE_SHA256="$training_source_sha",EXPECTED_ROWS="$musr_rows",EXPECTED_IDENTITY_ORDER_SHA256="$musr_identity",OUTPUT="$musr_result" \
  "$SAI_ROOT/jobs/sai-smollm3-3b-workspace-development-mc-single-h100.sbatch")"
admitted_jobs+=("$musr_job")

mmlu_dependency="$(IFS=:; echo "${mmlu_jobs[*]}")"
mmlu_result_paths="$(IFS=:; echo "${mmlu_results[*]}")"
merge_job="$(env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch --parsable \
  --dependency="afterok:$mmlu_dependency" --kill-on-invalid-dep=yes \
  --job-name="sai-smol3-${STATE_MODE}-mmlu-merge" \
  --output="$LOG_ROOT/sai_smol3_${STATE_MODE}_mmlu_merge_%j.out" \
  --error="$LOG_ROOT/sai_smol3_${STATE_MODE}_mmlu_merge_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",MANIFEST="$mmlu_shards/manifest.json",RESULT_PATHS="$mmlu_result_paths",EXPECTED_RESULT_COUNT=8,OUTPUT="$RESULT_ROOT/mmlu_pro.json" \
  "$SAI_ROOT/jobs/sai-merge-development-mc-shards-cpu.sbatch")"
admitted_jobs+=("$merge_job")

"$PYTHON" -P -s -B - <<'PY' \
  "$RESULT_ROOT/dispatch.json" "$EXPECTED_COMMIT" "$STATE_MODE" \
  "$TRAINING_JOB_ID" "$POPULATION_JOB_ID" "$training_result_sha" \
  "$training_source_sha" "$musr_job" "$merge_job" "${mmlu_jobs[@]}"
import hashlib
import json
import os
import pathlib
import sys
import uuid

output = pathlib.Path(sys.argv[1])
payload = {
    "schema": "sai-smollm3-3b-workspace-development-mc-dispatch-v1",
    "status": "submitted",
    "source_commit": sys.argv[2],
    "state_mode": sys.argv[3],
    "training_job": int(sys.argv[4]),
    "population_job": int(sys.argv[5]),
    "training_result_file_sha256": sys.argv[6],
    "training_source_sha256": sys.argv[7],
    "musr_job": int(sys.argv[8]),
    "mmlu_merge_job": int(sys.argv[9]),
    "mmlu_shard_jobs": [int(value) for value in sys.argv[10:]],
    "h100_jobs": 9,
    "maximum_concurrent_h100_jobs": 9,
    "one_h100_per_job": True,
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
printf 'state_mode=%s mmlu=%s musr=%s merge=%s\n' \
  "$STATE_MODE" "$mmlu_dependency" "$musr_job" "$merge_job"
