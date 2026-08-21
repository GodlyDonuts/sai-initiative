#!/bin/bash
#SBATCH --job-name=sai-launch-screen-mc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${FLA_ROOT:?FLA_ROOT is required}"
: "${ENVIRONMENT_RECEIPT:?ENVIRONMENT_RECEIPT is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${GEOMETRY:?GEOMETRY is required}"
: "${FAMILY:?FAMILY is required}"
: "${SHORT_SCREEN_JOB_ID:?SHORT_SCREEN_JOB_ID is required}"
: "${SHORT_SCREEN_RESULT:?SHORT_SCREEN_RESULT is required}"
: "${CHECKPOINT:?CHECKPOINT is required}"
: "${CHECKPOINT_MANIFEST:?CHECKPOINT_MANIFEST is required}"
: "${TRAINING_STREAM:?TRAINING_STREAM is required}"
: "${TOKENIZER_ROOT:?TOKENIZER_ROOT is required}"
: "${POPULATION_JOB_ID:?POPULATION_JOB_ID is required}"
: "${POPULATION_AGGREGATE:?POPULATION_AGGREGATE is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${DISPATCH:?DISPATCH is required}"

EVALUATOR_JOB="$SAI_ROOT/jobs/sai-short-screen-development-mc-single-h100.sbatch"
EVALUATOR_RUNTIME="$EVALUATOR_JOB"

require_absolute() {
  case "$1" in
    /*) ;;
    *) echo "path must be absolute: $1" >&2; exit 2 ;;
  esac
}

for path in \
  "$SAI_ROOT" "$PYTHON" "$FLA_ROOT" "$ENVIRONMENT_RECEIPT" "$LOG_ROOT" \
  "$GEOMETRY" "$SHORT_SCREEN_RESULT" "$CHECKPOINT" "$CHECKPOINT_MANIFEST" \
  "$TRAINING_STREAM" "$TOKENIZER_ROOT" "$POPULATION_AGGREGATE" \
  "$OUTPUT_ROOT" "$DISPATCH" "$EVALUATOR_JOB"; do
  require_absolute "$path"
done

test -n "${SLURM_JOB_ID:-}"
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU evaluation launcher was exposed to a GPU" >&2; exit 2 ;;
esac
test -d "$SAI_ROOT/.git"
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$FLA_ROOT/fla"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test -d "$OUTPUT_ROOT"
test ! -L "$OUTPUT_ROOT"
for path in \
  "$ENVIRONMENT_RECEIPT" "$GEOMETRY" "$SHORT_SCREEN_RESULT" "$CHECKPOINT" \
  "$CHECKPOINT_MANIFEST" "$POPULATION_AGGREGATE" "$EVALUATOR_JOB"; do
  test -f "$path"
  test ! -L "$path"
done
test -d "$TRAINING_STREAM"
test ! -L "$TRAINING_STREAM"
test -d "$TOKENIZER_ROOT"
test ! -L "$TOKENIZER_ROOT"
test "$(dirname "$DISPATCH")" = "$OUTPUT_ROOT"
test ! -e "$DISPATCH"
test ! -L "$DISPATCH"

case "$FAMILY" in
  gated_gqa|gdn_hybrid|kda_mla_hybrid) ;;
  *) echo "family is outside the bounded 100M tournament" >&2; exit 2 ;;
esac

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

short_screen_state="$(sacct -j "$SHORT_SCREEN_JOB_ID" -X -n -P -o State | sed -n '1p')"
population_state="$(sacct -j "$POPULATION_JOB_ID" -X -n -P -o State | sed -n '1p')"
test "$short_screen_state" = "COMPLETED"
test "$population_state" = "COMPLETED"

MMLU_OUTPUT="$OUTPUT_ROOT/${FAMILY}.mmlu_pro.development_mc.json"
MUSR_OUTPUT="$OUTPUT_ROOT/${FAMILY}.musr.development_mc.json"
for path in "$MMLU_OUTPUT" "$MUSR_OUTPUT"; do
  test ! -e "$path"
  test ! -L "$path"
done

export PYTHONPATH="$FLA_ROOT:$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

validated_lines="$("$PYTHON" -P -s -B - \
  "$GEOMETRY" "$FAMILY" "$SHORT_SCREEN_RESULT" "$CHECKPOINT" \
  "$CHECKPOINT_MANIFEST" "$TRAINING_STREAM" "$TOKENIZER_ROOT" \
  "$POPULATION_AGGREGATE" "$ENVIRONMENT_RECEIPT" "$EVALUATOR_RUNTIME" <<'PY'
import hashlib
import json
import pathlib
import sys

from sai.data.token_stream import canonical_sha256, sha256_tree, validate_frozen_stream
from sai.evaluation.short_screen_mc import (
    load_validated_model_state,
    validate_short_screen_result,
)
from sai.model.reference import SaiCausalLM, exact_parameter_count
from sai.training.short_screen import load_bounded_config


def sha256_file(path):
    assert path.is_file() and not path.is_symlink()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def validate_population(entry, benchmark, expected_rows):
    assert entry["benchmark"] == benchmark
    assert entry["rows"] == expected_rows
    identity_order = entry["identity_order_sha256"]
    source = pathlib.Path(entry["source"]["path"])
    disjoint = pathlib.Path(entry["source_disjoint_receipt"]["path"])
    conversion = pathlib.Path(entry["conversion_receipt"]["path"])
    for path in (source, disjoint, conversion):
        assert path.is_absolute() and path.is_file() and not path.is_symlink()
    assert source.stat().st_size == entry["source"]["bytes"]
    assert sha256_file(source) == entry["source"]["sha256"]
    assert disjoint.stat().st_size == entry["source_disjoint_receipt"]["bytes"]
    assert sha256_file(disjoint) == entry["source_disjoint_receipt"]["sha256"]
    assert conversion.stat().st_size == entry["conversion_receipt"]["bytes"]
    assert sha256_file(conversion) == entry["conversion_receipt"]["sha256"]
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows) == expected_rows
    assert all(row["benchmark"] == benchmark for row in rows)
    assert canonical_sha256([row["row_id"] for row in rows]) == identity_order
    disjoint_receipt = load_json(disjoint)
    assert disjoint_receipt["benchmark"] == benchmark
    assert disjoint_receipt["benchmark_source_sha256"] == entry["source"]["sha256"]
    assert disjoint_receipt["source_disjoint"] is True
    assert disjoint_receipt["method"] == "identity-and-contamination-audit"
    conversion_receipt = load_json(conversion)
    unsigned = dict(conversion_receipt)
    claimed = unsigned.pop("receipt_sha256")
    assert claimed == entry["conversion_receipt"]["receipt_sha256"]
    assert claimed == canonical_sha256(unsigned)
    assert conversion_receipt["status"] == "complete"
    assert conversion_receipt["benchmark"] == benchmark
    assert conversion_receipt["evidence"]["paired_rows"] == expected_rows
    assert conversion_receipt["evidence"]["identity_order_sha256"] == identity_order
    return {
        "source": str(source),
        "source_sha256": entry["source"]["sha256"],
        "disjoint": str(disjoint),
        "disjoint_sha256": entry["source_disjoint_receipt"]["sha256"],
        "training_source_sha256": disjoint_receipt["training_source_sha256"],
        "rows": expected_rows,
        "identity_order_sha256": identity_order,
        "decontamination_receipt_sha256": conversion_receipt["evidence"][
            "training_decontamination_receipt"
        ]["file_sha256"],
    }


geometry = pathlib.Path(sys.argv[1])
family = sys.argv[2]
result_path = pathlib.Path(sys.argv[3])
checkpoint = pathlib.Path(sys.argv[4])
manifest = pathlib.Path(sys.argv[5])
training_stream_path = pathlib.Path(sys.argv[6])
tokenizer_root = pathlib.Path(sys.argv[7])
aggregate_path = pathlib.Path(sys.argv[8])
environment_receipt_path = pathlib.Path(sys.argv[9])
evaluator_runtime_path = pathlib.Path(sys.argv[10])

config, geometry_row = load_bounded_config(geometry, family)
assert geometry_row["scale"] == "100m"
assert geometry_row["parameter_ledger"]["total"] <= 101_000_000
geometry_sha256 = sha256_file(geometry)
result_sha256 = sha256_file(result_path)
result, bindings = validate_short_screen_result(
    result_path,
    expected_sha256=result_sha256,
    config=config,
    family=family,
    geometry_parameter_count=geometry_row["parameter_ledger"]["total"],
)
training_stream = validate_frozen_stream(training_stream_path, verify_sources=True)
tokenizer_sha256 = sha256_tree(tokenizer_root)
assert training_stream["ordered_stream_identity_sha256"] == result[
    "training_stream_identity_sha256"
]
assert training_stream["tokenizer_identity_sha256"] == tokenizer_sha256
assert training_stream["vocab_size"] == config.vocab_size
model = SaiCausalLM(config, delta_backend=result["delta_backend"])
checkpoint_observation = load_validated_model_state(
    checkpoint,
    manifest,
    model=model,
    expected_bindings=bindings,
    expected_descriptor=result["checkpoint"],
    expected_counters=result["counters"],
    expected_cursor=result["stream_cursor"],
    expected_final_state_sha256=result["final_state_sha256"],
)
assert exact_parameter_count(model) == geometry_row["parameter_ledger"]["total"]

aggregate = load_json(aggregate_path)
unsigned = dict(aggregate)
claimed = unsigned.pop("receipt_sha256")
assert aggregate["schema"] == "sai-development-mc-populations-aggregate-v1"
assert aggregate["status"] == "complete"
assert aggregate["development_only"] is True
assert aggregate["official_benchmark_result"] is False
assert aggregate["architecture_promotion_allowed"] is False
assert aggregate["total_rows"] == 12788
assert claimed == canonical_sha256(unsigned)
assert len(aggregate["populations"]) == 2
mmlu = validate_population(aggregate["populations"][0], "mmlu_pro", 12032)
musr = validate_population(aggregate["populations"][1], "musr", 756)
decontamination = aggregate["training_decontamination_receipt"]
decontamination_path = pathlib.Path(decontamination["path"])
assert decontamination_path.is_file() and not decontamination_path.is_symlink()
assert decontamination_path.stat().st_size == decontamination["bytes"]
assert sha256_file(decontamination_path) == decontamination["sha256"]
assert mmlu["decontamination_receipt_sha256"] == decontamination["sha256"]
assert musr["decontamination_receipt_sha256"] == decontamination["sha256"]
assert mmlu["training_source_sha256"] == musr["training_source_sha256"]

environment_receipt = load_json(environment_receipt_path)
assert environment_receipt["schema"] == "sai-fla-parity-receipt-v1"
assert environment_receipt["status"] == "cuda_fla_parity_qualified"
assert environment_receipt["parity_qualified"] is True
assert environment_receipt["production_cuda_qualified"] is True
assert environment_receipt["operators"]["version"] == "0.4.2"
assert environment_receipt["operators"]["mock"] is False
assert evaluator_runtime_path.is_file() and not evaluator_runtime_path.is_symlink()

values = [
    geometry_sha256,
    result_sha256,
    training_stream["ordered_stream_identity_sha256"],
    tokenizer_sha256,
    sha256_file(aggregate_path),
    checkpoint_observation["checkpoint_sha256"],
    checkpoint_observation["manifest_sha256"],
    sha256_file(environment_receipt_path),
    sha256_file(evaluator_runtime_path),
]
for population in (mmlu, musr):
    values.extend(
        [
            population["source"],
            population["source_sha256"],
            population["disjoint"],
            population["disjoint_sha256"],
            population["training_source_sha256"],
            str(population["rows"]),
            population["identity_order_sha256"],
        ]
    )
assert all("\n" not in value for value in values)
print("\n".join(values))
PY
)"

mapfile -t validated <<< "$validated_lines"
test "${#validated[@]}" = 23
GEOMETRY_SHA256="${validated[0]}"
SHORT_SCREEN_RESULT_SHA256="${validated[1]}"
TRAINING_STREAM_IDENTITY="${validated[2]}"
TOKENIZER_SHA256="${validated[3]}"
POPULATION_AGGREGATE_SHA256="${validated[4]}"
CHECKPOINT_SHA256="${validated[5]}"
CHECKPOINT_MANIFEST_SHA256="${validated[6]}"
ENVIRONMENT_RECEIPT_SHA256="${validated[7]}"
EVALUATOR_RUNTIME_SHA256="${validated[8]}"
MMLU_SOURCE="${validated[9]}"
MMLU_SOURCE_SHA256="${validated[10]}"
MMLU_DISJOINT="${validated[11]}"
MMLU_DISJOINT_SHA256="${validated[12]}"
MMLU_TRAINING_SOURCE_SHA256="${validated[13]}"
MMLU_EXPECTED_ROWS="${validated[14]}"
MMLU_IDENTITY_ORDER_SHA256="${validated[15]}"
MUSR_SOURCE="${validated[16]}"
MUSR_SOURCE_SHA256="${validated[17]}"
MUSR_DISJOINT="${validated[18]}"
MUSR_DISJOINT_SHA256="${validated[19]}"
MUSR_TRAINING_SOURCE_SHA256="${validated[20]}"
MUSR_EXPECTED_ROWS="${validated[21]}"
MUSR_IDENTITY_ORDER_SHA256="${validated[22]}"

submitted=()
cancel_partial_fanout() {
  for job_id in "${submitted[@]}"; do
    scancel "$job_id" || true
  done
}
trap cancel_partial_fanout EXIT

common_export="SAI_ROOT=$SAI_ROOT,EXPECTED_COMMIT=$EXPECTED_COMMIT,PYTHON=$PYTHON,FLA_ROOT=$FLA_ROOT,ENVIRONMENT_RECEIPT=$ENVIRONMENT_RECEIPT,LOG_ROOT=$LOG_ROOT,GEOMETRY=$GEOMETRY,GEOMETRY_SHA256=$GEOMETRY_SHA256,FAMILY=$FAMILY,SHORT_SCREEN_RESULT=$SHORT_SCREEN_RESULT,SHORT_SCREEN_RESULT_SHA256=$SHORT_SCREEN_RESULT_SHA256,CHECKPOINT=$CHECKPOINT,CHECKPOINT_MANIFEST=$CHECKPOINT_MANIFEST,TRAINING_STREAM=$TRAINING_STREAM,TRAINING_STREAM_IDENTITY=$TRAINING_STREAM_IDENTITY,TOKENIZER_ROOT=$TOKENIZER_ROOT,TOKENIZER_SHA256=$TOKENIZER_SHA256,RUNTIME_PATH=$EVALUATOR_RUNTIME"

mmlu_job_id="$(env -i PATH=/apps/slurm/current/bin:/usr/bin:/bin sbatch --parsable \
  --chdir="$SAI_ROOT" \
  --job-name="sai-${FAMILY}-mmlu" \
  --output="$LOG_ROOT/sai_${FAMILY}_mmlu_%j.out" \
  --error="$LOG_ROOT/sai_${FAMILY}_mmlu_%j.err" \
  --export="$common_export,BENCHMARK=mmlu_pro,BENCHMARK_SOURCE=$MMLU_SOURCE,BENCHMARK_SOURCE_SHA256=$MMLU_SOURCE_SHA256,DISJOINT_RECEIPT=$MMLU_DISJOINT,DISJOINT_RECEIPT_SHA256=$MMLU_DISJOINT_SHA256,TRAINING_SOURCE_SHA256=$MMLU_TRAINING_SOURCE_SHA256,EXPECTED_ROWS=$MMLU_EXPECTED_ROWS,EXPECTED_IDENTITY_ORDER_SHA256=$MMLU_IDENTITY_ORDER_SHA256,MAX_SEQUENCE_TOKENS=2048,OUTPUT=$MMLU_OUTPUT" \
  "$EVALUATOR_JOB")"
[[ "$mmlu_job_id" =~ ^[0-9]+$ ]]
submitted+=("$mmlu_job_id")

musr_job_id="$(env -i PATH=/apps/slurm/current/bin:/usr/bin:/bin sbatch --parsable \
  --chdir="$SAI_ROOT" \
  --job-name="sai-${FAMILY}-musr" \
  --output="$LOG_ROOT/sai_${FAMILY}_musr_%j.out" \
  --error="$LOG_ROOT/sai_${FAMILY}_musr_%j.err" \
  --export="$common_export,BENCHMARK=musr,BENCHMARK_SOURCE=$MUSR_SOURCE,BENCHMARK_SOURCE_SHA256=$MUSR_SOURCE_SHA256,DISJOINT_RECEIPT=$MUSR_DISJOINT,DISJOINT_RECEIPT_SHA256=$MUSR_DISJOINT_SHA256,TRAINING_SOURCE_SHA256=$MUSR_TRAINING_SOURCE_SHA256,EXPECTED_ROWS=$MUSR_EXPECTED_ROWS,EXPECTED_IDENTITY_ORDER_SHA256=$MUSR_IDENTITY_ORDER_SHA256,MAX_SEQUENCE_TOKENS=2048,OUTPUT=$MUSR_OUTPUT" \
  "$EVALUATOR_JOB")"
[[ "$musr_job_id" =~ ^[0-9]+$ ]]
submitted+=("$musr_job_id")

"$PYTHON" -P -s -B - \
  "$DISPATCH" "$EXPECTED_COMMIT" "$FAMILY" \
  "$SHORT_SCREEN_JOB_ID" "$SHORT_SCREEN_RESULT" "$SHORT_SCREEN_RESULT_SHA256" \
  "$CHECKPOINT" "$CHECKPOINT_SHA256" "$CHECKPOINT_MANIFEST" \
  "$CHECKPOINT_MANIFEST_SHA256" "$GEOMETRY" "$GEOMETRY_SHA256" \
  "$TRAINING_STREAM" "$TRAINING_STREAM_IDENTITY" \
  "$TOKENIZER_ROOT" "$TOKENIZER_SHA256" \
  "$FLA_ROOT" "$ENVIRONMENT_RECEIPT" "$ENVIRONMENT_RECEIPT_SHA256" \
  "$EVALUATOR_RUNTIME" "$EVALUATOR_RUNTIME_SHA256" \
  "$POPULATION_JOB_ID" "$POPULATION_AGGREGATE" \
  "$POPULATION_AGGREGATE_SHA256" \
  "$mmlu_job_id" "$MMLU_OUTPUT" "$MMLU_SOURCE" "$MMLU_SOURCE_SHA256" \
  "$MMLU_DISJOINT" "$MMLU_DISJOINT_SHA256" \
  "$MMLU_TRAINING_SOURCE_SHA256" "$MMLU_EXPECTED_ROWS" \
  "$MMLU_IDENTITY_ORDER_SHA256" \
  "$musr_job_id" "$MUSR_OUTPUT" "$MUSR_SOURCE" "$MUSR_SOURCE_SHA256" \
  "$MUSR_DISJOINT" "$MUSR_DISJOINT_SHA256" \
  "$MUSR_TRAINING_SOURCE_SHA256" "$MUSR_EXPECTED_ROWS" \
  "$MUSR_IDENTITY_ORDER_SHA256" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import uuid


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


values = sys.argv[1:]
dispatch = pathlib.Path(values[0])
payload = {
    "schema": "sai-short-screen-development-mc-dispatch-v1",
    "status": "submitted",
    "four_b_training_authorized": False,
    "development_only": True,
    "code_commit": values[1],
    "family": values[2],
    "short_screen": {
        "job_id": values[3],
        "result": {"path": values[4], "sha256": values[5]},
        "checkpoint": {"path": values[6], "sha256": values[7]},
        "checkpoint_manifest": {"path": values[8], "sha256": values[9]},
    },
    "geometry": {"path": values[10], "sha256": values[11], "scale": "100m"},
    "training_stream": {"path": values[12], "identity_sha256": values[13]},
    "tokenizer": {"path": values[14], "tree_sha256": values[15]},
    "fla_root": {"path": values[16], "version": "0.4.2"},
    "environment_receipt": {"path": values[17], "sha256": values[18]},
    "evaluator_runtime": {"path": values[19], "sha256": values[20]},
    "populations": {
        "job_id": values[21],
        "aggregate": {"path": values[22], "sha256": values[23]},
    },
    "jobs": [
        {
            "benchmark": "mmlu_pro",
            "job_id": values[24],
            "output": values[25],
            "source": {"path": values[26], "sha256": values[27]},
            "disjoint_receipt": {"path": values[28], "sha256": values[29]},
            "training_source_sha256": values[30],
            "expected_rows": int(values[31]),
            "identity_order_sha256": values[32],
            "h100s": 1,
        },
        {
            "benchmark": "musr",
            "job_id": values[33],
            "output": values[34],
            "source": {"path": values[35], "sha256": values[36]},
            "disjoint_receipt": {"path": values[37], "sha256": values[38]},
            "training_source_sha256": values[39],
            "expected_rows": int(values[40]),
            "identity_order_sha256": values[41],
            "h100s": 1,
        },
    ],
    "gpu_jobs_submitted": 2,
    "maximum_concurrent_single_h100_jobs": 2,
    "retry": False,
    "requeue": False,
}
payload["receipt_sha256"] = canonical_sha256(payload)
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
temporary = dispatch.parent / f".{dispatch.name}.{uuid.uuid4().hex}.tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, dispatch)
    temporary.unlink()
    directory = os.open(dispatch.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
PY

trap - EXIT
printf '%s\n' "$mmlu_job_id" "$musr_job_id"
