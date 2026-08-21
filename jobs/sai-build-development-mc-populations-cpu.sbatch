#!/bin/bash
#SBATCH --job-name=sai-build-dev-mc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${DECONTAMINATION_JOB_ID:?DECONTAMINATION_JOB_ID is required}"
: "${MMLU_QUESTIONS_SHA256:?MMLU_QUESTIONS_SHA256 is required}"
: "${MMLU_ASSESSORS_SHA256:?MMLU_ASSESSORS_SHA256 is required}"
: "${MMLU_IDENTITY_ORDER_SHA256:?MMLU_IDENTITY_ORDER_SHA256 is required}"
: "${MUSR_QUESTIONS_SHA256:?MUSR_QUESTIONS_SHA256 is required}"
: "${MUSR_ASSESSORS_SHA256:?MUSR_ASSESSORS_SHA256 is required}"
: "${MUSR_IDENTITY_ORDER_SHA256:?MUSR_IDENTITY_ORDER_SHA256 is required}"

DECONTAMINATION_RECEIPT=/lustre/fs1/home/sa305415/sai-initiative/artifacts/fineweb_edu_mechanics_admitted_be505b6_r1.receipt.json
MMLU_QUESTIONS=/lustre/fs1/home/sa305415/shohin/artifacts/public_bench_qwen9_766196_r1/data/mmlu_pro/full.questions.jsonl
MMLU_ASSESSORS=/lustre/fs1/home/sa305415/shohin/artifacts/public_bench_qwen9_766196_r1/data/mmlu_pro/full.assessors.jsonl
MUSR_QUESTIONS=/lustre/fs1/home/sa305415/shohin/artifacts/public_bench_qwen9_766196_r1/data/musr/full.questions.jsonl
MUSR_ASSESSORS=/lustre/fs1/home/sa305415/shohin/artifacts/public_bench_qwen9_766196_r1/data/musr/full.assessors.jsonl
OUTPUT_ROOT=/lustre/fs1/home/sa305415/sai-initiative/artifacts/development_mc_populations_753348f_r1

require_absolute() {
  case "$1" in
    /*) ;;
    *) echo "path must be absolute: $1" >&2; exit 2 ;;
  esac
}

for path in \
  "$SAI_ROOT" "$PYTHON" "$LOG_ROOT" "$DECONTAMINATION_RECEIPT" \
  "$MMLU_QUESTIONS" "$MMLU_ASSESSORS" "$MUSR_QUESTIONS" \
  "$MUSR_ASSESSORS" "$OUTPUT_ROOT"; do
  require_absolute "$path"
done

test -n "${SLURM_JOB_ID:-}"
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU population job was exposed to a GPU" >&2; exit 2 ;;
esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$LOG_ROOT"
test ! -L "$LOG_ROOT"
test -d "$OUTPUT_ROOT"
test ! -L "$OUTPUT_ROOT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

decontamination_state="$(sacct -j "$DECONTAMINATION_JOB_ID" -X -n -P -o State | sed -n '1p')"
test "$decontamination_state" = "COMPLETED"

for path in \
  "$DECONTAMINATION_RECEIPT" "$MMLU_QUESTIONS" "$MMLU_ASSESSORS" \
  "$MUSR_QUESTIONS" "$MUSR_ASSESSORS"; do
  test -f "$path"
  test ! -L "$path"
done

test "$(sha256sum "$MMLU_QUESTIONS" | cut -d ' ' -f 1)" = "$MMLU_QUESTIONS_SHA256"
test "$(sha256sum "$MMLU_ASSESSORS" | cut -d ' ' -f 1)" = "$MMLU_ASSESSORS_SHA256"
test "$(sha256sum "$MUSR_QUESTIONS" | cut -d ' ' -f 1)" = "$MUSR_QUESTIONS_SHA256"
test "$(sha256sum "$MUSR_ASSESSORS" | cut -d ' ' -f 1)" = "$MUSR_ASSESSORS_SHA256"
decontamination_receipt_sha256="$(sha256sum "$DECONTAMINATION_RECEIPT" | cut -d ' ' -f 1)"
test "${#decontamination_receipt_sha256}" = 64

MMLU_SOURCE="$OUTPUT_ROOT/mmlu_pro.jsonl"
MMLU_DISJOINT="$OUTPUT_ROOT/mmlu_pro.source_disjoint.receipt.json"
MMLU_CONVERSION="$OUTPUT_ROOT/mmlu_pro.conversion.receipt.json"
MUSR_SOURCE="$OUTPUT_ROOT/musr.jsonl"
MUSR_DISJOINT="$OUTPUT_ROOT/musr.source_disjoint.receipt.json"
MUSR_CONVERSION="$OUTPUT_ROOT/musr.conversion.receipt.json"
AGGREGATE_RECEIPT="$OUTPUT_ROOT/populations.aggregate.receipt.json"
outputs=(
  "$MMLU_SOURCE" "$MMLU_DISJOINT" "$MMLU_CONVERSION"
  "$MUSR_SOURCE" "$MUSR_DISJOINT" "$MUSR_CONVERSION"
  "$AGGREGATE_RECEIPT"
)
for path in "${outputs[@]}"; do
  test ! -e "$path"
  test ! -L "$path"
done

created_outputs=()
cleanup_partial_publication() {
  for path in "${created_outputs[@]}"; do
    rm -f -- "$path"
  done
}
trap cleanup_partial_publication EXIT

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

"$PYTHON" -P -s -B -m sai.evaluation.population_builder \
  --benchmark mmlu_pro \
  --questions "$MMLU_QUESTIONS" \
  --assessors "$MMLU_ASSESSORS" \
  --expected-questions-sha256 "$MMLU_QUESTIONS_SHA256" \
  --expected-assessors-sha256 "$MMLU_ASSESSORS_SHA256" \
  --expected-rows 12032 \
  --expected-identity-order-sha256 "$MMLU_IDENTITY_ORDER_SHA256" \
  --training-decontamination-receipt "$DECONTAMINATION_RECEIPT" \
  --expected-training-decontamination-receipt-sha256 "$decontamination_receipt_sha256" \
  --output-source "$MMLU_SOURCE" \
  --output-disjoint-receipt "$MMLU_DISJOINT" \
  --output-conversion-receipt "$MMLU_CONVERSION"
created_outputs+=("$MMLU_SOURCE" "$MMLU_DISJOINT" "$MMLU_CONVERSION")

"$PYTHON" -P -s -B -m sai.evaluation.population_builder \
  --benchmark musr \
  --questions "$MUSR_QUESTIONS" \
  --assessors "$MUSR_ASSESSORS" \
  --expected-questions-sha256 "$MUSR_QUESTIONS_SHA256" \
  --expected-assessors-sha256 "$MUSR_ASSESSORS_SHA256" \
  --expected-rows 756 \
  --expected-identity-order-sha256 "$MUSR_IDENTITY_ORDER_SHA256" \
  --training-decontamination-receipt "$DECONTAMINATION_RECEIPT" \
  --expected-training-decontamination-receipt-sha256 "$decontamination_receipt_sha256" \
  --output-source "$MUSR_SOURCE" \
  --output-disjoint-receipt "$MUSR_DISJOINT" \
  --output-conversion-receipt "$MUSR_CONVERSION"
created_outputs+=("$MUSR_SOURCE" "$MUSR_DISJOINT" "$MUSR_CONVERSION")

"$PYTHON" -P -s -B - \
  "$AGGREGATE_RECEIPT" "$EXPECTED_COMMIT" \
  "$DECONTAMINATION_RECEIPT" "$decontamination_receipt_sha256" \
  "$MMLU_SOURCE" "$MMLU_DISJOINT" "$MMLU_CONVERSION" \
  "$MMLU_QUESTIONS_SHA256" "$MMLU_ASSESSORS_SHA256" \
  "$MMLU_IDENTITY_ORDER_SHA256" \
  "$MUSR_SOURCE" "$MUSR_DISJOINT" "$MUSR_CONVERSION" \
  "$MUSR_QUESTIONS_SHA256" "$MUSR_ASSESSORS_SHA256" \
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


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_population(
    benchmark,
    expected_rows,
    source,
    disjoint,
    conversion,
    questions_sha256,
    assessors_sha256,
    identity_order_sha256,
):
    receipt = json.loads(conversion.read_text())
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256")
    assert receipt["schema"] == "sai-development-mc-population-conversion-v1"
    assert receipt["status"] == "complete"
    assert receipt["benchmark"] == benchmark
    assert receipt["development_only"] is True
    assert receipt["official_benchmark_result"] is False
    assert receipt["public_terminal_result"] is False
    assert receipt["architecture_promotion_allowed"] is False
    assert claimed == canonical_sha256(unsigned)
    assert receipt["evidence"]["questions"]["sha256"] == questions_sha256
    assert receipt["evidence"]["assessors"]["sha256"] == assessors_sha256
    assert receipt["evidence"]["paired_rows"] == expected_rows
    assert receipt["evidence"]["identity_order_sha256"] == identity_order_sha256
    output = receipt["output"]
    assert output["source"]["path"] == str(source.resolve())
    assert output["source"]["rows"] == expected_rows
    assert output["source"]["identity_order_sha256"] == identity_order_sha256
    assert output["source"]["bytes"] == source.stat().st_size
    assert output["source"]["sha256"] == sha256_file(source)
    assert output["source_disjoint_receipt"]["path"] == str(disjoint.resolve())
    assert output["source_disjoint_receipt"]["bytes"] == disjoint.stat().st_size
    assert output["source_disjoint_receipt"]["sha256"] == sha256_file(disjoint)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows) == expected_rows
    assert canonical_sha256([row["row_id"] for row in rows]) == identity_order_sha256
    return {
        "benchmark": benchmark,
        "rows": expected_rows,
        "identity_order_sha256": identity_order_sha256,
        "source": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "source_disjoint_receipt": {
            "path": str(disjoint.resolve()),
            "bytes": disjoint.stat().st_size,
            "sha256": sha256_file(disjoint),
        },
        "conversion_receipt": {
            "path": str(conversion.resolve()),
            "bytes": conversion.stat().st_size,
            "sha256": sha256_file(conversion),
            "receipt_sha256": claimed,
        },
    }


arguments = sys.argv[1:]
aggregate_path = pathlib.Path(arguments[0])
commit = arguments[1]
decontamination_path = pathlib.Path(arguments[2])
decontamination_sha256 = arguments[3]
mmlu = validate_population(
    "mmlu_pro",
    12032,
    *map(pathlib.Path, arguments[4:7]),
    *arguments[7:10],
)
musr = validate_population(
    "musr",
    756,
    *map(pathlib.Path, arguments[10:13]),
    *arguments[13:16],
)
payload = {
    "schema": "sai-development-mc-populations-aggregate-v1",
    "status": "complete",
    "development_only": True,
    "official_benchmark_result": False,
    "architecture_promotion_allowed": False,
    "code_commit": commit,
    "training_decontamination_receipt": {
        "path": str(decontamination_path.resolve()),
        "bytes": decontamination_path.stat().st_size,
        "sha256": decontamination_sha256,
    },
    "populations": [mmlu, musr],
    "total_rows": 12788,
}
payload["receipt_sha256"] = canonical_sha256(payload)
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
temporary = aggregate_path.parent / f".{aggregate_path.name}.{uuid.uuid4().hex}.tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, aggregate_path)
    temporary.unlink()
    directory = os.open(aggregate_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
print(json.dumps({"receipt_sha256": payload["receipt_sha256"], "status": "complete"}))
PY

created_outputs+=("$AGGREGATE_RECEIPT")
trap - EXIT
