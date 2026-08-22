#!/bin/bash
#SBATCH --job-name=sai-stage-screen-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${LOG_ROOT:?LOG_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${GATED_GQA_LAUNCH_JOB_ID:?GATED_GQA_LAUNCH_JOB_ID is required}"
: "${GDN_HYBRID_LAUNCH_JOB_ID:?GDN_HYBRID_LAUNCH_JOB_ID is required}"
: "${KDA_MLA_HYBRID_LAUNCH_JOB_ID:?KDA_MLA_HYBRID_LAUNCH_JOB_ID is required}"

COMPARISON_JOB="$SAI_ROOT/jobs/sai-short-screen-development-mc-comparison-cpu.sbatch"
DISPATCH="$OUTPUT_ROOT/comparison.dispatch.json"
OUTPUT="$OUTPUT_ROOT/comparison.json"

test -n "${SLURM_JOB_ID:-}"
case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU comparison stager was exposed to a GPU" >&2; exit 2 ;;
esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$LOG_ROOT"
test -d "$OUTPUT_ROOT"
test ! -L "$OUTPUT_ROOT"
test -f "$COMPARISON_JOB"
test ! -L "$COMPARISON_JOB"
test ! -e "$DISPATCH"
test ! -e "$OUTPUT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

validated_lines="$("$PYTHON" -P -s -B - \
  "$OUTPUT_ROOT" "$GATED_GQA_LAUNCH_JOB_ID" "$GDN_HYBRID_LAUNCH_JOB_ID" \
  "$KDA_MLA_HYBRID_LAUNCH_JOB_ID" <<'PY'
import hashlib
import json
import pathlib
import sys


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


root = pathlib.Path(sys.argv[1])
families = ["gated_gqa", "gdn_hybrid", "kda_mla_hybrid"]
launcher_ids = [int(value) for value in sys.argv[2:]]
for family, launcher_id in zip(families, launcher_ids, strict=True):
    dispatch_path = root / family / "dispatch.json"
    assert dispatch_path.is_file() and not dispatch_path.is_symlink()
    payload = json.loads(dispatch_path.read_text())
    unsigned = dict(payload)
    claimed = unsigned.pop("receipt_sha256")
    assert payload["schema"] == "sai-short-screen-development-mc-dispatch-v1"
    assert payload["status"] == "submitted"
    assert payload["family"] == family
    assert payload["development_only"] is True
    assert payload["four_b_training_authorized"] is False
    assert payload["gpu_jobs_submitted"] == 2
    assert payload["maximum_concurrent_single_h100_jobs"] == 2
    assert payload["retry"] is False and payload["requeue"] is False
    assert payload["populations"]["job_id"] > 0
    assert claimed == canonical_sha256(unsigned)
    jobs = payload["jobs"]
    assert [job["benchmark"] for job in jobs] == ["mmlu_pro", "musr"]
    assert all(job["h100s"] == 1 for job in jobs)
    assert all(isinstance(job["job_id"], int) and job["job_id"] > 0 for job in jobs)
    assert all(not pathlib.Path(job["output"]).exists() for job in jobs)
    print(f"{family}\t{launcher_id}\t{dispatch_path}\t{claimed}\t"
          f"{jobs[0]['job_id']}\t{jobs[0]['output']}\t"
          f"{jobs[1]['job_id']}\t{jobs[1]['output']}")
PY
)"
mapfile -t rows <<< "$validated_lines"
test "${#rows[@]}" = 3

evaluation_job_ids=()
comparison_args=()
for row in "${rows[@]}"; do
  IFS=$'\t' read -r family launcher_id family_dispatch family_receipt \
    mmlu_job mmlu_output musr_job musr_output <<< "$row"
  case "$family" in gated_gqa|gdn_hybrid|kda_mla_hybrid) ;; *) exit 2 ;; esac
  [[ "$launcher_id" =~ ^[0-9]+$ ]]
  [[ "$mmlu_job" =~ ^[0-9]+$ ]]
  [[ "$musr_job" =~ ^[0-9]+$ ]]
  launcher_state="$(sacct -j "$launcher_id" -X -n -P -o State | sed -n '1p')"
  test "$launcher_state" = "COMPLETED"
  evaluation_job_ids+=("$mmlu_job" "$musr_job")
  comparison_args+=("--${family//_/-}-mmlu-pro" "$mmlu_output")
  comparison_args+=("--${family//_/-}-musr" "$musr_output")
done

test "${#evaluation_job_ids[@]}" = 6
test "$(printf '%s\n' "${evaluation_job_ids[@]}" | sort -u | wc -l | tr -d ' ')" = 6
dependency="$(IFS=:; printf '%s' "${evaluation_job_ids[*]}")"
comparison_job_id="$(env -i PATH=/apps/slurm/current/bin:/usr/bin:/bin sbatch --parsable \
  --chdir="$SAI_ROOT" \
  --dependency="afterok:$dependency" \
  --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/sai_100m_development_mc_compare_%j.out" \
  --error="$LOG_ROOT/sai_100m_development_mc_compare_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",OUTPUT="$OUTPUT",EVALUATION_JOB_IDS="${evaluation_job_ids[*]}",COMPARISON_ARGUMENTS="${comparison_args[*]}" \
  "$COMPARISON_JOB")"
[[ "$comparison_job_id" =~ ^[0-9]+$ ]]

"$PYTHON" -P -s -B - "$DISPATCH" "$EXPECTED_COMMIT" "$comparison_job_id" \
  "${evaluation_job_ids[*]}" "${rows[@]}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import uuid


def canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


output = pathlib.Path(sys.argv[1])
rows = [row.split("\t") for row in sys.argv[5:]]
payload = {
    "schema": "sai-short-screen-development-mc-comparison-dispatch-v1",
    "status": "submitted",
    "source_commit": sys.argv[2],
    "comparison_job_id": int(sys.argv[3]),
    "evaluation_job_ids": [int(value) for value in sys.argv[4].split()],
    "families": [
        {
            "family": row[0],
            "launcher_job_id": int(row[1]),
            "dispatch": {"path": row[2], "receipt_sha256": row[3]},
            "mmlu_pro": {"job_id": int(row[4]), "output": row[5]},
            "musr": {"job_id": int(row[6]), "output": row[7]},
        }
        for row in rows
    ],
    "gpu_jobs_submitted": 0,
    "expected_completed_single_h100_evaluations": 6,
    "retry": False,
    "requeue": False,
    "development_only": True,
    "scientific_promotion_authorized": False,
    "four_b_training_authorized": False,
}
payload["receipt_sha256"] = canonical_sha256(payload)
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, output)
    temporary.unlink()
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
PY

printf '%s\n' "comparison_job=$comparison_job_id"
