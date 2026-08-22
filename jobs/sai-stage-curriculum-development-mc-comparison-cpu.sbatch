#!/bin/bash
#SBATCH --job-name=sai-stage-curriculum-compare
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
: "${ORDER_COMPARISON_JOB_ID:?ORDER_COMPARISON_JOB_ID is required}"
: "${ORDER_COMPARISON:?ORDER_COMPARISON is required}"
: "${CURRICULUM_LAUNCH_JOB_ID:?CURRICULUM_LAUNCH_JOB_ID is required}"
: "${CONTROL_LAUNCH_JOB_ID:?CONTROL_LAUNCH_JOB_ID is required}"

COMPARISON_JOB="$SAI_ROOT/jobs/sai-compare-curriculum-development-benchmarks-cpu.sbatch"
DISPATCH="$OUTPUT_ROOT/comparison.dispatch.json"
OUTPUT="$OUTPUT_ROOT/comparison.json"

for value in "$ORDER_COMPARISON_JOB_ID" "$CURRICULUM_LAUNCH_JOB_ID" \
  "$CONTROL_LAUNCH_JOB_ID"; do [[ "$value" =~ ^[0-9]+$ ]]; done
for path in "$SAI_ROOT" "$PYTHON" "$LOG_ROOT" "$OUTPUT_ROOT" \
  "$ORDER_COMPARISON" "$COMPARISON_JOB" "$DISPATCH" "$OUTPUT"; do
  case "$path" in /*) ;; *) echo "path must be absolute: $path" >&2; exit 2 ;; esac
done
case "${CUDA_VISIBLE_DEVICES:-}" in ""|NoDevFiles) ;; *) exit 2 ;; esac
test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -x "$PYTHON"
test -d "$LOG_ROOT" && test ! -L "$LOG_ROOT"
test -d "$OUTPUT_ROOT" && test ! -L "$OUTPUT_ROOT"
test -f "$ORDER_COMPARISON" && test ! -L "$ORDER_COMPARISON"
test -f "$COMPARISON_JOB" && test ! -L "$COMPARISON_JOB"
test ! -e "$DISPATCH" && test ! -e "$OUTPUT"

job_record="$(scontrol show job -o "$SLURM_JOB_ID")"
stdout_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdOut=//p')"
stderr_path="$(printf '%s\n' "$job_record" | tr ' ' '\n' | sed -n 's/^StdErr=//p')"
case "$stdout_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac
case "$stderr_path" in "$LOG_ROOT"/*) ;; *) exit 2 ;; esac

for job_id in "$CURRICULUM_LAUNCH_JOB_ID" "$CONTROL_LAUNCH_JOB_ID"; do
  row="$(sacct -X -j "$job_id" -n -P -o State,ExitCode,Restarts | sed -n '1p')"
  test "$row" = "COMPLETED|0:0|0"
done

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1

validated_lines="$("$PYTHON" -P -s -B - "$OUTPUT_ROOT" \
  "$CURRICULUM_LAUNCH_JOB_ID" "$CONTROL_LAUNCH_JOB_ID" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for arm, launcher_id in zip(("curriculum", "order_control"), map(int, sys.argv[2:]), strict=True):
    dispatch_path = root / arm / "gated_gqa" / "dispatch.json"
    assert dispatch_path.is_file() and not dispatch_path.is_symlink()
    payload = json.loads(dispatch_path.read_text())
    unsigned = dict(payload); claimed = unsigned.pop("receipt_sha256")
    canonical = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert payload["schema"] == "sai-short-screen-development-mc-dispatch-v1"
    assert payload["status"] == "submitted" and payload["family"] == "gated_gqa"
    assert payload["development_only"] is True and payload["four_b_training_authorized"] is False
    assert payload["gpu_jobs_submitted"] == 9 and payload["cpu_merge_jobs_submitted"] == 1
    assert claimed == canonical
    mmlu, musr = payload["jobs"]
    assert mmlu["benchmark"] == "mmlu_pro" and mmlu["shard_count"] == 8
    assert len(mmlu["shard_job_ids"]) == 8 and len(set(mmlu["shard_job_ids"])) == 8
    assert musr["benchmark"] == "musr" and musr["h100s"] == 1
    for job in (mmlu, musr):
        output = pathlib.Path(job["output"]); assert output.is_absolute() and not output.is_symlink()
        if output.exists(): assert output.is_file()
    print("\t".join((arm, str(launcher_id), str(dispatch_path), claimed, str(mmlu["job_id"]), mmlu["output"], str(musr["job_id"]), musr["output"], " ".join(map(str, mmlu["shard_job_ids"])))))
PY
)"
mapfile -t rows <<< "$validated_lines"
test "${#rows[@]}" = 2

terminal_ids=()
h100_ids=()
comparison_exports=()
for row in "${rows[@]}"; do
  IFS=$'\t' read -r arm launcher_id arm_dispatch arm_receipt \
    mmlu_job mmlu_output musr_job musr_output shard_jobs <<< "$row"
  case "$arm" in curriculum|order_control) ;; *) exit 2 ;; esac
  terminal_ids+=("$mmlu_job" "$musr_job")
  read -r -a shards <<< "$shard_jobs"
  test "${#shards[@]}" = 8
  h100_ids+=("${shards[@]}" "$musr_job")
  comparison_exports+=("$arm" "$mmlu_job" "$mmlu_output" "$musr_job" "$musr_output")
done
test "${#terminal_ids[@]}" = 4
test "$(printf '%s\n' "${terminal_ids[@]}" | sort -u | wc -l | tr -d ' ')" = 4
test "${#h100_ids[@]}" = 18
test "$(printf '%s\n' "${h100_ids[@]}" | sort -u | wc -l | tr -d ' ')" = 18
dependency="$(IFS=:; printf '%s' "${terminal_ids[*]}")"

comparison_job_id="$(env -i PATH=/apps/slurm/current/bin:/usr/bin:/bin sbatch --parsable \
  --chdir="$SAI_ROOT" --dependency="afterok:$dependency" \
  --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/sai_compare_curriculum_boards_%j.out" \
  --error="$LOG_ROOT/sai_compare_curriculum_boards_%j.err" \
  --export=SAI_ROOT="$SAI_ROOT",EXPECTED_COMMIT="$EXPECTED_COMMIT",PYTHON="$PYTHON",LOG_ROOT="$LOG_ROOT",ORDER_COMPARISON_JOB_ID="$ORDER_COMPARISON_JOB_ID",ORDER_COMPARISON="$ORDER_COMPARISON",CURRICULUM_MMLU_JOB_ID="${comparison_exports[1]}",CURRICULUM_MMLU="${comparison_exports[2]}",CURRICULUM_MUSR_JOB_ID="${comparison_exports[3]}",CURRICULUM_MUSR="${comparison_exports[4]}",CONTROL_MMLU_JOB_ID="${comparison_exports[6]}",CONTROL_MMLU="${comparison_exports[7]}",CONTROL_MUSR_JOB_ID="${comparison_exports[8]}",CONTROL_MUSR="${comparison_exports[9]}",H100_EVALUATION_JOB_IDS="${h100_ids[*]}",OUTPUT="$OUTPUT" \
  "$COMPARISON_JOB")"
[[ "$comparison_job_id" =~ ^[0-9]+$ ]]

"$PYTHON" -P -s -B - "$DISPATCH" "$EXPECTED_COMMIT" "$comparison_job_id" \
  "${terminal_ids[*]}" "${h100_ids[*]}" "${rows[@]}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import uuid

path = pathlib.Path(sys.argv[1]); rows = [row.split("\t") for row in sys.argv[6:]]
payload = {
    "schema": "sai-curriculum-development-mc-comparison-dispatch-v1",
    "status": "submitted",
    "source_commit": sys.argv[2],
    "comparison_job_id": int(sys.argv[3]),
    "terminal_dependency_job_ids": [int(value) for value in sys.argv[4].split()],
    "evaluation_h100_job_ids": [int(value) for value in sys.argv[5].split()],
    "arms": [
        {"arm": row[0], "launcher_job_id": int(row[1]), "dispatch": {"path": row[2], "receipt_sha256": row[3]}, "mmlu_pro": {"merge_job_id": int(row[4]), "output": row[5], "shard_job_ids": [int(value) for value in row[8].split()]}, "musr": {"job_id": int(row[6]), "output": row[7]}}
        for row in rows
    ],
    "expected_completed_single_h100_evaluations": 18,
    "comparison_terminal_dependencies": 4,
    "development_only": True,
    "architecture_promotion_authorized": False,
    "four_b_training_authorized": False,
    "retry": False,
    "requeue": False,
}
payload["receipt_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    os.link(temporary, path); temporary.unlink()
except BaseException:
    temporary.unlink(missing_ok=True); raise
PY

printf 'comparison_job=%s\n' "$comparison_job_id"
