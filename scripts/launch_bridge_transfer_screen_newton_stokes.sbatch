#!/usr/bin/env bash
#SBATCH --job-name=sai-bridge-transfer-launch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-transfer-launch-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-transfer-launch-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
: "${SAI_NEWTON_SLURM_CONF_SERVER:?Newton Slurm configuration server is required}"

[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]

sai_root=/lustre/fs1/home/sa305415/sai_data_sources
reconciliation_root="${sai_root}/grounded-bridge-practical-reconciled-20260826-r1"
screen_root="${sai_root}/bridge-transfer-proxy-screen-20260826-r1"
evidence_root="${sai_root}/sai_evidence/bridge-transfer-proxy-screen/20260826-r1"
state_root="${evidence_root}/launch-state"
receipt_path="${evidence_root}/launch-receipt.json"
arm_script="${SAI_RUNTIME_ROOT}/scripts/run_bridge_transfer_screen_arm_stokes.sbatch"
aggregate_script="${SAI_RUNTIME_ROOT}/scripts/aggregate_bridge_transfer_screen_stokes.sbatch"

[[ -f "${reconciliation_root}/receipt.json" ]]
[[ ! -e "${screen_root}" ]]
[[ ! -e "${receipt_path}" ]]
mkdir -p "${evidence_root}"
mkdir "${state_root}"

export SLURM_CONF_SERVER="${SAI_NEWTON_SLURM_CONF_SERVER}"
sinfo -p normal -h -o '%G' | grep -qx 'gpu:nvidia_h100_pcie:2(S:0-1)'

declare -a submitted=()
cleanup_partial_graph() {
  local exit_code=$?
  if (( exit_code != 0 && ${#submitted[@]} > 0 )); then
    scancel "${submitted[@]}" || true
  fi
  exit "${exit_code}"
}
trap cleanup_partial_graph EXIT

for arm in unchanged source_control connections; do
  job_id="$({ sbatch --parsable \
    --export="ALL,SAI_BRIDGE_TRANSFER_ARM=${arm},SAI_RUNTIME_ROOT=${SAI_RUNTIME_ROOT},SAI_RUNTIME_COMMIT=${SAI_RUNTIME_COMMIT}" \
    "${arm_script}"; } | cut -d';' -f1)"
  [[ "${job_id}" =~ ^[0-9]+$ ]]
  submitted+=("${job_id}")
  printf '%s\n' "${job_id}" > "${state_root}/${arm}.job_id"
done

dependency="afterok:${submitted[0]}:${submitted[1]}:${submitted[2]}"
aggregate_id="$({ sbatch --parsable \
  --dependency="${dependency}" \
  --export="ALL,SAI_RUNTIME_ROOT=${SAI_RUNTIME_ROOT},SAI_RUNTIME_COMMIT=${SAI_RUNTIME_COMMIT}" \
  "${aggregate_script}"; } | cut -d';' -f1)"
[[ "${aggregate_id}" =~ ^[0-9]+$ ]]
submitted+=("${aggregate_id}")
printf '%s\n' "${aggregate_id}" > "${state_root}/aggregate.job_id"

python3 - "${receipt_path}" "${SAI_RUNTIME_COMMIT}" "${SLURM_JOB_ID}" \
  "${submitted[0]}" "${submitted[1]}" "${submitted[2]}" "${aggregate_id}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    destination,
    runtime_commit,
    launcher_job_id,
    unchanged_job_id,
    source_control_job_id,
    connections_job_id,
    aggregate_job_id,
) = sys.argv[1:]
payload = {
    "schema": "sai-bridge-transfer-newton-launch-v1",
    "status": "complete_newton_transfer_graph_launch",
    "runtime_commit": runtime_commit,
    "launcher_job_id": int(launcher_job_id),
    "newton_jobs": {
        "unchanged": int(unchanged_job_id),
        "source_control": int(source_control_job_id),
        "connections": int(connections_job_id),
        "aggregate": int(aggregate_job_id),
    },
    "one_h100_per_arm": True,
    "matched_token_budget": True,
    "four_b_training_authorized": False,
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
path = Path(destination)
temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY

trap - EXIT
printf 'unchanged=%s source_control=%s connections=%s aggregate=%s\n' \
  "${submitted[0]}" "${submitted[1]}" "${submitted[2]}" "${aggregate_id}"
