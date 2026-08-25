#!/usr/bin/env bash
set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
: "${SAI_NEWTON_SLURM_CONF_SERVER:?Newton Slurm configuration server is required}"

[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]

sai_root=/lustre/fs1/home/sa305415/sai_data_sources
screen_path="${sai_root}/sai_evidence/bridge-transfer-proxy-screen/20260826-r1/aggregate.json"
confirmation_root="${sai_root}/bridge-transfer-proxy-confirmation-20260826-r1"
evidence_root="${sai_root}/sai_evidence/bridge-transfer-proxy-confirmation/20260826-r1"
state_root="${evidence_root}/launch-state"
receipt_path="${evidence_root}/launch-receipt.json"
arm_script="${SAI_RUNTIME_ROOT}/scripts/run_bridge_transfer_confirmation_arm_newton.sbatch"
aggregate_script="${SAI_RUNTIME_ROOT}/scripts/aggregate_bridge_transfer_confirmation_newton.sbatch"
sai_python=/lustre/fs1/home/sa305415/hfenv/bin/python

[[ -f "${screen_path}" ]]
[[ ! -e "${confirmation_root}" ]]
[[ ! -e "${receipt_path}" ]]
PYTHONPATH="${SAI_RUNTIME_ROOT}/src" "${sai_python}" - "${screen_path}" <<'PY'
import sys
from pathlib import Path

from sai.data.bridge_transfer_confirmation import _load_screen

_load_screen(Path(sys.argv[1]))
PY

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

for seed in 20260827 20260828 20260829; do
  for arm in unchanged source_control connections; do
    job_id="$({ sbatch --parsable \
      --export="ALL,SAI_BRIDGE_TRANSFER_ARM=${arm},SAI_BRIDGE_TRANSFER_SEED=${seed},SAI_RUNTIME_ROOT=${SAI_RUNTIME_ROOT},SAI_RUNTIME_COMMIT=${SAI_RUNTIME_COMMIT}" \
      "${arm_script}"; } | cut -d';' -f1)"
    [[ "${job_id}" =~ ^[0-9]+$ ]]
    submitted+=("${job_id}")
    printf '%s\n' "${job_id}" > "${state_root}/${seed}.${arm}.job_id"
  done
done

dependency="afterok"
for job_id in "${submitted[@]}"; do dependency+=":${job_id}"; done
aggregate_id="$({ sbatch --parsable \
  --dependency="${dependency}" \
  --export="ALL,SAI_RUNTIME_ROOT=${SAI_RUNTIME_ROOT},SAI_RUNTIME_COMMIT=${SAI_RUNTIME_COMMIT}" \
  "${aggregate_script}"; } | cut -d';' -f1)"
[[ "${aggregate_id}" =~ ^[0-9]+$ ]]
submitted+=("${aggregate_id}")
printf '%s\n' "${aggregate_id}" > "${state_root}/aggregate.job_id"

PYTHONPATH="${SAI_RUNTIME_ROOT}/src" "${sai_python}" - \
  "${receipt_path}" "${screen_path}" "${SAI_RUNTIME_COMMIT}" "${aggregate_id}" \
  "${submitted[@]:0:9}" <<'PY'
import json
import os
import sys
from pathlib import Path

from sai.data.bridge_transfer_confirmation import _load_screen
from sai.data.token_stream import canonical_sha256, sha256_file

destination, screen_path, runtime_commit, aggregate_id, *arm_ids = sys.argv[1:]
if len(arm_ids) != 9:
    raise SystemExit("confirmation launch job coverage differs")
screen = _load_screen(Path(screen_path))
seeds = (20260827, 20260828, 20260829)
arms = ("unchanged", "source_control", "connections")
jobs = {}
for index, job_id in enumerate(arm_ids):
    seed = seeds[index // len(arms)]
    arm = arms[index % len(arms)]
    jobs[f"{seed}.{arm}"] = int(job_id)
payload = {
    "schema": "sai-bridge-transfer-confirmation-launch-v1",
    "status": "complete_newton_confirmation_graph_launch",
    "screen": {
        "path": Path(screen_path).name,
        "bytes": Path(screen_path).stat().st_size,
        "sha256": sha256_file(Path(screen_path)),
        "receipt_sha256": screen["receipt_sha256"],
    },
    "runtime_commit": runtime_commit,
    "seeds": list(seeds),
    "arms": list(arms),
    "arm_jobs": jobs,
    "aggregate_job": int(aggregate_id),
    "one_h100_per_arm": True,
    "matched_token_budget": True,
    "four_b_training_authorized": False,
}
payload["receipt_sha256"] = canonical_sha256(payload)
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
printf 'confirmation_arms=9 aggregate=%s\n' "${aggregate_id}"
