#!/usr/bin/env bash
set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
: "${SAI_NEWTON_SLURM_CONF_PATH:?Newton Slurm configuration path is required}"
: "${SAI_NEWTON_SLURM_CONF_SHA256:?Newton Slurm configuration hash is required}"
: "${SAI_STOKES_SLURM_CONF_PATH:?Stokes Slurm configuration path is required}"
: "${SAI_STOKES_SLURM_CONF_SHA256:?Stokes Slurm configuration hash is required}"
: "${SAI_FOUNDATION_AUDIT_JOB_ID:?foundation audit job identity is required}"
[[ "${SAI_FOUNDATION_AUDIT_JOB_ID}" =~ ^[0-9]+$ ]]

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
admission_script="${SAI_RUNTIME_ROOT}/scripts/admit_bridge_component_newton.sbatch"
publication_script="${SAI_RUNTIME_ROOT}/scripts/publish_bridge_component_hf_newton.sbatch"
final_stage_script="${SAI_RUNTIME_ROOT}/scripts/stage_final_training_release_newton.sbatch"
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
source "${SAI_RUNTIME_ROOT}/scripts/verified_slurm_config.sh"
sai_activate_verified_slurm_config \
  "${SAI_NEWTON_SLURM_CONF_PATH}" \
  "${SAI_NEWTON_SLURM_CONF_SHA256}" \
  newton
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

admission_id="$({ sbatch --parsable \
  --dependency="afterok:${aggregate_id}" \
  --export="ALL,SAI_RUNTIME_ROOT=${SAI_RUNTIME_ROOT},SAI_RUNTIME_COMMIT=${SAI_RUNTIME_COMMIT}" \
  "${admission_script}"; } | cut -d';' -f1)"
[[ "${admission_id}" =~ ^[0-9]+$ ]]
submitted+=("${admission_id}")
printf '%s\n' "${admission_id}" > "${state_root}/admission.job_id"

publication_id="$({ sbatch --parsable \
  --dependency="afterok:${admission_id}" \
  --export="ALL,SAI_RUNTIME_ROOT=${SAI_RUNTIME_ROOT},SAI_RUNTIME_COMMIT=${SAI_RUNTIME_COMMIT}" \
  "${publication_script}"; } | cut -d';' -f1)"
[[ "${publication_id}" =~ ^[0-9]+$ ]]
submitted+=("${publication_id}")
printf '%s\n' "${publication_id}" > "${state_root}/publication.job_id"

final_stage_id="$({ sbatch --parsable \
  --dependency="afterok:${publication_id}" \
  --export="ALL,SAI_RUNTIME_ROOT=${SAI_RUNTIME_ROOT},SAI_RUNTIME_COMMIT=${SAI_RUNTIME_COMMIT},SAI_STOKES_SLURM_CONF_PATH=${SAI_STOKES_SLURM_CONF_PATH},SAI_STOKES_SLURM_CONF_SHA256=${SAI_STOKES_SLURM_CONF_SHA256},SAI_FOUNDATION_AUDIT_JOB_ID=${SAI_FOUNDATION_AUDIT_JOB_ID}" \
  "${final_stage_script}"; } | cut -d';' -f1)"
[[ "${final_stage_id}" =~ ^[0-9]+$ ]]
submitted+=("${final_stage_id}")
printf '%s\n' "${final_stage_id}" > "${state_root}/final-stage.job_id"

PYTHONPATH="${SAI_RUNTIME_ROOT}/src" "${sai_python}" - \
  "${receipt_path}" "${screen_path}" "${SAI_RUNTIME_COMMIT}" \
  "${aggregate_id}" "${admission_id}" "${publication_id}" "${final_stage_id}" \
  "${submitted[@]:0:9}" <<'PY'
import json
import os
import sys
from pathlib import Path

from sai.data.bridge_transfer_confirmation import _load_screen
from sai.data.token_stream import canonical_sha256, sha256_file

(
    destination,
    screen_path,
    runtime_commit,
    aggregate_id,
    admission_id,
    publication_id,
    final_stage_id,
    *arm_ids,
) = sys.argv[1:]
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
    "admission_job": int(admission_id),
    "publication_job": int(publication_id),
    "final_release_stage_job": int(final_stage_id),
    "foundation_audit_job": int(os.environ["SAI_FOUNDATION_AUDIT_JOB_ID"]),
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
printf 'confirmation_arms=9 aggregate=%s admission=%s publication=%s final_stage=%s\n' \
  "${aggregate_id}" "${admission_id}" "${publication_id}" "${final_stage_id}"
