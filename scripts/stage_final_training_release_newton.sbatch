#!/usr/bin/env bash
#SBATCH --job-name=sai-stage-final-release
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/stage-final-release-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/stage-final-release-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
: "${SAI_STOKES_SLURM_CONF_SERVER:?Stokes Slurm configuration server is required}"
: "${SAI_FOUNDATION_AUDIT_JOB_ID:?foundation audit job identity is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
[[ "${SAI_FOUNDATION_AUDIT_JOB_ID}" =~ ^[0-9]+$ ]]

export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
source /lustre/fs1/home/sa305415/hfenv/bin/activate
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
bridge_publication="${sai_root}/sai_evidence/bridge-training-component-hf-publication/20260826-r1/receipt.json"
stage_root="${sai_root}/sai_evidence/final-training-release-staging/20260826-r1"
stage_receipt="${stage_root}/receipt.json"
final_script="${SAI_RUNTIME_ROOT}/scripts/build_final_training_release_stokes.sbatch"

[[ ! -e "${stage_root}" ]]
python - "${bridge_publication}" <<'PY'
import sys
from pathlib import Path

from sai.data.bridge_component_hf_publish import SCHEMA
from sai.data.final_training_release import _load_signed

payload = _load_signed(Path(sys.argv[1]), SCHEMA)
if (
    payload.get("status") != "complete_bridge_training_component_hf_publication"
    or payload.get("training_ready") is not True
    or payload.get("development_rows_uploaded") is not False
    or payload.get("transfer_ablation_complete") is not True
):
    raise SystemExit("bridge publication is not final-release eligible")
PY

mkdir -p "$(dirname "${stage_root}")"
mkdir "${stage_root}"
final_job_id=
cleanup() {
  local exit_code=$?
  if (( exit_code != 0 )); then
    if [[ "${final_job_id}" =~ ^[0-9]+$ ]]; then
      scancel "${final_job_id}" || true
    fi
    rmdir "${stage_root}" || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT

export SLURM_CONF_SERVER="${SAI_STOKES_SLURM_CONF_SERVER}"
sacct -n -X -P -j "${SAI_FOUNDATION_AUDIT_JOB_ID}" --format=JobIDRaw | \
  grep -qx "${SAI_FOUNDATION_AUDIT_JOB_ID}|"
final_job_id="$({ sbatch --parsable \
  --dependency="afterok:${SAI_FOUNDATION_AUDIT_JOB_ID}" \
  --kill-on-invalid-dep=yes \
  --export="ALL,SAI_RUNTIME_ROOT=${SAI_RUNTIME_ROOT},SAI_RUNTIME_COMMIT=${SAI_RUNTIME_COMMIT}" \
  "${final_script}"; } | cut -d';' -f1)"
[[ "${final_job_id}" =~ ^[0-9]+$ ]]

python - "${stage_receipt}" "${bridge_publication}" \
  "${SAI_RUNTIME_COMMIT}" "${SAI_FOUNDATION_AUDIT_JOB_ID}" \
  "${final_job_id}" <<'PY'
import json
import os
import sys
from pathlib import Path

from sai.data.token_stream import canonical_sha256, sha256_file

destination, bridge_path, runtime_commit, audit_job, final_job = sys.argv[1:]
bridge = Path(bridge_path)
payload = {
    "schema": "sai-final-training-release-stage-v1",
    "status": "complete_final_training_release_staging",
    "bridge_publication": {
        "bytes": bridge.stat().st_size,
        "sha256": sha256_file(bridge),
    },
    "runtime_commit": runtime_commit,
    "foundation_audit_job": int(audit_job),
    "final_release_job": int(final_job),
    "dependency": f"afterok:{audit_job}",
    "bridge_publication_complete_before_submission": True,
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
printf 'foundation_audit=%s final_release=%s\n' \
  "${SAI_FOUNDATION_AUDIT_JOB_ID}" "${final_job_id}"
