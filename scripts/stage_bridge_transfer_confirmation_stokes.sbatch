#!/usr/bin/env bash
#SBATCH --job-name=sai-stage-bridge-confirm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/stage-bridge-confirmation-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/stage-bridge-confirmation-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
: "${SAI_SCREEN_RUNTIME_COMMIT:?initial screen runtime commit is required}"
: "${SAI_NEWTON_SLURM_CONF_PATH:?Newton Slurm configuration path is required}"
: "${SAI_NEWTON_SLURM_CONF_SHA256:?Newton Slurm configuration hash is required}"
: "${SAI_STOKES_SLURM_CONF_PATH:?Stokes Slurm configuration path is required}"
: "${SAI_STOKES_SLURM_CONF_SHA256:?Stokes Slurm configuration hash is required}"
: "${SAI_FOUNDATION_AUDIT_JOB_ID:?foundation audit job identity is required}"
[[ "${SAI_FOUNDATION_AUDIT_JOB_ID}" =~ ^[0-9]+$ ]]

[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
source /lustre/fs1/home/sa305415/hfenv/bin/activate

sai_root=/lustre/fs1/home/sa305415/sai_data_sources
screen_launch="${sai_root}/sai_evidence/bridge-transfer-proxy-screen/20260826-r1/launch-receipt.json"
stage_parent="${sai_root}/sai_evidence/bridge-transfer-proxy-confirmation/20260826-r1"
stage_root="${stage_parent}/staging"
stage_receipt="${stage_root}/receipt.json"
confirmation_launcher="${SAI_RUNTIME_ROOT}/scripts/launch_bridge_transfer_confirmation_newton.sh"

[[ ! -e "${stage_root}" ]]
[[ ! -e "${sai_root}/sai_evidence/bridge-transfer-proxy-confirmation/20260826-r1/launch-receipt.json" ]]
mkdir -p "${stage_parent}"
mkdir "${stage_root}"
confirmation_launcher_id=
cleanup() {
  local exit_code=$?
  if (( exit_code != 0 )); then
    if [[ "${confirmation_launcher_id}" =~ ^[0-9]+$ ]]; then
      scancel "${confirmation_launcher_id}" || true
    fi
    rmdir "${stage_root}" || true
  fi
  exit "${exit_code}"
}
trap cleanup EXIT
screen_aggregate_id="$(python -m sai.data.bridge_transfer_confirmation_stage inspect \
  --screen-launch "${screen_launch}" \
  --screen-runtime-commit "${SAI_SCREEN_RUNTIME_COMMIT}")"
[[ "${screen_aggregate_id}" =~ ^[0-9]+$ ]]

source "${SAI_RUNTIME_ROOT}/scripts/verified_slurm_config.sh"
sai_activate_verified_slurm_config \
  "${SAI_NEWTON_SLURM_CONF_PATH}" \
  "${SAI_NEWTON_SLURM_CONF_SHA256}" \
  newton
sacct -n -X -P -j "${screen_aggregate_id}" --format=JobIDRaw | \
  grep -qx "${screen_aggregate_id}|"
confirmation_launcher_id="$({ sbatch --parsable \
  --partition=normal \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=1G \
  --time=00:10:00 \
  --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50 \
  --no-requeue \
  --dependency="afterok:${screen_aggregate_id}" \
  --kill-on-invalid-dep=yes \
  --output="${sai_root}/logs/bridge-confirmation-launch-%j.out" \
  --error="${sai_root}/logs/bridge-confirmation-launch-%j.err" \
  --export="ALL,SAI_RUNTIME_ROOT=${SAI_RUNTIME_ROOT},SAI_RUNTIME_COMMIT=${SAI_RUNTIME_COMMIT},SAI_NEWTON_SLURM_CONF_PATH=${SAI_NEWTON_SLURM_CONF_PATH},SAI_NEWTON_SLURM_CONF_SHA256=${SAI_NEWTON_SLURM_CONF_SHA256},SAI_STOKES_SLURM_CONF_PATH=${SAI_STOKES_SLURM_CONF_PATH},SAI_STOKES_SLURM_CONF_SHA256=${SAI_STOKES_SLURM_CONF_SHA256},SAI_FOUNDATION_AUDIT_JOB_ID=${SAI_FOUNDATION_AUDIT_JOB_ID}" \
  "${confirmation_launcher}"; } | cut -d';' -f1)"
if [[ ! "${confirmation_launcher_id}" =~ ^[0-9]+$ ]]; then
  exit 2
fi

python -m sai.data.bridge_transfer_confirmation_stage record \
  --screen-launch "${screen_launch}" \
  --screen-runtime-commit "${SAI_SCREEN_RUNTIME_COMMIT}" \
  --confirmation-runtime-commit "${SAI_RUNTIME_COMMIT}" \
  --confirmation-launcher-job "${confirmation_launcher_id}" \
  --output "${stage_receipt}"

trap - EXIT
printf 'screen_aggregate=%s confirmation_launcher=%s\n' \
  "${screen_aggregate_id}" "${confirmation_launcher_id}"
