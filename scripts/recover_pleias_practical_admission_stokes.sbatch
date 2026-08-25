#!/usr/bin/env bash
#SBATCH --job-name=sai-pleias-admit-recover
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-practical-admit-recover-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-practical-admit-recover-%j.err

set -euo pipefail

: "${SAI_RECOVERY_RUNTIME_ROOT:?immutable recovery runtime root is required}"
: "${SAI_RECOVERY_RUNTIME_COMMIT:?immutable recovery runtime commit is required}"
: "${SAI_SCIENTIFIC_RUNTIME_ROOT:?immutable scientific runtime root is required}"
: "${SAI_SCIENTIFIC_RUNTIME_COMMIT:?immutable scientific runtime commit is required}"
: "${SAI_PRIMARY_ADMISSION_JOB_ID:?primary admission job identity is required}"
[[ "${SAI_PRIMARY_ADMISSION_JOB_ID}" =~ ^[0-9]+$ ]]
[[ "$(git -C "${SAI_RECOVERY_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RECOVERY_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RECOVERY_RUNTIME_ROOT}" status --porcelain)" ]]
[[ "$(git -C "${SAI_SCIENTIFIC_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_SCIENTIFIC_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_SCIENTIFIC_RUNTIME_ROOT}" status --porcelain)" ]]

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_SCIENTIFIC_RUNTIME_ROOT}/src"

sai_root=/lustre/fs1/home/sa305415/sai_data_sources
sai_output_root="${sai_root}/pleias-practical-admission-20260826-r3"
sai_receipt="${sai_output_root}/receipt.json"
sai_cleanup_receipt="${sai_root}/sai_evidence/pleias-practical-admission-recovery/20260826-r3/cleanup-primary-${SAI_PRIMARY_ADMISSION_JOB_ID}.json"
sai_primary_state="$({ sacct -X -n -P -j "${SAI_PRIMARY_ADMISSION_JOB_ID}" \
  --format=JobIDRaw,State | awk -F'|' -v job="${SAI_PRIMARY_ADMISSION_JOB_ID}" '$1 == job {print $2; exit}'; })"
[[ -n "${sai_primary_state}" ]]
[[ "${sai_primary_state}" != RUNNING* ]]
[[ "${sai_primary_state}" != PENDING* ]]

if [[ -f "${sai_receipt}" ]] && python -m sai.data.pleias_practical_admission_recovery \
  validate --receipt "${sai_receipt}" >/dev/null
then
  printf 'primary=%s state=%s recovery=noop receipt=valid\n' \
    "${SAI_PRIMARY_ADMISSION_JOB_ID}" "${sai_primary_state}"
  exit 0
fi

python -m sai.data.pleias_practical_admission_recovery cleanup \
  --root "${sai_output_root}" \
  --expected-root "${sai_output_root}" \
  --destination "${sai_cleanup_receipt}" \
  --primary-job "${SAI_PRIMARY_ADMISSION_JOB_ID}" \
  --primary-state "${sai_primary_state}" \
  >/dev/null

export SAI_RUNTIME_ROOT="${SAI_SCIENTIFIC_RUNTIME_ROOT}"
export SAI_RUNTIME_COMMIT="${SAI_SCIENTIFIC_RUNTIME_COMMIT}"
exec bash "${SAI_SCIENTIFIC_RUNTIME_ROOT}/scripts/admit_pleias_practical_core_stokes.sbatch"
