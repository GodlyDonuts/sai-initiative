#!/bin/bash
#SBATCH --job-name=sai-pleias-meta-dispatch
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-meta-dispatch-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-meta-dispatch-%j.err

set -euo pipefail

: "${SAI_AGGREGATE_JOB_ID:?SAI_AGGREGATE_JOB_ID is required}"

sai_repo=/lustre/fs1/home/sa305415/sai-initiative
sai_census=/lustre/fs1/home/sa305415/sai_data_sources/pleias-metadata-census-20260826-r2
sai_recovery=/lustre/fs1/home/sa305415/sai_data_sources/pleias-metadata-census-recovery-20260826-r1
sai_evidence=/lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/pleias-metadata-census-incomplete-originals/20260826-r1
sai_dispatch_receipt=${sai_recovery}/dispatch-${SLURM_JOB_ID}.jsonl

mkdir -p "${sai_recovery}" "${sai_evidence}"
: > "${sai_dispatch_receipt}.partial"

sai_merge_jobs=()
for sai_shard_index in $(seq 0 127); do
  sai_shard=$(printf '%s/shards/shard_%05d' "${sai_census}" "${sai_shard_index}")
  if [[ -f "${sai_shard}/receipt.json" ]]; then
    continue
  fi
  if [[ -e "${sai_shard}" ]]; then
    sai_incomplete=$(printf '%s/shard_%05d' "${sai_evidence}" "${sai_shard_index}")
    if [[ -e "${sai_incomplete}" ]]; then
      echo "incomplete evidence target already exists: ${sai_incomplete}" >&2
      exit 1
    fi
    mv "${sai_shard}" "${sai_incomplete}"
  fi
  sai_segment_job=$(sbatch --parsable \
    --array=0-7%8 \
    --export=ALL,SAI_SHARD_INDEX="${sai_shard_index}" \
    "${sai_repo}/scripts/recover_pleias_metadata_census_segment_stokes.sbatch")
  sai_merge_job=$(sbatch --parsable \
    --dependency=afterok:"${sai_segment_job}" \
    --export=ALL,SAI_SHARD_INDEX="${sai_shard_index}" \
    "${sai_repo}/scripts/merge_pleias_metadata_census_recovery_stokes.sbatch")
  sai_merge_jobs+=("${sai_merge_job}")
  printf '{"merge_job":"%s","segment_job":"%s","shard_index":%d}\n' \
    "${sai_merge_job}" "${sai_segment_job}" "${sai_shard_index}" \
    >> "${sai_dispatch_receipt}.partial"
done

mv "${sai_dispatch_receipt}.partial" "${sai_dispatch_receipt}"

if (( ${#sai_merge_jobs[@]} > 0 )); then
  sai_dependency=$(IFS=:; echo "${sai_merge_jobs[*]}")
  scontrol update JobId="${SAI_AGGREGATE_JOB_ID}" Dependency=afterok:"${sai_dependency}"
fi

printf '{"aggregate_job":"%s","dispatch_job":"%s","recovered_shards":%d}\n' \
  "${SAI_AGGREGATE_JOB_ID}" "${SLURM_JOB_ID}" "${#sai_merge_jobs[@]}"
