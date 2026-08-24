#!/bin/bash
#SBATCH --job-name=sai-pleias-subdoc-decide
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15%16
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-subdoc-decision-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-subdoc-decision-%A_%a.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_signatures=/lustre/fs1/home/sa305415/sai_data_sources/pleias-subdocument-signatures-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-subdocument-decision-20260826-r1
sai_output=${sai_parent}/buckets/bucket_$(printf '%02x' "${SLURM_ARRAY_TASK_ID}")
sai_scratch=/lustre/fs1/home/sa305415/sai_data_sources/tmp/pleias-subdocument-decision-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

mkdir -p "${sai_parent}/buckets" "${sai_scratch}"
trap 'rmdir "${sai_scratch}" 2>/dev/null || true' EXIT

python -m sai.data.pleias_subdocument_decision \
  --signature-root "${sai_signatures}" \
  --output-root "${sai_output}" \
  --logical-shards 128 \
  --bucket-index "${SLURM_ARRAY_TASK_ID}" \
  --chunk-records 500000 \
  --maximum-open-runs 128 \
  --temporary-root "${sai_scratch}"
