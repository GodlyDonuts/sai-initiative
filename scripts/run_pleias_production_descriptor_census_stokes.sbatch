#!/bin/bash
#SBATCH --job-name=sai-pleias-prod-census
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-127%64
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-production-census-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-production-census-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_manifest=/lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl
sai_policy=/lustre/fs1/home/sa305415/sai_data_sources/pleias-quality-core-policy-20260826-r1.json
sai_decision=/lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-stratum-decision-20260826-r1.json
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-descriptor-census-20260826-r1
sai_output=${sai_parent}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")
sai_scratch=${TMPDIR:-/tmp}

mkdir -p "${sai_parent}/shards" "${sai_scratch}"

python -m sai.data.pleias_production_descriptor_census shard \
  --manifest "${sai_manifest}" \
  --policy "${sai_policy}" \
  --decision "${sai_decision}" \
  --output-root "${sai_output}" \
  --logical-shards 128 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --token-env HF_TOKEN \
  --scratch-root "${sai_scratch}"
