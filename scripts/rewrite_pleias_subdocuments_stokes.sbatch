#!/bin/bash
#SBATCH --job-name=sai-pleias-subdoc-write
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-127%8
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-subdoc-rewrite-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-subdoc-rewrite-%A_%a.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_materialized=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-materialized-20260826-r1
sai_decisions=/lustre/fs1/home/sa305415/sai_data_sources/pleias-subdocument-decision-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-subdocument-rewritten-20260826-r1
sai_output=${sai_parent}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")
sai_scratch=${TMPDIR:-/tmp}

mkdir -p "${sai_parent}/shards" "${sai_scratch}"

python -m sai.data.pleias_subdocument_rewrite \
  --materialized-root "${sai_materialized}" \
  --decision-root "${sai_decisions}" \
  --output-root "${sai_output}" \
  --logical-shards 128 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --token-env HF_TOKEN \
  --scratch-root "${sai_scratch}"
