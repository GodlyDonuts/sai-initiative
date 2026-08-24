#!/bin/bash
#SBATCH --job-name=sai-pleias-virtual-internal
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-127%8
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-virtual-internal-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-virtual-internal-%A_%a.err

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
sai_selection=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-selection-20260826-r1
sai_virtual=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-subdocument-signatures-20260826-r1
sai_decisions=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-subdocument-decision-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-internal-signatures-20260826-r1
sai_output=${sai_parent}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")
sai_scratch=${TMPDIR:-/tmp}

mkdir -p "${sai_parent}/shards" "${sai_scratch}"

python -m sai.data.pleias_virtual_internal_rewrite_signature shard \
  --manifest "${sai_manifest}" \
  --selection-root "${sai_selection}" \
  --semantic-decision /lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-stratum-decision-20260826-r1.json \
  --virtual-root "${sai_virtual}" \
  --decision-root "${sai_decisions}" \
  --output-root "${sai_output}" \
  --logical-shards 128 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --token-env HF_TOKEN \
  --scratch-root "${sai_scratch}"
