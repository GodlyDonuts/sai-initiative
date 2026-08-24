#!/bin/bash
#SBATCH --job-name=sai-pleias-virtual-final
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-127%8
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-virtual-final-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-virtual-final-%A_%a.err

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
sai_internal=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-internal-signatures-20260826-r1
sai_internal_decisions=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-subdocument-decision-20260826-r1
sai_cross_decisions=/lustre/fs1/home/sa305415/sai_data_sources/virtual-cross-source-subdocument-decision-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-final-reconstruction-20260826-r1
sai_output=${sai_parent}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")
sai_scratch=${TMPDIR:-/tmp}

mkdir -p "${sai_parent}/shards" "${sai_scratch}"

python -m sai.data.pleias_virtual_cross_source_reconstruction shard \
  --manifest "${sai_manifest}" \
  --selection-root "${sai_selection}" \
  --semantic-decision /lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-stratum-decision-20260826-r1.json \
  --internal-root "${sai_internal}" \
  --internal-decision-root "${sai_internal_decisions}" \
  --cross-decision-root "${sai_cross_decisions}" \
  --output-root "${sai_output}" \
  --logical-shards 128 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --token-env HF_TOKEN \
  --scratch-root "${sai_scratch}"
