#!/bin/bash
#SBATCH --job-name=sai-pleias-decontam
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --array=0-127%64
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-decontam-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-decontam-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_bounded=/lustre/fs1/home/sa305415/sai_data_sources/pleias-bounded-mechanical-20260826-r1
sai_output=/lustre/fs1/home/sa305415/sai_data_sources/pleias-full-decontamination-20260826-r1

mkdir -p "${sai_output}/shards"

python -m sai.data.pleias_full_candidate_decontamination shard \
  --manifest /lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl \
  --bounded-root "${sai_bounded}" \
  --bounded-aggregate "${sai_bounded}/aggregate.json" \
  --decision /lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-stratum-decision-20260826-r1.json \
  --boundary-index /lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_official_benchmark_boundary_index_20260824_r2 \
  --output-root "${sai_output}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")" \
  --logical-shards 128 \
  --shard-index "${SLURM_ARRAY_TASK_ID}"
