#!/bin/bash
#SBATCH --job-name=sai-pleias-exact-filter
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-127%64
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-global-exact-filter-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-global-exact-filter-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_source=/lustre/fs1/home/sa305415/sai_data_sources/pleias-full-decontamination-20260826-r1
sai_output=/lustre/fs1/home/sa305415/sai_data_sources/pleias-global-exact-filtered-20260826-r1

mkdir -p "${sai_output}/shards"

python -m sai.data.pleias_global_exact_dedup filter-shard \
  --decontamination-root "${sai_source}" \
  --decontamination-aggregate "${sai_source}/aggregate.json" \
  --decision-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-global-exact-decision-20260826-r1 \
  --output-root "${sai_output}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")" \
  --logical-shards 128 \
  --shard-index "${SLURM_ARRAY_TASK_ID}"
