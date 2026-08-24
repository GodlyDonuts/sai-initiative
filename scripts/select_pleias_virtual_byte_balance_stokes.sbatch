#!/bin/bash
#SBATCH --job-name=sai-pleias-byte-select
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --array=0-127%32
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-byte-selection-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-byte-selection-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=${SAI_RUNTIME_ROOT}/src

sai_root=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-byte-balance-20260826-r1
sai_output=${sai_root}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")
[[ ! -e "${sai_output}" ]]

python -m sai.data.pleias_virtual_byte_balance shard \
  --pleias-final-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-final-reconstruction-20260826-r1 \
  --allocation "${sai_root}/allocation.json" \
  --output-root "${sai_output}" \
  --logical-shards 128 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --scratch-root "${TMPDIR:-/tmp}"
