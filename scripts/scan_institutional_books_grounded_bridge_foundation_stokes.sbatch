#!/bin/bash
#SBATCH --job-name=sai-book-bridge-scan
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-63%16
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-bridge-scan-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-bridge-scan-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
[[ -d "${SAI_RUNTIME_ROOT}/.git" || -f "${SAI_RUNTIME_ROOT}/.git" ]]

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

sai_final=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-virtual-cross-source-rewritten-20260826-r1
sai_selection=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-selection-20260826-r1
sai_query=/lustre/fs1/home/sa305415/sai_data_sources/grounded-bridge-foundation-query-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/grounded-bridge-foundation-scan-20260826-r1/books
sai_output=${sai_parent}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")

mkdir -p "${sai_parent}/shards"
[[ ! -e "${sai_output}" ]]

python -m sai.data.grounded_bridge_foundation_scan \
  --component institutional_books \
  --query-root "${sai_query}" \
  --selection-root "${sai_selection}" \
  --final-root "${sai_final}" \
  --logical-shards 64 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${sai_output}"
