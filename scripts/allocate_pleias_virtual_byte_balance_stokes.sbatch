#!/bin/bash
#SBATCH --job-name=sai-pleias-byte-allocate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-byte-allocation-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-byte-allocation-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=${SAI_RUNTIME_ROOT}/src

sai_root=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-byte-balance-20260826-r1
mkdir -p "${sai_root}"
[[ ! -e "${sai_root}/allocation.json" ]]

python -m sai.data.pleias_virtual_byte_balance allocate \
  --book-aggregate /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-virtual-cross-source-rewritten-20260826-r1/aggregate.json \
  --pleias-final-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-final-reconstruction-20260826-r1 \
  --output "${sai_root}/allocation.json" \
  --byte-ceiling 2000000000000 \
  --reserved-bytes 1000000000
