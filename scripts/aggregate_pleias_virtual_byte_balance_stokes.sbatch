#!/bin/bash
#SBATCH --job-name=sai-pleias-byte-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-byte-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-byte-aggregate-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=${SAI_RUNTIME_ROOT}/src

sai_root=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-byte-balance-20260826-r1
[[ ! -e "${sai_root}/aggregate.json" ]]

python -m sai.data.pleias_virtual_byte_balance aggregate \
  --allocation "${sai_root}/allocation.json" \
  --shards-root "${sai_root}" \
  --output "${sai_root}/aggregate.json"
