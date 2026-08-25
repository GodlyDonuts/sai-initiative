#!/usr/bin/env bash
#SBATCH --job-name=sai-bridge-transfer-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-transfer-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-transfer-aggregate-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

source /lustre/fs1/home/sa305415/hfenv/bin/activate
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
python -m sai.data.bridge_transfer_screen_aggregate \
  --arm-root "${sai_root}/bridge-transfer-proxy-screen-20260826-r1/arms" \
  --output "${sai_root}/sai_evidence/bridge-transfer-proxy-screen/20260826-r1/aggregate.json"
