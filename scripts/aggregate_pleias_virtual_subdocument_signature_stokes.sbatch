#!/bin/bash
#SBATCH --job-name=sai-pleias-virtual-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-virtual-signature-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-virtual-signature-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_selection=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-selection-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-subdocument-signatures-20260826-r1

python -m sai.data.pleias_virtual_subdocument_signature aggregate \
  --selection-root "${sai_selection}" \
  --shards-root "${sai_parent}/shards" \
  --logical-shards 128 \
  --output "${sai_parent}/aggregate.json"
