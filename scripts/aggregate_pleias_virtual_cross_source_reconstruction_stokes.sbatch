#!/bin/bash
#SBATCH --job-name=sai-pleias-virtual-final-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-virtual-final-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-virtual-final-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_internal=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-internal-signatures-20260826-r1
sai_cross_decisions=/lustre/fs1/home/sa305415/sai_data_sources/virtual-cross-source-subdocument-decision-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-final-reconstruction-20260826-r1

python -m sai.data.pleias_virtual_cross_source_reconstruction aggregate \
  --internal-root "${sai_internal}" \
  --cross-decision-root "${sai_cross_decisions}" \
  --shards-root "${sai_parent}/shards" \
  --logical-shards 128 \
  --output "${sai_parent}/aggregate.json"
