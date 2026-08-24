#!/bin/bash
#SBATCH --job-name=sai-virtual-cross-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/virtual-cross-subdocument-decision-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/virtual-cross-subdocument-decision-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_root=/lustre/fs1/home/sa305415/sai_data_sources/virtual-cross-source-subdocument-decision-20260826-r1

python -m sai.data.cross_source_subdocument_decision_aggregate \
  --decision-root "${sai_root}" \
  --output "${sai_root}/aggregate.json"
