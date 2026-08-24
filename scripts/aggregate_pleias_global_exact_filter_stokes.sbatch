#!/bin/bash
#SBATCH --job-name=sai-pleias-exact-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-global-exact-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-global-exact-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_source=/lustre/fs1/home/sa305415/sai_data_sources/pleias-full-decontamination-20260826-r1
sai_decision=/lustre/fs1/home/sa305415/sai_data_sources/pleias-global-exact-decision-20260826-r1
sai_output=/lustre/fs1/home/sa305415/sai_data_sources/pleias-global-exact-filtered-20260826-r1

python -m sai.data.pleias_global_exact_dedup aggregate \
  --decontamination-aggregate "${sai_source}/aggregate.json" \
  --decision-root "${sai_decision}" \
  --filters-root "${sai_output}/shards" \
  --logical-shards 128 \
  --output "${sai_output}/aggregate.json"
