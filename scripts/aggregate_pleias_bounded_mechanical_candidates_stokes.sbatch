#!/bin/bash
#SBATCH --job-name=sai-pleias-row-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-row-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-row-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_root=/lustre/fs1/home/sa305415/sai_data_sources/pleias-bounded-mechanical-20260826-r1

python -m sai.data.pleias_bounded_mechanical_candidates aggregate \
  --manifest /lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl \
  --policy /lustre/fs1/home/sa305415/sai_data_sources/pleias-quality-core-policy-20260826-r1.json \
  --shards-root "${sai_root}/shards" \
  --logical-shards 128 \
  --maximum-text-bytes-per-shard 1610612736 \
  --output "${sai_root}/aggregate.json"
