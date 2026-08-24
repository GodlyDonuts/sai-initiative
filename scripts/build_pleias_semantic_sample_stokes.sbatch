#!/bin/bash
#SBATCH --job-name=sai-pleias-sem-pop
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-semantic-population-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-semantic-population-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_bounded=/lustre/fs1/home/sa305415/sai_data_sources/pleias-bounded-mechanical-20260826-r1

python -m sai.data.pleias_semantic_sample \
  --manifest /lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl \
  --bounded-root "${sai_bounded}" \
  --bounded-aggregate "${sai_bounded}/aggregate.json" \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-sample-20260826-r1 \
  --logical-shards 128 \
  --maximum-rows 8192 \
  --maximum-rows-per-stratum 32
