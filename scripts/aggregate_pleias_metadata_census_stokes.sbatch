#!/bin/bash
#SBATCH --job-name=sai-pleias-meta-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-meta-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-meta-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_manifest=/lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl
sai_root=/lustre/fs1/home/sa305415/sai_data_sources/pleias-metadata-census-20260826-r1

python -m sai.data.pleias_metadata_census aggregate \
  --manifest "${sai_manifest}" \
  --shards-root "${sai_root}/shards" \
  --logical-shards 64 \
  --output "${sai_root}/aggregate.json"
