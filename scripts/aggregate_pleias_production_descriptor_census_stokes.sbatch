#!/bin/bash
#SBATCH --job-name=sai-pleias-prod-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-production-census-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-production-census-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_manifest=/lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl
sai_policy=/lustre/fs1/home/sa305415/sai_data_sources/pleias-quality-core-policy-20260826-r1.json
sai_decision=/lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-stratum-decision-20260826-r1.json
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-descriptor-census-20260826-r1

python -m sai.data.pleias_production_descriptor_census aggregate \
  --manifest "${sai_manifest}" \
  --policy "${sai_policy}" \
  --decision "${sai_decision}" \
  --shards-root "${sai_parent}/shards" \
  --logical-shards 128 \
  --output "${sai_parent}/aggregate.json"
