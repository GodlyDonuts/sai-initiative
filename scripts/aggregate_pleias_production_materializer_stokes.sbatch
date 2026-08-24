#!/bin/bash
#SBATCH --job-name=sai-pleias-prod-mat-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-production-materializer-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-production-materializer-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_manifest=/lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl
sai_selection=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-selection-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-materialized-20260826-r1

python -m sai.data.pleias_production_materializer aggregate \
  --manifest "${sai_manifest}" \
  --selection-root "${sai_selection}" \
  --shards-root "${sai_parent}/shards" \
  --logical-shards 128 \
  --output "${sai_parent}/aggregate.json" \
  --token-env HF_TOKEN
