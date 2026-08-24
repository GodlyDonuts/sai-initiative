#!/bin/bash
#SBATCH --job-name=sai-pleias-subdoc-write-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-subdoc-rewrite-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-subdoc-rewrite-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_materialized=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-materialized-20260826-r1
sai_rewritten=/lustre/fs1/home/sa305415/sai_data_sources/pleias-subdocument-rewritten-20260826-r1

python -m sai.data.pleias_subdocument_rewrite_aggregate \
  --materialized-root "${sai_materialized}" \
  --rewrite-root "${sai_rewritten}" \
  --output "${sai_rewritten}/aggregate.json" \
  --logical-shards 128 \
  --token-env HF_TOKEN
