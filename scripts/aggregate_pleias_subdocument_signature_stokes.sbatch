#!/bin/bash
#SBATCH --job-name=sai-pleias-subdoc-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-subdoc-signature-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-subdoc-signature-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_materialized=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-materialized-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-subdocument-signatures-20260826-r1

python -m sai.data.pleias_subdocument_signature aggregate \
  --materialized-root "${sai_materialized}" \
  --shards-root "${sai_parent}/shards" \
  --logical-shards 128 \
  --output "${sai_parent}/aggregate.json"
