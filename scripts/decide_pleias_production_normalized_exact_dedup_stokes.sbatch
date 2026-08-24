#!/bin/bash
#SBATCH --job-name=sai-pleias-prod-exact
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-production-normalized-exact-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-production-normalized-exact-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_descriptors=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-descriptor-census-20260826-r1
sai_output=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-normalized-exact-20260826-r1

python -m sai.data.pleias_production_normalized_exact_dedup \
  --descriptor-root "${sai_descriptors}" \
  --aggregate "${sai_descriptors}/aggregate.json" \
  --output-root "${sai_output}" \
  --logical-shards 128
