#!/bin/bash
#SBATCH --job-name=sai-pleias-meta-merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-meta-merge-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-meta-merge-%j.err

set -euo pipefail

: "${SAI_SHARD_INDEX:?SAI_SHARD_INDEX is required}"

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_manifest=/lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl
sai_census_root=/lustre/fs1/home/sa305415/sai_data_sources/pleias-metadata-census-20260826-r2
sai_segments_root=$(printf '/lustre/fs1/home/sa305415/sai_data_sources/pleias-metadata-census-recovery-20260826-r1/shard_%05d' "${SAI_SHARD_INDEX}")
sai_output_root=$(printf '%s/shards/shard_%05d' "${sai_census_root}" "${SAI_SHARD_INDEX}")

rmdir "${sai_output_root}" 2>/dev/null || true

python -m sai.data.pleias_metadata_census merge-segments \
  --manifest "${sai_manifest}" \
  --segments-root "${sai_segments_root}" \
  --output-root "${sai_output_root}" \
  --logical-shards 128 \
  --shard-index "${SAI_SHARD_INDEX}" \
  --segments-per-shard 8
