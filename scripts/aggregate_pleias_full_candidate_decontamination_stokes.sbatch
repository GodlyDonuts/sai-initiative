#!/bin/bash
#SBATCH --job-name=sai-pleias-decon-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-decontam-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-decontam-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_bounded=/lustre/fs1/home/sa305415/sai_data_sources/pleias-bounded-mechanical-20260826-r1
sai_output=/lustre/fs1/home/sa305415/sai_data_sources/pleias-full-decontamination-20260826-r1

python -m sai.data.pleias_full_candidate_decontamination aggregate \
  --manifest /lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl \
  --bounded-aggregate "${sai_bounded}/aggregate.json" \
  --decision /lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-stratum-decision-20260826-r1.json \
  --shards-root "${sai_output}/shards" \
  --logical-shards 128 \
  --output "${sai_output}/aggregate.json"
