#!/usr/bin/env bash
#SBATCH --job-name=sai-pleias-policy
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-policy-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-policy-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.pleias_quality_core_policy \
  --metadata-census /lustre/fs1/home/sa305415/sai_data_sources/pleias-metadata-census-20260826-r2/aggregate.json \
  --quality-strata /lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_pleias_quality_strata_20260826_r1.json \
  --independent-calibration /lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_independent_review_comparison_20260826_r2/all-providers.json \
  --output /lustre/fs1/home/sa305415/sai_data_sources/pleias-quality-core-policy-20260826-r1.json
