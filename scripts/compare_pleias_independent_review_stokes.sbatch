#!/bin/bash
#SBATCH --job-name=sai-pleias-ind-compare
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-independent-comparison-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-independent-comparison-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.independent_review_compare \
  --population-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-independent-review-population-20260826-r1 \
  --primary pleias=/lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-hermes-20260826-r1 \
  --review nemotron=nvidia/nemotron-3-ultra-550b-a55b,https://integrate.api.nvidia.com/v1,/lustre/fs1/home/sa305415/sai_data_sources/pleias-independent-nemotron-20260826-r1 \
  --output /lustre/fs1/home/sa305415/sai_data_sources/pleias-independent-comparison-20260826-r1.json
