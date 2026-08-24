#!/bin/bash
#SBATCH --job-name=sai-pleias-ind-pop
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-independent-population-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-independent-population-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.independent_review_population \
  --lane pleias=/lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-sample-20260826-r1/candidates.jsonl,/lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-hermes-20260826-r1 \
  --per-stratum 256 \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-independent-review-population-20260826-r1
