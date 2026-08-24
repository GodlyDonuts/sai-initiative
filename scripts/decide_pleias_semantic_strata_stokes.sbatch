#!/bin/bash
#SBATCH --job-name=sai-pleias-strata
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-semantic-strata-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-semantic-strata-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.pleias_semantic_stratum_decision \
  --population-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-sample-20260826-r1 \
  --judgments-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-hermes-20260826-r1 \
  --comparison /lustre/fs1/home/sa305415/sai_data_sources/pleias-independent-comparison-20260826-r1.json \
  --output /lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-stratum-decision-20260826-r1.json
