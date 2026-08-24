#!/bin/bash
#SBATCH --job-name=sai-pleias-hermes-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-hermes-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-hermes-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.reservoir_audit_aggregate \
  --population-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-sample-20260826-r1 \
  --judgments-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-hermes-20260826-r1 \
  --output /lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-hermes-aggregate-20260826-r1.json
