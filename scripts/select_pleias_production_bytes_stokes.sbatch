#!/bin/bash
#SBATCH --job-name=sai-pleias-prod-select
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-production-selection-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-production-selection-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_exact=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-normalized-exact-20260826-r1
sai_near=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-near-20260826-r1
sai_output=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-selection-20260826-r1

python -m sai.data.pleias_production_byte_selection \
  --exact-root "${sai_exact}" \
  --near-root "${sai_near}" \
  --output-root "${sai_output}" \
  --maximum-bytes 2000000000000
