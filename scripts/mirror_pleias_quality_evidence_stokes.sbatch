#!/bin/bash
#SBATCH --job-name=sai-pleias-evidence
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-evidence-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-evidence-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.pleias_quality_evidence_mirror \
  --source-root /lustre/fs1/home/sa305415/sai_data_sources \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/pleias-quality/20260826-r1
