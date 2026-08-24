#!/usr/bin/env bash
#SBATCH --job-name=sai-book-evidence
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-evidence-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-evidence-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.institutional_books_evidence_mirror \
  --source-root /lustre/fs1/home/sa305415/sai_data_sources \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/institutional-books-quality/20260826-r1
