#!/usr/bin/env bash
#SBATCH --job-name=sai-book-semantic-decision
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-semantic-decision-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-semantic-decision-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.institutional_books_semantic_decision \
  --population-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-semantic-population-20260826-r1 \
  --judgments-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-semantic-judgments-20260826-r1 \
  --aggregate /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-semantic-judgments-20260826-r1/aggregate.json \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-semantic-decisions-20260826-r1
