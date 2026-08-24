#!/usr/bin/env bash
#SBATCH --job-name=sai-book-agreement
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-agreement-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-agreement-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.institutional_books_independent_agreement \
  --source-population-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-semantic-population-20260826-r1 \
  --independent-population-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-independent-population-20260826-r1 \
  --original-judgments-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-semantic-judgments-20260826-r1 \
  --independent-judgments-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-independent-nemotron-20260826-r1 \
  --independent-aggregate /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-independent-nemotron-20260826-r1/aggregate.json \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-independent-agreement-20260826-r1
