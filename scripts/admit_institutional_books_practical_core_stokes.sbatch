#!/usr/bin/env bash
#SBATCH --job-name=sai-book-practical-admit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-practical-admit-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-practical-admit-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.institutional_books_practical_admission \
  --selection-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-selection-20260826-r1 \
  --materialized-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-english-materialized-20260826-r1 \
  --filtered-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-mechanical-filtered-20260826-r1 \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-practical-admission-20260826-r1
