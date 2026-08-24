#!/usr/bin/env bash
#SBATCH --job-name=sai-book-full-decontam
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-full-decontam-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-full-decontam-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.institutional_books_full_decontamination \
  --agreement-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-independent-agreement-20260826-r1 \
  --filtered-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-mechanical-filtered-20260826-r1 \
  --boundary-index /lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_official_benchmark_boundary_index_20260824_r2 \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-full-decontamination-20260826-r1
