#!/usr/bin/env bash
#SBATCH --job-name=sai-book-filter
#SBATCH --array=0-63%64
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-filter-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-filter-%A_%a.err

set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.institutional_books_mechanical_filter shard \
  --materialized-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-english-materialized-20260826-r1 \
  --mechanical-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-mechanical-gate-20260826-r1 \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-mechanical-filtered-20260826-r1 \
  --logical-shards 64 \
  --shard-index "${SLURM_ARRAY_TASK_ID}"
