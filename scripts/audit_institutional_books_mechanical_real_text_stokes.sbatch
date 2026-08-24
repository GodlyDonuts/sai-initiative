#!/bin/bash
#SBATCH --job-name=sai-book-real-quality
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-real-quality-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-real-quality-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.institutional_books_mechanical_sample_audit \
  --materialized-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-english-materialized-20260826-r1 \
  --output /lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/institutional-books-real-mechanical-audit-20260826-r2.json \
  --logical-shards 64 \
  --rows-per-complete-shard 2
