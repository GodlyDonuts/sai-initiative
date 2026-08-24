#!/usr/bin/env bash
#SBATCH --job-name=sai-book-nemotron-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-nemotron-agg-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-nemotron-agg-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.institutional_books_compiler_aggregate \
  --population-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-independent-population-20260826-r1 \
  --judgments-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-independent-nemotron-20260826-r1 \
  --logical-shards 128 \
  --expected-model nvidia/nemotron-3-ultra-550b-a55b \
  --output /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-independent-nemotron-20260826-r1/aggregate.json
