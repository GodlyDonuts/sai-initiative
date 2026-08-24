#!/bin/bash
#SBATCH --job-name=sai-book-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-agg-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-agg-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.institutional_books_materializer aggregate \
  --manifest /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-source-reservoir-20260823-r2/manifest.jsonl \
  --reservoir-receipt /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-source-reservoir-20260823-r2/receipt.json \
  --selection-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-selection-20260826-r1 \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-english-materialized-20260826-r1 \
  --logical-shards 64 \
  --output /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-english-materialized-20260826-r1/aggregate.json
