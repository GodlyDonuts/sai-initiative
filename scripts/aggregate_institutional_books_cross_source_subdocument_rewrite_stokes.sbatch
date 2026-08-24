#!/bin/bash
#SBATCH --job-name=sai-book-cross-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-cross-source-rewrite-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-cross-source-rewrite-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_filtered=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-mechanical-filtered-20260826-r1
sai_decontamination=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-full-decontamination-20260826-r1
sai_decision=/lustre/fs1/home/sa305415/sai_data_sources/cross-source-subdocument-decision-20260826-r1
sai_root=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-cross-source-rewritten-20260826-r1

python -m sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate \
  --filtered-root "${sai_filtered}" \
  --decontamination-root "${sai_decontamination}" \
  --decision-root "${sai_decision}" \
  --rewrite-root "${sai_root}" \
  --output "${sai_root}/aggregate.json" \
  --logical-shards 64
