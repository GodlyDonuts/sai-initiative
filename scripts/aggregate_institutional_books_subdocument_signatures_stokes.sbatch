#!/bin/bash
#SBATCH --job-name=sai-book-subdoc-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-subdocument-signature-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-subdocument-signature-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_filtered=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-mechanical-filtered-20260826-r1
sai_decontamination=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-full-decontamination-20260826-r1
sai_root=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-subdocument-signatures-20260826-r1

python -m sai.data.institutional_books_subdocument_signature aggregate \
  --filtered-root "${sai_filtered}" \
  --decontamination-root "${sai_decontamination}" \
  --shards-root "${sai_root}/shards" \
  --logical-shards 64 \
  --output "${sai_root}/aggregate.json"
