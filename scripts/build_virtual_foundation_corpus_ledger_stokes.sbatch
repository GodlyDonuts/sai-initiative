#!/bin/bash
#SBATCH --job-name=sai-virtual-foundation-ledger
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/virtual-foundation-ledger-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/virtual-foundation-ledger-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_books=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-virtual-cross-source-rewritten-20260826-r1/aggregate.json
sai_pleias=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-final-reconstruction-20260826-r1/aggregate.json
sai_root=/lustre/fs1/home/sa305415/sai_data_sources/virtual-foundation-corpus-ledger-20260826-r1

mkdir -p "${sai_root}"

python -m sai.data.virtual_foundation_corpus_ledger \
  --book-aggregate "${sai_books}" \
  --pleias-aggregate "${sai_pleias}" \
  --output "${sai_root}/receipt.json" \
  --byte-ceiling 2000000000000
