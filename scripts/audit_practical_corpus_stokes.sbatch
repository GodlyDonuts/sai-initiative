#!/usr/bin/env bash
#SBATCH --job-name=sai-practical-audit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-corpus-audit-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-corpus-audit-%j.err

set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_output=/lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/practical-corpus-readiness/20260826-r3/receipt.json
mkdir -p "$(dirname "${sai_output}")"

python -m sai.data.practical_corpus_audit \
  --books-receipt /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-practical-admission-20260826-r1/receipt.json \
  --pleias-receipt /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-admission-20260826-r3/receipt.json \
  --publication-receipt /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-hf-publish-20260826-r3/metadata-publication.json \
  --output "${sai_output}" \
  --minimum-combined-text-bytes 1900000000000 \
  --maximum-combined-text-bytes 2000000000000
