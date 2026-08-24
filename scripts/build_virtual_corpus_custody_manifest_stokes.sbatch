#!/bin/bash
#SBATCH --job-name=sai-virtual-corpus-custody
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/virtual-corpus-custody-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/virtual-corpus-custody-%j.err

set -euo pipefail

sai_repo=/lustre/fs1/home/sa305415/sai-initiative
cd "${sai_repo}"
git diff --quiet
git diff --cached --quiet
sai_commit=$(git rev-parse HEAD)

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=${sai_repo}/src

sai_root=/lustre/fs1/home/sa305415/sai_data_sources/virtual-corpus-custody-20260826-r1
sai_evidence=/lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/virtual-corpus-custody/20260826-r1
mkdir -p "${sai_root}" "${sai_evidence}"

python -m sai.data.virtual_corpus_custody_manifest \
  --source-manifest ${sai_repo}/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl \
  --book-aggregate /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-virtual-cross-source-rewritten-20260826-r1/aggregate.json \
  --pleias-aggregate /lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-final-reconstruction-20260826-r1/aggregate.json \
  --pleias-balance /lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-byte-balance-20260826-r1/aggregate.json \
  --cross-decision /lustre/fs1/home/sa305415/sai_data_sources/virtual-cross-source-subdocument-decision-20260826-r1/aggregate.json \
  --ledger /lustre/fs1/home/sa305415/sai_data_sources/virtual-foundation-corpus-ledger-20260826-r1/receipt.json \
  --output "${sai_root}/receipt.json" \
  --durable-output "${sai_evidence}/receipt.json" \
  --runtime-commit "${sai_commit}"
