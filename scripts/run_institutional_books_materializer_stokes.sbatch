#!/bin/bash
#SBATCH --job-name=sai-book-select
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-63%64
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-select-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-select-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.institutional_books_materializer shard \
  --manifest /lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_source_reservoir_8tib_20260823_r2/manifest.jsonl \
  --reservoir-receipt /lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_source_reservoir_8tib_20260823_r2/receipt.json \
  --selection-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-selection-20260826-r1 \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-english-materialized-20260826-r1 \
  --logical-shards 64 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --token-env HF_TOKEN \
  --scratch-root "${TMPDIR:-/tmp}"
