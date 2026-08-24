#!/bin/bash
#SBATCH --job-name=sai-book-subdoc-sig
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-63%32
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-subdocument-signature-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-subdocument-signature-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_filtered=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-mechanical-filtered-20260826-r1
sai_decontamination=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-full-decontamination-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-subdocument-signatures-20260826-r1
sai_output=${sai_parent}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")

mkdir -p "${sai_parent}/shards"

python -m sai.data.institutional_books_subdocument_signature shard \
  --filtered-root "${sai_filtered}" \
  --decontamination-root "${sai_decontamination}" \
  --output-root "${sai_output}" \
  --logical-shards 64 \
  --shard-index "${SLURM_ARRAY_TASK_ID}"
