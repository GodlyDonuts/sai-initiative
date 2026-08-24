#!/bin/bash
#SBATCH --job-name=sai-book-virtual-cross-write
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-63%16
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-virtual-cross-source-rewrite-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-virtual-cross-source-rewrite-%A_%a.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_filtered=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-mechanical-filtered-20260826-r1
sai_decontamination=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-full-decontamination-20260826-r1
sai_decision=/lustre/fs1/home/sa305415/sai_data_sources/virtual-cross-source-subdocument-decision-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-virtual-cross-source-rewritten-20260826-r1
sai_output=${sai_parent}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")
sai_scratch=${TMPDIR:-/tmp}

mkdir -p "${sai_parent}/shards" "${sai_scratch}"

python -m sai.data.institutional_books_cross_source_subdocument_rewrite \
  --filtered-root "${sai_filtered}" \
  --decontamination-root "${sai_decontamination}" \
  --decision-root "${sai_decision}" \
  --output-root "${sai_output}" \
  --logical-shards 64 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --scratch-root "${sai_scratch}"
