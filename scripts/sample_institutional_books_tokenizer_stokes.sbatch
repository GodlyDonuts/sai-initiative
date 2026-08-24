#!/bin/bash
#SBATCH --job-name=sai-book-tokenizer-sample
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --array=0-63%16
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-tokenizer-sample-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-tokenizer-sample-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

sai_final=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-virtual-cross-source-rewritten-20260826-r1
sai_selection=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-selection-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-tokenizer-samples-20260826-r1
sai_identity=$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")
sai_source_receipt=${sai_parent}/source-receipts/shard_${sai_identity}.json
sai_output=${sai_parent}/samples/shard_${sai_identity}

mkdir -p "${sai_parent}/source-receipts" "${sai_parent}/samples"
[[ ! -e "${sai_source_receipt}" && ! -e "${sai_output}" ]]

python -m sai.data.institutional_books_transient_tokenizer_stream \
  --final-root "${sai_final}" \
  --selection-root "${sai_selection}" \
  --receipt "${sai_source_receipt}" \
  --logical-shards 64 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
| python -m sai.data.transient_tokenizer_sample \
  --source-receipt "${sai_source_receipt}" \
  --output-root "${sai_output}" \
  --maximum-utf8-bytes 32000000
