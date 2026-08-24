#!/bin/bash
#SBATCH --job-name=sai-virtual-cross-subdoc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-15%16
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/virtual-cross-subdocument-decision-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/virtual-cross-subdocument-decision-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_books=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-subdocument-signatures-20260826-r1
sai_pleias=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-internal-signatures-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/virtual-cross-source-subdocument-decision-20260826-r1
sai_bucket=$(printf '%02x' "${SLURM_ARRAY_TASK_ID}")
sai_output=${sai_parent}/buckets/bucket_${sai_bucket}
sai_scratch=${TMPDIR:-/tmp}

mkdir -p "${sai_parent}/buckets" "${sai_scratch}"

python -m sai.data.cross_source_subdocument_decision \
  --book-signature-root "${sai_books}" \
  --pleias-signature-root "${sai_pleias}" \
  --output-root "${sai_output}" \
  --bucket-index "${SLURM_ARRAY_TASK_ID}" \
  --book-logical-shards 64 \
  --pleias-logical-shards 128 \
  --temporary-root "${sai_scratch}"
