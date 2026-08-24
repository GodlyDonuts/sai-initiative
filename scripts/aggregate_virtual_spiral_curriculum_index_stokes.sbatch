#!/bin/bash
#SBATCH --job-name=sai-curriculum-index-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/curriculum-index-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/curriculum-index-aggregate-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
[[ -d "${SAI_RUNTIME_ROOT}/.git" || -f "${SAI_RUNTIME_ROOT}/.git" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

sai_root=/lustre/fs1/home/sa305415/sai_data_sources/virtual-spiral-curriculum-index-20260826-r1
sai_evidence=/lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/virtual-spiral-curriculum-index/20260826-r1
sai_pleias_final=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-final-reconstruction-20260826-r1
sai_book_final=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-virtual-cross-source-rewritten-20260826-r1

mkdir -p "${sai_root}" "${sai_evidence}"
python -m sai.data.virtual_spiral_curriculum_index aggregate \
  --pleias-index-root "${sai_root}/pleias" \
  --book-index-root "${sai_root}/books" \
  --pleias-final-root "${sai_pleias_final}" \
  --book-final-root "${sai_book_final}" \
  --output "${sai_root}/aggregate.json" \
  --durable-output "${sai_evidence}/aggregate.json" \
  --scratch-root "${TMPDIR:-/tmp}"
