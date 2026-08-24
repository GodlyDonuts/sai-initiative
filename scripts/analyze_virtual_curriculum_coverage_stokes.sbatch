#!/bin/bash
#SBATCH --job-name=sai-curriculum-coverage
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/curriculum-coverage-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/curriculum-coverage-%j.err

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

sai_index=/lustre/fs1/home/sa305415/sai_data_sources/virtual-spiral-curriculum-index-20260826-r1
sai_evidence=/lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/virtual-curriculum-coverage/20260826-r1

mkdir -p "${sai_evidence}"
python -m sai.data.virtual_curriculum_coverage \
  --index-root "${sai_index}" \
  --output "${sai_index}/coverage.json" \
  --durable-output "${sai_evidence}/coverage.json" \
  --scratch-root "${TMPDIR:-/tmp}"
