#!/bin/bash
#SBATCH --job-name=sai-fm-decontam
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-decontam-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-decontam-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_source_root=/lustre/fs1/home/sa305415/sai_data_sources
sai_candidate_root=${sai_source_root}/finemath-conservative-candidate-aggregate-20260825-r1
sai_boundary_root=${sai_source_root}/sai-official-benchmark-boundary-index-20260824-r2
sai_output_root=${sai_source_root}/finemath-benchmark-disjoint-candidate-20260825-r2

mkdir -p "${sai_output_root}"

python -m sai.data.decontamination build \
  --source "${sai_candidate_root}/raw-candidates.jsonl" \
  --boundary-index "${sai_boundary_root}" \
  --output "${sai_output_root}/documents.jsonl" \
  --receipt "${sai_output_root}/receipt.json" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --materialize-boundary-indexes
