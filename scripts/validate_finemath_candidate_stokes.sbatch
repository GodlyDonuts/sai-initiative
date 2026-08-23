#!/bin/bash
#SBATCH --job-name=sai-fm-validate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-validate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-validate-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_root=/lustre/fs1/home/sa305415/sai_data_sources

python -m sai.data.decontamination validate \
  --receipt "${sai_root}/finemath-benchmark-disjoint-candidate-20260825-r2/receipt.json" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --materialize-boundary-indexes
