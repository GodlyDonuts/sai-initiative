#!/bin/bash
#SBATCH --job-name=sai-fm-census
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-63%64
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-census-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-census-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_source_root=/lustre/fs1/home/sa305415/sai_data_sources/finemath-4plus-e92b25a-r1
sai_output_parent=/lustre/fs1/home/sa305415/sai_data_sources/finemath-full-census-20260825-r1
sai_output_root=${sai_output_parent}/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")

mkdir -p "${sai_output_parent}"

python -m sai.data.finemath_full_census \
  --source-root "${sai_source_root}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${sai_output_root}"
