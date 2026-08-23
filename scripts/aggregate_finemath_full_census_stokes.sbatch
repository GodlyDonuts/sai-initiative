#!/bin/bash
#SBATCH --job-name=sai-fm-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-census-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-census-aggregate-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.finemath_full_census \
  --source-root /lustre/fs1/home/sa305415/sai_data_sources/finemath-4plus-e92b25a-r1 \
  --shards-root /lustre/fs1/home/sa305415/sai_data_sources/finemath-full-census-20260825-r1 \
  --aggregate-output /lustre/fs1/home/sa305415/sai_data_sources/finemath-full-census-aggregate-20260825-r1.json
