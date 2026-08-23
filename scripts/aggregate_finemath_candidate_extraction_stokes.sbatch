#!/bin/bash
#SBATCH --job-name=sai-fm-extract-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-extract-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-extract-aggregate-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.finemath_candidate_extraction aggregate \
  --shard-root /lustre/fs1/home/sa305415/sai_data_sources/finemath-conservative-candidate-shards-20260825-r1 \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/finemath-conservative-candidate-aggregate-20260825-r1
