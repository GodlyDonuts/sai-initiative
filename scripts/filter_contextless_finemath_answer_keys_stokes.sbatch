#!/bin/bash
#SBATCH --job-name=sai-fm-answer-key
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-answer-key-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-answer-key-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_root=/lustre/fs1/home/sa305415/sai_data_sources
sai_output_root=${sai_root}/finemath-contextless-answer-key-filtered-20260825-r1

mkdir -p "${sai_output_root}"

python -m sai.data.contextless_answer_key_filter build \
  --source "${sai_root}/finemath-benchmark-disjoint-candidate-20260825-r2/documents.jsonl" \
  --output "${sai_output_root}/documents.jsonl" \
  --receipt "${sai_output_root}/receipt.json"

python -m sai.data.contextless_answer_key_filter validate \
  --receipt "${sai_output_root}/receipt.json"
