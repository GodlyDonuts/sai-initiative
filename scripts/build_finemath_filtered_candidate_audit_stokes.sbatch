#!/bin/bash
#SBATCH --job-name=sai-fm-audit-r2
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-audit-r2-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/finemath-audit-r2-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_root=/lustre/fs1/home/sa305415/sai_data_sources
sai_output_root=${sai_root}/finemath-candidate-audit-population-20260825-r2

mkdir -p "${sai_output_root}"

python -m sai.data.agent_candidate_population \
  --source "${sai_root}/finemath-contextless-answer-key-filtered-20260825-r1/documents.jsonl" \
  --output "${sai_output_root}/candidates.jsonl" \
  --receipt "${sai_output_root}/receipt.json" \
  --source-revision e92b25a616738fe95dc186b64dfb19f9c8525594 \
  --source-type educational_web \
  --sample-size 512 \
  --seed 20260826
