#!/usr/bin/env bash
#SBATCH --job-name=sai-practical-bridge-reconcile
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-bridge-reconcile-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-bridge-reconcile-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_root=/lustre/fs1/home/sa305415/sai_data_sources
python -m sai.data.practical_bridge_reconcile \
  --candidate-root "${sai_root}/grounded-bridge-curriculum-candidates-practical-20260826-r1" \
  --books-receipt "${sai_root}/institutional-books-practical-admission-20260826-r1/receipt.json" \
  --pleias-receipt "${sai_root}/pleias-practical-admission-20260826-r3/receipt.json" \
  --output-root "${sai_root}/grounded-bridge-practical-reconciled-20260826-r1" \
  --durable-receipt "${sai_root}/sai_evidence/grounded-bridge-practical-reconciled/20260826-r1/receipt.json"
