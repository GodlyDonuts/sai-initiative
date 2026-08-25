#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-foundation-plan
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-foundation-plan-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-foundation-plan-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
python -m sai.data.one_b_foundation_window_plan \
  --ledger "${sai_root}/one-b-curriculum-index-20260826-r1/unique-token-ledger.json" \
  --tokenizer "${sai_root}/one-b-production-tokenizer-20260826-r1/qualification.json" \
  --output "${sai_root}/one-b-foundation-window-20260826-r1/plan.json"
