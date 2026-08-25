#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-readiness
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-readiness-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-readiness-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
curriculum_release=one-b-curriculum-index-20260826-r2
foundation_release=one-b-foundation-window-20260826-r2
python -m sai.training.one_b_readiness \
  --release "${sai_root}/final-training-release-20260826-r1/release.json" \
  --ledger "${sai_root}/${curriculum_release}/unique-token-ledger.json" \
  --plan "${sai_root}/${foundation_release}/plan.json" \
  --window "${sai_root}/${foundation_release}/exact/receipt.json" \
  --schedule "${sai_root}/${foundation_release}/schedule/receipt.json" \
  --tokenizer "${sai_root}/one-b-production-tokenizer-20260826-r1/qualification.json" \
  --config "${sai_root}/${foundation_release}/olmo-configs/receipt.json" \
  --environment "${sai_root}/one-b-olmo-environment-20260826-r1/receipt.json" \
  --preflight "${sai_root}/${foundation_release}/h100-preflight.json" \
  --hf-publication "${sai_root}/${foundation_release}/hf-publication.json" \
  --output "${sai_root}/${foundation_release}/training-readiness.json"
