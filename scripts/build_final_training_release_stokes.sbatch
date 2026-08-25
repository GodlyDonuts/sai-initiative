#!/usr/bin/env bash
#SBATCH --job-name=sai-final-training-release
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/final-training-release-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/final-training-release-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

sai_root=/lustre/fs1/home/sa305415/sai_data_sources
sai_release_root=${sai_root}/final-training-release-20260826-r1
mkdir -p "${sai_release_root}"

python -m sai.data.final_training_release \
  --foundation-audit "${sai_root}/sai_evidence/practical-corpus-readiness/20260826-r3/receipt.json" \
  --code-admission "${sai_root}/common-pile-stack-edu-practical-admission-20260826-r1/receipt.json" \
  --code-publication "${sai_root}/common-pile-stack-edu-practical-hf-publish-20260826-r1/metadata-publication.json" \
  --bridge-admission "${sai_root}/bridge-training-component-admission-20260826-r1/receipt.json" \
  --bridge-publication "${sai_root}/sai_evidence/bridge-training-component-hf-publication/20260826-r1/receipt.json" \
  --output "${sai_release_root}/release.json"

python -m sai.data.final_training_release_hf_publish \
  --release "${sai_release_root}/release.json" \
  --output "${sai_release_root}/hf-publication-receipt.json" \
  --token-env HF_TOKEN
