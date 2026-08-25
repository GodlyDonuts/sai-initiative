#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-hf-publish
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-hf-publish-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-hf-publish-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
foundation_release="${SAI_ONE_B_FOUNDATION_RELEASE:-one-b-foundation-window-20260826-r2}"
python -m sai.data.one_b_hf_publish \
  --schedule "${sai_root}/${foundation_release}/schedule/receipt.json" \
  --qualification "${sai_root}/one-b-production-tokenizer-20260826-r1/qualification.json" \
  --tokenizer-root "${sai_root}/one-b-production-tokenizer-20260826-r1/build/48k" \
  --config-receipt "${sai_root}/${foundation_release}/olmo-configs/receipt.json" \
  --output "${sai_root}/${foundation_release}/hf-publication.json" \
  --token-env HF_TOKEN
