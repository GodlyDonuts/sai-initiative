#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-pack-connect
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-pack-connect-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-pack-connect-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
python -m sai.data.one_b_foundation_pack connections \
  --admission-root "${sai_root}/bridge-training-component-admission-20260826-r1" \
  --index-root "${sai_root}/one-b-curriculum-index-20260826-r1/connections" \
  --plan "${sai_root}/one-b-foundation-window-20260826-r1/plan.json" \
  --tokenizer-root "${sai_root}/one-b-production-tokenizer-20260826-r1/build/48k" \
  --output-root "${sai_root}/one-b-foundation-window-20260826-r1/packed/connections"
