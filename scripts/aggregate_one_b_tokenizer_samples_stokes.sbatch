#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-tokenizer-samples
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-tokenizer-samples-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-tokenizer-samples-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
sai_root=/lustre/fs1/home/sa305415/sai_data_sources/one-b-tokenizer-sample-20260826-r1
python -m sai.data.one_b_tokenizer_sample_aggregate \
  --root "${sai_root}" \
  --output "${sai_root}/aggregate.json"
