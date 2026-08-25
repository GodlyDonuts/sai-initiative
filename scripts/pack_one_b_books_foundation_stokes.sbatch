#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-pack-books
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --array=0-63%16
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-pack-books-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-pack-books-%A_%a.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
curriculum_release="${SAI_ONE_B_CURRICULUM_RELEASE:-one-b-curriculum-index-20260826-r2}"
foundation_release="${SAI_ONE_B_FOUNDATION_RELEASE:-one-b-foundation-window-20260826-r2}"
printf -v sai_shard 'shard_%05d' "${SLURM_ARRAY_TASK_ID}"
python -m sai.data.one_b_foundation_pack books \
  --admission-root "${sai_root}/institutional-books-practical-admission-20260826-r1" \
  --index-root "${sai_root}/${curriculum_release}/books" \
  --source-shard "${SLURM_ARRAY_TASK_ID}" \
  --plan "${sai_root}/${foundation_release}/plan.json" \
  --tokenizer-root "${sai_root}/one-b-production-tokenizer-20260826-r1/build/48k" \
  --output-root "${sai_root}/${foundation_release}/packed/books/${sai_shard}"
