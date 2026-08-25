#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-index-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-index-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-index-aggregate-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
curriculum_release="${SAI_ONE_B_CURRICULUM_RELEASE:-one-b-curriculum-index-20260826-r2}"
python -m sai.data.one_b_curriculum_aggregate \
  --index-root "${sai_root}/${curriculum_release}" \
  --release "${sai_root}/final-training-release-20260826-r1/release.json" \
  --book-admission "${sai_root}/institutional-books-practical-admission-20260826-r1/receipt.json" \
  --pleias-admission "${sai_root}/pleias-practical-admission-20260826-r3/receipt.json" \
  --code-admission "${sai_root}/common-pile-stack-edu-practical-admission-20260826-r1/receipt.json" \
  --bridge-admission "${sai_root}/bridge-training-component-admission-20260826-r1/receipt.json" \
  --output "${sai_root}/${curriculum_release}/aggregate.json"
