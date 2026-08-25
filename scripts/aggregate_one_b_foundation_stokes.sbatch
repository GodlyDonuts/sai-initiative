#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-foundation-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-foundation-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-foundation-aggregate-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
foundation_release="${SAI_ONE_B_FOUNDATION_RELEASE:-one-b-foundation-window-20260826-r2}"
python -m sai.data.one_b_foundation_aggregate \
  --pack-root "${sai_root}/${foundation_release}/packed" \
  --plan "${sai_root}/${foundation_release}/plan.json" \
  --output-root "${sai_root}/${foundation_release}/exact"
