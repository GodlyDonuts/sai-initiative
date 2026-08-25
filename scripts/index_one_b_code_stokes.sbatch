#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-index-code
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --array=0-31%16
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-index-code-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-index-code-%A_%a.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
curriculum_release="${SAI_ONE_B_CURRICULUM_RELEASE:-one-b-curriculum-index-20260826-r2}"
printf -v sai_shard 'shard_%05d' "${SLURM_ARRAY_TASK_ID}"
python -m sai.data.one_b_curriculum_index code \
  --admission-root "${sai_root}/common-pile-stack-edu-practical-admission-20260826-r1" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${sai_root}/${curriculum_release}/code/${sai_shard}"
