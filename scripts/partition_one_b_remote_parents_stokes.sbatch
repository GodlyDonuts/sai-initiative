#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-parent-partition
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-parent-partition-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-parent-partition-%A_%a.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
: "${SAI_PARENT_COMPONENT:?pleias or code is required}"
[[ "${SAI_PARENT_COMPONENT}" == pleias || "${SAI_PARENT_COMPONENT}" == code ]]
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
printf -v sai_shard 'shard_%05d' "${SLURM_ARRAY_TASK_ID}"
python -m sai.data.one_b_parent_partition \
  --component "${SAI_PARENT_COMPONENT}" \
  --index-root "${sai_root}/one-b-curriculum-index-20260826-r1/${SAI_PARENT_COMPONENT}/${sai_shard}" \
  --output-root "${sai_root}/one-b-parent-partitions-20260826-r1/${SAI_PARENT_COMPONENT}/${sai_shard}"
