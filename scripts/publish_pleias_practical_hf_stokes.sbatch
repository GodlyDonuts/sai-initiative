#!/usr/bin/env bash
#SBATCH --job-name=sai-practical-hf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --array=0-127%32
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-hf-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-hf-%A_%a.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

python -m sai.data.practical_hf_publish shard \
  --admission-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-admission-20260826-r3 \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-hf-publish-20260826-r3/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}") \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --logical-shards 128 \
  --token-env HF_TOKEN
