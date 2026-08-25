#!/usr/bin/env bash
#SBATCH --job-name=sai-practical-hf-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-hf-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-hf-aggregate-%j.err

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

python -m sai.data.practical_hf_publish aggregate \
  --admission-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-admission-20260826-r3 \
  --publish-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-hf-publish-20260826-r3 \
  --logical-shards 128 \
  --output /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-hf-publish-20260826-r3/aggregate.json \
  --token-env HF_TOKEN
