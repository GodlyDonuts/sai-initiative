#!/usr/bin/env bash
#SBATCH --job-name=sai-practical-meta-hf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-metadata-hf-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-metadata-hf-%j.err

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

python -m sai.data.practical_hf_publish metadata \
  --books-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-practical-admission-20260826-r1 \
  --pleias-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-admission-20260826-r3 \
  --publish-aggregate /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-hf-publish-20260826-r3/aggregate.json \
  --output /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-hf-publish-20260826-r3/metadata-publication.json \
  --token-env HF_TOKEN
