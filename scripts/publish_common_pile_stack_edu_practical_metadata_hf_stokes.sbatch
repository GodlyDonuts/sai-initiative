#!/usr/bin/env bash
#SBATCH --job-name=sai-stack-edu-meta-hf
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/stack-edu-metadata-hf-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/stack-edu-metadata-hf-%j.err

set -euo pipefail
source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.common_pile_stack_edu_practical_hf_publish metadata \
  --admission-root /lustre/fs1/home/sa305415/sai_data_sources/common-pile-stack-edu-practical-admission-20260826-r1 \
  --aggregate /lustre/fs1/home/sa305415/sai_data_sources/common-pile-stack-edu-practical-hf-publish-20260826-r1/aggregate.json \
  --output /lustre/fs1/home/sa305415/sai_data_sources/common-pile-stack-edu-practical-hf-publish-20260826-r1/metadata-publication.json \
  --token-env HF_TOKEN
