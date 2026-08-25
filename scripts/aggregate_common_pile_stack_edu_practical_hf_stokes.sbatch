#!/usr/bin/env bash
#SBATCH --job-name=sai-stack-edu-hf-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/stack-edu-hf-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/stack-edu-hf-aggregate-%j.err

set -euo pipefail
source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.common_pile_stack_edu_practical_hf_publish aggregate \
  --admission-root /lustre/fs1/home/sa305415/sai_data_sources/common-pile-stack-edu-practical-admission-20260826-r1 \
  --publish-root /lustre/fs1/home/sa305415/sai_data_sources/common-pile-stack-edu-practical-hf-publish-20260826-r1 \
  --logical-shards 32 \
  --output /lustre/fs1/home/sa305415/sai_data_sources/common-pile-stack-edu-practical-hf-publish-20260826-r1/aggregate.json \
  --token-env HF_TOKEN
