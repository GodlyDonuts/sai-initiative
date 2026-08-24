#!/bin/bash
#SBATCH --job-name=sai-pleias-nemotron
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=16:00:00
#SBATCH --array=0-31%8
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-nemotron-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-nemotron-%A_%a.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

python -m sai.data.independent_compiler_review_worker \
  --candidates /lustre/fs1/home/sa305415/sai_data_sources/pleias-independent-review-population-20260826-r1/candidates.jsonl \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-independent-nemotron-20260826-r1 \
  --model nvidia/nemotron-3-ultra-550b-a55b \
  --base-url https://integrate.api.nvidia.com/v1 \
  --api-key-env NVIDIA_API_KEY \
  --logical-shards 32 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --concurrency 1 \
  --timeout-seconds 600 \
  --maximum-attempts 5 \
  --stream-transport
