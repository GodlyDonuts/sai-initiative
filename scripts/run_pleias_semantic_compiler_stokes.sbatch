#!/bin/bash
#SBATCH --job-name=sai-pleias-hermes
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=16:00:00
#SBATCH --array=0-63%16
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-hermes-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-hermes-%A_%a.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_population=/lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-sample-20260826-r1
sai_judgments=/lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-hermes-20260826-r1
sai_summary=$(printf '%s/shard_%05d.summary.json' "${sai_judgments}" "${SLURM_ARRAY_TASK_ID}")

mkdir -p "${sai_judgments}"
if [[ -f "${sai_summary}" ]]; then
  exit 0
fi

while ! python -m sai.data.nous_compiler_worker \
    --candidates "${sai_population}/candidates.jsonl" \
    --output-root "${sai_judgments}" \
    --model stealth/ox-alpha \
    --base-url https://openrouter.ai/api/v1 \
    --api-key-env OPENROUTER_API_KEY \
    --logical-shards 64 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --concurrency 2 \
    --timeout-seconds 600 \
    --maximum-attempts 5 \
    --stream-transport; do
  sleep 60
done
