#!/usr/bin/env bash
#SBATCH --job-name=sai-stack-edu-practical
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-94%95
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/stack-edu-practical-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/stack-edu-practical-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_manifest=/lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/stack-edu-practical-locators-20260826-r1
sai_output=${sai_parent}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")

mkdir -p "${sai_parent}/shards" "${TMPDIR:-/tmp}"

python -m sai.data.common_pile_stack_edu_practical_scan \
  --manifest "${sai_manifest}" \
  --output-root "${sai_output}" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --expected-parents 95 \
  --token-env HF_TOKEN \
  --scratch-root "${TMPDIR:-/tmp}"
