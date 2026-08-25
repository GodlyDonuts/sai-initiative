#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-pack-remote
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-127%8
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-pack-remote-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-pack-remote-%A_%a.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
: "${SAI_PACK_COMPONENT:?pleias or code is required}"
[[ "${SAI_PACK_COMPONENT}" == pleias || "${SAI_PACK_COMPONENT}" == code ]]
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
partition_release="${SAI_ONE_B_PARTITION_RELEASE:-one-b-parent-partitions-20260826-r2}"
foundation_release="${SAI_ONE_B_FOUNDATION_RELEASE:-one-b-foundation-window-20260826-r2}"
printf -v sai_bucket 'bucket_%03d' "${SLURM_ARRAY_TASK_ID}"
python -m sai.data.one_b_foundation_pack "${SAI_PACK_COMPONENT}" \
  --partition-root "${sai_root}/${partition_release}/${SAI_PACK_COMPONENT}" \
  --bucket "${SLURM_ARRAY_TASK_ID}" \
  --manifest "${SAI_RUNTIME_ROOT}/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl" \
  --plan "${sai_root}/${foundation_release}/plan.json" \
  --tokenizer-root "${sai_root}/one-b-production-tokenizer-20260826-r1/build/48k" \
  --token-env HF_TOKEN \
  --scratch-root "${TMPDIR:-/tmp}" \
  --output-root "${sai_root}/${foundation_release}/packed/${SAI_PACK_COMPONENT}/${sai_bucket}"
