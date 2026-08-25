#!/usr/bin/env bash
#SBATCH --job-name=sai-stack-edu-repair
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=ec57,ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/stack-edu-practical-repair-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/stack-edu-practical-repair-%j.err

set -euo pipefail

: "${SAI_STACK_EDU_SHARD_INDEX:?fixed Stack-Edu shard index is required}"
: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
[[ "${SAI_STACK_EDU_SHARD_INDEX}" =~ ^[0-9]+$ ]]
(( SAI_STACK_EDU_SHARD_INDEX >= 0 && SAI_STACK_EDU_SHARD_INDEX < 95 ))
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

sai_manifest="${SAI_RUNTIME_ROOT}/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl"
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/stack-edu-practical-locators-20260826-r1
sai_output=${sai_parent}/shards/shard_$(printf '%05d' "${SAI_STACK_EDU_SHARD_INDEX}")

mkdir -p "${sai_parent}/shards" "${TMPDIR:-/tmp}"

python -m sai.data.common_pile_stack_edu_practical_scan \
  --manifest "${sai_manifest}" \
  --output-root "${sai_output}" \
  --shard-index "${SAI_STACK_EDU_SHARD_INDEX}" \
  --expected-parents 95 \
  --token-env HF_TOKEN \
  --scratch-root "${TMPDIR:-/tmp}"
