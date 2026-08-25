#!/usr/bin/env bash
#SBATCH --job-name=sai-pleias-practical-repair
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-practical-repair-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-practical-repair-%j.err

set -euo pipefail

: "${SAI_PLEIAS_SHARD_INDEX:?fixed PleIAs shard index is required}"
: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
[[ "${SAI_PLEIAS_SHARD_INDEX}" =~ ^[0-9]+$ ]]
(( SAI_PLEIAS_SHARD_INDEX >= 0 && SAI_PLEIAS_SHARD_INDEX < 128 ))
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
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-locators-20260826-r3
sai_output=${sai_parent}/shards/shard_$(printf '%05d' "${SAI_PLEIAS_SHARD_INDEX}")

mkdir -p "${sai_parent}/shards" "${TMPDIR:-/tmp}"

python -m sai.data.pleias_practical_locator_scan \
  --manifest "${sai_manifest}" \
  --output-root "${sai_output}" \
  --logical-shards 128 \
  --shard-index "${SAI_PLEIAS_SHARD_INDEX}" \
  --token-env HF_TOKEN \
  --sample-ppm 1000000 \
  --maximum-text-bytes 15625000000 \
  --stop-at-byte-cap \
  --scratch-root "${TMPDIR:-/tmp}"
