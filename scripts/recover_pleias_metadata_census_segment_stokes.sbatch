#!/bin/bash
#SBATCH --job-name=sai-pleias-meta-recover
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --array=0-7%8
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-meta-recover-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-meta-recover-%A_%a.err

set -euo pipefail

: "${SAI_SHARD_INDEX:?SAI_SHARD_INDEX is required}"

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
sai_recovery_root=/lustre/fs1/home/sa305415/sai_data_sources/pleias-metadata-census-recovery-20260826-r1
sai_output_root=$(printf '%s/shard_%05d/segment_%05d' "${sai_recovery_root}" "${SAI_SHARD_INDEX}" "${SLURM_ARRAY_TASK_ID}")

rmdir "${sai_output_root}" 2>/dev/null || true

python -m sai.data.pleias_metadata_census segment \
  --manifest "${sai_manifest}" \
  --output-root "${sai_output_root}" \
  --logical-shards 128 \
  --shard-index "${SAI_SHARD_INDEX}" \
  --segments-per-shard 8 \
  --segment-index "${SLURM_ARRAY_TASK_ID}" \
  --token-env HF_TOKEN \
  --scratch-root "${TMPDIR:-/tmp}"
