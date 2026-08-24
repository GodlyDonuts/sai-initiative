#!/bin/bash
#SBATCH --job-name=sai-pleias-bridge-scan
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-127%16
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-bridge-scan-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-bridge-scan-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
[[ -d "${SAI_RUNTIME_ROOT}/.git" || -f "${SAI_RUNTIME_ROOT}/.git" ]]

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

sai_manifest=${SAI_RUNTIME_ROOT}/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl
sai_selection=/lustre/fs1/home/sa305415/sai_data_sources/pleias-production-selection-20260826-r1
sai_semantic=/lustre/fs1/home/sa305415/sai_data_sources/pleias-semantic-stratum-decision-20260826-r1.json
sai_internal=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-subdocument-decision-20260826-r1
sai_cross=/lustre/fs1/home/sa305415/sai_data_sources/virtual-cross-source-subdocument-decision-20260826-r1
sai_final=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-final-reconstruction-20260826-r1
sai_query=/lustre/fs1/home/sa305415/sai_data_sources/grounded-bridge-foundation-query-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/grounded-bridge-foundation-scan-20260826-r1/pleias
sai_output=${sai_parent}/shards/shard_$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")
sai_scratch=${TMPDIR:-/tmp}

mkdir -p "${sai_parent}/shards" "${sai_scratch}"
[[ ! -e "${sai_output}" ]]

python -m sai.data.grounded_bridge_foundation_scan \
  --component pleias_common_corpus \
  --query-root "${sai_query}" \
  --manifest "${sai_manifest}" \
  --selection-root "${sai_selection}" \
  --semantic-decision "${sai_semantic}" \
  --internal-decision-root "${sai_internal}" \
  --cross-decision-root "${sai_cross}" \
  --final-root "${sai_final}" \
  --logical-shards 128 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --output-root "${sai_output}" \
  --token-env HF_TOKEN \
  --scratch-root "${sai_scratch}"
