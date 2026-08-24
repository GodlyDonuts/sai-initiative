#!/bin/bash
#SBATCH --job-name=sai-pleias-tokenizer-sample
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-127%16
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-tokenizer-sample-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-tokenizer-sample-%A_%a.err

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
sai_internal_decisions=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-subdocument-decision-20260826-r1
sai_cross_decisions=/lustre/fs1/home/sa305415/sai_data_sources/virtual-cross-source-subdocument-decision-20260826-r1
sai_final=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-final-reconstruction-20260826-r1
sai_balance=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-byte-balance-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-tokenizer-samples-20260826-r1
sai_identity=$(printf '%05d' "${SLURM_ARRAY_TASK_ID}")
sai_source_receipt=${sai_parent}/source-receipts/shard_${sai_identity}.json
sai_output=${sai_parent}/samples/shard_${sai_identity}
sai_scratch=${TMPDIR:-/tmp}

mkdir -p "${sai_parent}/source-receipts" "${sai_parent}/samples" "${sai_scratch}"
[[ ! -e "${sai_source_receipt}" && ! -e "${sai_output}" ]]

python -m sai.data.pleias_virtual_transient_stream \
  --manifest "${sai_manifest}" \
  --selection-root "${sai_selection}" \
  --semantic-decision "${sai_semantic}" \
  --internal-decision-root "${sai_internal_decisions}" \
  --cross-decision-root "${sai_cross_decisions}" \
  --final-root "${sai_final}" \
  --balance-root "${sai_balance}" \
  --receipt "${sai_source_receipt}" \
  --logical-shards 128 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --token-env HF_TOKEN \
  --scratch-root "${sai_scratch}" \
| python -m sai.data.transient_tokenizer_sample \
  --source-receipt "${sai_source_receipt}" \
  --output-root "${sai_output}" \
  --maximum-utf8-bytes 64000000
