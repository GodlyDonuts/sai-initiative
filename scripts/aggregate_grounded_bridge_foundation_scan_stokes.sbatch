#!/bin/bash
#SBATCH --job-name=sai-bridge-scan-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-scan-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-scan-aggregate-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
[[ -d "${SAI_RUNTIME_ROOT}/.git" || -f "${SAI_RUNTIME_ROOT}/.git" ]]

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

sai_query=/lustre/fs1/home/sa305415/sai_data_sources/grounded-bridge-foundation-query-20260826-r1
sai_scan=/lustre/fs1/home/sa305415/sai_data_sources/grounded-bridge-foundation-scan-20260826-r1
sai_output=${sai_scan}/aggregate
sai_evidence=/lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/grounded-bridge-foundation-scan/20260826-r1/aggregate-receipt.json

mkdir -p "$(dirname "${sai_evidence}")"
[[ ! -e "${sai_output}" && ! -e "${sai_evidence}" ]]

python -m sai.data.grounded_bridge_foundation_scan_aggregate \
  --query-root "${sai_query}" \
  --pleias-scan-root "${sai_scan}/pleias" \
  --book-scan-root "${sai_scan}/books" \
  --output-root "${sai_output}" \
  --pleias-logical-shards 128 \
  --book-logical-shards 64 \
  --durable-receipt "${sai_evidence}"
