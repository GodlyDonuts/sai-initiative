#!/bin/bash
#SBATCH --job-name=sai-bridge-foundation-reconcile
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-foundation-reconcile-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-foundation-reconcile-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
[[ -d "${SAI_RUNTIME_ROOT}/.git" || -f "${SAI_RUNTIME_ROOT}/.git" ]]

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

sai_root=/lustre/fs1/home/sa305415/sai_data_sources
sai_candidates=${sai_root}/grounded-bridge-curriculum-candidates-20260826-r1
sai_query=${sai_root}/grounded-bridge-foundation-query-20260826-r1
sai_aggregate=${sai_root}/grounded-bridge-foundation-scan-20260826-r1/aggregate
sai_output=${sai_root}/grounded-bridge-foundation-reconciled-20260826-r1
sai_evidence=${sai_root}/sai_evidence/grounded-bridge-foundation-scan/20260826-r1/reconciled-receipt.json

mkdir -p "$(dirname "${sai_evidence}")"
[[ ! -e "${sai_output}" && ! -e "${sai_evidence}" ]]

python -m sai.data.grounded_bridge_foundation_reconcile \
  --candidate-root "${sai_candidates}" \
  --query-root "${sai_query}" \
  --aggregate-root "${sai_aggregate}" \
  --output-root "${sai_output}" \
  --durable-receipt "${sai_evidence}"
