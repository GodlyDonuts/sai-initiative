#!/usr/bin/env bash
#SBATCH --job-name=sai-bridge-admit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-component-admit-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-component-admit-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

source /lustre/fs1/home/sa305415/hfenv/bin/activate
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
python -m sai.data.bridge_component_admission \
  --reconciliation-root "${sai_root}/grounded-bridge-practical-reconciled-20260826-r1" \
  --confirmation "${sai_root}/sai_evidence/bridge-transfer-proxy-confirmation/20260826-r1/aggregate.json" \
  --output-root "${sai_root}/bridge-training-component-admission-20260826-r1" \
  --durable-receipt "${sai_root}/sai_evidence/bridge-training-component-admission/20260826-r1/receipt.json"
