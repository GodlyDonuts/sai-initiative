#!/usr/bin/env bash
#SBATCH --job-name=sai-bridge-transfer-arm
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc26,evc31,evc34,evc36,evc37,evc38,evc43,evc45,evc49,evc50
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-transfer-arm-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-transfer-arm-%j.err

set -euo pipefail

: "${SAI_BRIDGE_TRANSFER_ARM:?bridge transfer arm is required}"
: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
case "${SAI_BRIDGE_TRANSFER_ARM}" in
  unchanged|source_control|connections) ;;
  *) echo 'bridge transfer arm differs' >&2; exit 2 ;;
esac

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
export HF_HOME=/lustre/fs1/home/sa305415/sai_data_sources/hf-cache/bridge-transfer-smollm2-360m

set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a

sai_python=/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603/bin/python
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
"${sai_python}" -m sai.data.bridge_transfer_screen \
  --arm "${SAI_BRIDGE_TRANSFER_ARM}" \
  --reconciliation-root "${sai_root}/grounded-bridge-practical-reconciled-20260826-r1" \
  --population-root "${sai_root}/grounded-bridge-verification-population-20260826-r1" \
  --output "${sai_root}/bridge-transfer-proxy-screen-20260826-r1/arms/${SAI_BRIDGE_TRANSFER_ARM}.json" \
  --token-env HF_TOKEN
