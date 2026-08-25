#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-h100-preflight
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-h100-preflight-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-h100-preflight-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
module load cuda/cuda-12.4.0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=1
export CUDA_MODULE_LOADING=LAZY TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
sai_data_root=/lustre/fs1/home/sa305415/sai_data_sources
foundation_release="${SAI_ONE_B_FOUNDATION_RELEASE:-one-b-foundation-window-20260826-r2}"
environment_root="${sai_data_root}/one-b-olmo-environment-20260826-r1"
config_root="${sai_data_root}/${foundation_release}/olmo-configs"
"${environment_root}/venv/bin/python" -m sai.training.one_b_gpu_preflight \
  --environment "${environment_root}/receipt.json" \
  --config-receipt "${config_root}/receipt.json" \
  --config-root "${config_root}" \
  --output "${sai_data_root}/${foundation_release}/h100-preflight.json"
