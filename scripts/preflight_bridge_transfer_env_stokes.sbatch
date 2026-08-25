#!/usr/bin/env bash
#SBATCH --job-name=sai-bridge-transfer-env
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-transfer-env-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-transfer-env-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

sai_python=/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603/bin/python
"${sai_python}" - <<'PY'
import json

import huggingface_hub
import safetensors
import tokenizers
import torch
import transformers

payload = {
    "schema": "sai-bridge-transfer-environment-preflight-v1",
    "python": __import__("sys").version,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "safetensors": safetensors.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "cuda_build": torch.version.cuda,
    "smollm2_360m_revision": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    "four_b_training_authorized": False,
}
print(json.dumps(payload, sort_keys=True))
PY
