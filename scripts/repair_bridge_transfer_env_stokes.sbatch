#!/usr/bin/env bash
#SBATCH --job-name=sai-bridge-transfer-env-repair
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-transfer-env-repair-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/bridge-transfer-env-repair-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

sai_python=/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603/bin/python
"${sai_python}" -m pip install --disable-pip-version-check --no-deps regex==2025.11.3
"${sai_python}" - <<'PY'
import json
import regex

print(json.dumps({"regex": regex.__version__}, sort_keys=True))
PY
