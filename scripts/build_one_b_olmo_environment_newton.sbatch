#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-olmo-env
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-olmo-env-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-olmo-env-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
module load python/python-3.11.4-gcc-12.2.0 cuda/cuda-12.4.0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=8
sai_root=/lustre/fs1/home/sa305415/sai_data_sources/one-b-olmo-environment-20260826-r1
stage="${sai_root}.partial.${SLURM_JOB_ID}"
[[ ! -e "${sai_root}" && ! -e "${stage}" ]]
mkdir -p "${stage}/sources"
git -c fetch.parallel=1 clone -q https://github.com/allenai/OLMo.git "${stage}/sources/OLMo"
git -C "${stage}/sources/OLMo" checkout -q 090253dac6688f2532509daa7aa2eb5fae50e956
git -c fetch.parallel=1 clone -q https://github.com/allenai/OLMo-core.git "${stage}/sources/OLMo-core"
git -C "${stage}/sources/OLMo-core" checkout -q 7899e7cefaae44e30766ee654bd177f1e1474bc7
python -m venv "${stage}/venv"
source "${stage}/venv/bin/activate"
python -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
python -m pip install --disable-pip-version-check \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install --disable-pip-version-check "${stage}/sources/OLMo-core"
python -m pip install --disable-pip-version-check "${stage}/sources/OLMo[train]"
python -m pip install --disable-pip-version-check flash-attn==2.6.3 --no-build-isolation
python -m pip freeze --all | LC_ALL=C sort > "${stage}/pip-freeze.txt"
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
mv "${stage}" "${sai_root}"
"${sai_root}/venv/bin/python" -m sai.training.one_b_environment_receipt \
  --olmo-root "${sai_root}/sources/OLMo" \
  --core-root "${sai_root}/sources/OLMo-core" \
  --python "${sai_root}/venv/bin/python" \
  --freeze "${sai_root}/pip-freeze.txt" \
  --output "${sai_root}/receipt.json"
