#!/usr/bin/env bash
#SBATCH --job-name=sai-practical-stream-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-stream-smoke-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/practical-stream-smoke-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

python -m sai.data.practical_transient_stream \
  --manifest "${SAI_RUNTIME_ROOT}/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl" \
  --admission-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-admission-20260826-r3 \
  --shard-index 0 \
  --receipt /lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/practical-transient-stream/20260826-r3/shard_00000.json \
  --token-env HF_TOKEN \
  --scratch-root "${TMPDIR:-/tmp}" \
  >/dev/null
