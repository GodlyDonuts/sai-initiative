#!/usr/bin/env bash
#SBATCH --job-name=sai-pleias-admit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-practical-admit-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-practical-admit-%j.err

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
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

python -m sai.data.pleias_practical_admission \
  --manifest "${SAI_RUNTIME_ROOT}/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl" \
  --scan-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-locators-20260826-r3 \
  --books-receipt /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-practical-admission-20260826-r1/receipt.json \
  --quarantine-registry-root "${SAI_RUNTIME_ROOT}/artifacts/sai_quarantine_exclusion_registry_20260826_r6" \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-practical-admission-20260826-r3 \
  --logical-shards 128 \
  --total-text-byte-ceiling 2000000000000 \
  --output-shards 128 \
  --scratch-root "${TMPDIR:-/tmp}"
