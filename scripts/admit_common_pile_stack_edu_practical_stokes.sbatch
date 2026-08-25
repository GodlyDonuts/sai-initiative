#!/usr/bin/env bash
#SBATCH --job-name=sai-stack-edu-admit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/stack-edu-admit-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/stack-edu-admit-%j.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export ARROW_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_manifest=/lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_hf_materialized_source_lake_20260825_r1/manifest.jsonl
sai_scan=/lustre/fs1/home/sa305415/sai_data_sources/stack-edu-practical-locators-20260826-r1
sai_quarantine=/lustre/fs1/home/sa305415/sai-initiative/artifacts/sai_quarantine_exclusion_registry_20260826_r6
sai_output=/lustre/fs1/home/sa305415/sai_data_sources/common-pile-stack-edu-practical-admission-20260826-r1

python -m sai.data.common_pile_stack_edu_practical_admission \
  --manifest "${sai_manifest}" \
  --scan-root "${sai_scan}" \
  --quarantine-registry-root "${sai_quarantine}" \
  --output-root "${sai_output}" \
  --expected-parents 95 \
  --maximum-text-bytes 150000000000 \
  --output-shards 32 \
  --scratch-root "${TMPDIR:-/tmp}"
