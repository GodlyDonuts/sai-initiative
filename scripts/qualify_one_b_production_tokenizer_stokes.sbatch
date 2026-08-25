#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-tokenizer-qualify
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-tokenizer-qualify-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-tokenizer-qualify-%j.err

set -euo pipefail
: "${SAI_RUNTIME_ROOT:?immutable runtime is required}"
: "${SAI_RUNTIME_COMMIT:?immutable commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
sai_root=/lustre/fs1/home/sa305415/sai_data_sources
sai_samples=${sai_root}/one-b-tokenizer-sample-20260826-r1
sai_build=${sai_root}/one-b-production-tokenizer-20260826-r1/build
sai_output=${sai_root}/one-b-production-tokenizer-20260826-r1/qualification.json
mapfile -t sai_books < <(find "${sai_samples}/books" -mindepth 2 -maxdepth 2 -name sample.jsonl -type f | sort)
[[ ${#sai_books[@]} -eq 64 ]]
sai_corpora=()
for sai_path in "${sai_books[@]}" \
  "${sai_samples}/pleias/sample.jsonl" \
  "${sai_samples}/code/sample.jsonl" \
  "${sai_samples}/connections/sample.jsonl"; do
  [[ -s "${sai_path}" ]]
  sai_corpora+=(--corpus "${sai_path}")
done
python -m sai.tokenizer.production_qualification \
  --tokenizer-root "${sai_build}/48k" \
  --build-manifest "${sai_build}/manifest.json" \
  "${sai_corpora[@]}" \
  --protected-suite "${SAI_RUNTIME_ROOT}/docs/SAI_TOKENIZER_PROTECTED_SUITE.jsonl" \
  --output "${sai_output}"
