#!/usr/bin/env bash
#SBATCH --job-name=sai-1b-tokenizer-build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-tokenizer-build-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/one-b-tokenizer-build-%j.err

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
sai_output=${sai_root}/one-b-production-tokenizer-20260826-r1/build
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
python -m sai.tokenizer.build \
  "${sai_corpora[@]}" \
  --size 48k=48000 \
  --output-root "${sai_output}"
