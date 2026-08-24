#!/bin/bash
#SBATCH --job-name=sai-tokenizer-qualification
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/tokenizer-qualification-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/tokenizer-qualification-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

sai_pleias=/lustre/fs1/home/sa305415/sai_data_sources/pleias-tokenizer-samples-20260826-r1
sai_books=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-tokenizer-samples-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/sai-tokenizer-tournament-20260826-r1

mapfile -t sai_book_samples < <(find "${sai_books}/samples" -mindepth 2 -maxdepth 2 -type f -name sample.jsonl | sort)
mapfile -t sai_pleias_samples < <(find "${sai_pleias}/samples" -mindepth 2 -maxdepth 2 -type f -name sample.jsonl | sort)
[[ ${#sai_book_samples[@]} -eq 64 && ${#sai_pleias_samples[@]} -eq 128 ]]
[[ ! -e "${sai_parent}/qualification.json" ]]
[[ ! -e "${sai_parent}/selected-48k.json" ]]

sai_corpora=()
for sai_path in "${sai_book_samples[@]}" "${sai_pleias_samples[@]}"; do
  sai_corpora+=(--corpus "${sai_path}")
done

python -m sai.tokenizer.qualification \
  --candidate "32k=${sai_parent}/candidate-builds/32k/32k" \
  --candidate "48k=${sai_parent}/candidate-builds/48k/48k" \
  --candidate "64k=${sai_parent}/candidate-builds/64k/64k" \
  "${sai_corpora[@]}" \
  --protected-suite "${SAI_RUNTIME_ROOT}/docs/SAI_TOKENIZER_PROTECTED_SUITE.jsonl" \
  --output "${sai_parent}/qualification.json" \
  --selected-48k-output "${sai_parent}/selected-48k.json"

python -m sai.tokenizer.tournament_custody \
  --pleias-samples-root "${sai_pleias}" \
  --book-samples-root "${sai_books}" \
  --tournament-root "${sai_parent}" \
  --protected-suite "${SAI_RUNTIME_ROOT}/docs/SAI_TOKENIZER_PROTECTED_SUITE.jsonl" \
  --output "${sai_parent}/custody.json"
