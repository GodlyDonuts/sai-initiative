#!/bin/bash
#SBATCH --job-name=sai-tokenizer-candidate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/tokenizer-candidate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/tokenizer-candidate-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_TOKENIZER_NAME:?tokenizer candidate name is required}"
: "${SAI_TOKENIZER_SIZE:?tokenizer candidate size is required}"
case "${SAI_TOKENIZER_NAME}=${SAI_TOKENIZER_SIZE}" in
  32k=32000|48k=48000|64k=64000) ;;
  *) exit 2 ;;
esac

export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

sai_pleias=/lustre/fs1/home/sa305415/sai_data_sources/pleias-tokenizer-samples-20260826-r1
sai_books=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-tokenizer-samples-20260826-r1
sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/sai-tokenizer-tournament-20260826-r1
sai_output=${sai_parent}/candidate-builds/${SAI_TOKENIZER_NAME}

[[ -f "${sai_pleias}/aggregate.json" && -f "${sai_books}/aggregate.json" ]]
mapfile -t sai_book_samples < <(find "${sai_books}/samples" -mindepth 2 -maxdepth 2 -type f -name sample.jsonl | sort)
mapfile -t sai_pleias_samples < <(find "${sai_pleias}/samples" -mindepth 2 -maxdepth 2 -type f -name sample.jsonl | sort)
[[ ${#sai_book_samples[@]} -eq 64 && ${#sai_pleias_samples[@]} -eq 128 ]]
[[ ! -e "${sai_output}" ]]
mkdir -p "${sai_parent}/candidate-builds"

sai_corpora=()
for sai_path in "${sai_book_samples[@]}" "${sai_pleias_samples[@]}"; do
  sai_corpora+=(--corpus "${sai_path}")
done

python -m sai.tokenizer.build \
  "${sai_corpora[@]}" \
  --size "${SAI_TOKENIZER_NAME}=${SAI_TOKENIZER_SIZE}" \
  --output-root "${sai_output}"
