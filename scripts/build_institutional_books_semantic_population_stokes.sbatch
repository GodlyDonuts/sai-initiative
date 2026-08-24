#!/usr/bin/env bash
#SBATCH --job-name=sai-book-semantic-pop
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-semantic-pop-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-semantic-pop-%j.err

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

sai_metadata_root=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-metadata-e0c0b860
sai_metadata=${sai_metadata_root}/data/train-00000-of-00001.parquet
mkdir -p "${sai_metadata_root}"

python - <<'PY'
import os

from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="institutional/institutional-books-hl-metadata",
    filename="data/train-00000-of-00001.parquet",
    repo_type="dataset",
    revision="e0c0b860482e49cced246c5cd7c0d22ac8a86040",
    token=os.environ["HF_TOKEN"],
    local_dir=(
        "/lustre/fs1/home/sa305415/sai_data_sources/"
        "institutional-books-metadata-e0c0b860"
    ),
)
PY

python -m sai.data.institutional_books_semantic_population \
  --filtered-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-mechanical-filtered-20260826-r1 \
  --selection-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-strict-selection-20260826-r1 \
  --metadata "${sai_metadata}" \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/institutional-books-semantic-population-20260826-r1 \
  --maximum-candidates 8192
