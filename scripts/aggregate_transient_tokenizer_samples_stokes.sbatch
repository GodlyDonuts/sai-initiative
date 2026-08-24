#!/bin/bash
#SBATCH --job-name=sai-tokenizer-sample-aggregate
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/tokenizer-sample-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/tokenizer-sample-aggregate-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"

sai_parent=/lustre/fs1/home/sa305415/sai_data_sources/pleias-tokenizer-samples-20260826-r1

python -m sai.data.transient_tokenizer_sample_aggregate \
  --samples-root "${sai_parent}/samples" \
  --output "${sai_parent}/aggregate.json" \
  --logical-shards 128 \
  --maximum-bytes-per-shard 64000000 \
  --scratch-root "${TMPDIR:-/tmp}"
