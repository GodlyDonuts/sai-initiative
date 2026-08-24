#!/bin/bash
#SBATCH --job-name=sai-pleias-final-sig-agg
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-final-subdocument-signature-aggregate-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-final-subdocument-signature-aggregate-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_rewritten=/lustre/fs1/home/sa305415/sai_data_sources/pleias-subdocument-rewritten-20260826-r1
sai_root=/lustre/fs1/home/sa305415/sai_data_sources/pleias-final-subdocument-signatures-20260826-r1

python -m sai.data.pleias_final_subdocument_signature aggregate \
  --rewritten-root "${sai_rewritten}" \
  --shards-root "${sai_root}/shards" \
  --logical-shards 128 \
  --output "${sai_root}/aggregate.json"
