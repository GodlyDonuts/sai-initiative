#!/bin/bash
#SBATCH --job-name=sai-pleias-exact
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-global-exact-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/pleias-global-exact-%j.err

set -euo pipefail

source /lustre/fs1/home/sa305415/hfenv/bin/activate
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_source=/lustre/fs1/home/sa305415/sai_data_sources/pleias-full-decontamination-20260826-r1

python -m sai.data.pleias_global_exact_dedup decide \
  --decontamination-root "${sai_source}" \
  --decontamination-aggregate "${sai_source}/aggregate.json" \
  --output-root /lustre/fs1/home/sa305415/sai_data_sources/pleias-global-exact-decision-20260826-r1 \
  --logical-shards 128
