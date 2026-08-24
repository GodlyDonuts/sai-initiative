#!/usr/bin/env bash
#SBATCH --job-name=sai-book-hermes
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-31%16
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-hermes-%A_%a.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/book-hermes-%A_%a.err

set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /lustre/fs1/home/sa305415/hfenv/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a
export PYTHONPATH=/lustre/fs1/home/sa305415/sai-initiative/src

sai_population=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-semantic-population-20260826-r1
sai_candidates=${sai_population}/candidates.jsonl
sai_judgments=/lustre/fs1/home/sa305415/sai_data_sources/institutional-books-semantic-judgments-20260826-r1
sai_logical_shards=256

mkdir -p "${sai_judgments}"

sai_shards=$(python - "${sai_candidates}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
lane = int(sys.argv[2])
shards = set()
with path.open() as handle:
    for line in handle:
        identity = json.loads(line)["candidate_identity_sha256"]
        shards.add(int(identity, 16) % 256)
print(" ".join(str(value) for value in sorted(shards) if value % 32 == lane))
PY
)

for sai_shard in ${sai_shards}; do
  sai_summary=$(printf '%s/shard_%05d.summary.json' "${sai_judgments}" "${sai_shard}")
  if [[ -f "${sai_summary}" ]]; then
    continue
  fi
  while ! python -m sai.data.nous_book_compiler_worker \
    --candidates "${sai_candidates}" \
    --output-root "${sai_judgments}" \
    --model stealth/ox-alpha \
    --base-url https://openrouter.ai/api/v1 \
    --api-key-env OPENROUTER_API_KEY \
    --logical-shards "${sai_logical_shards}" \
    --shard-index "${sai_shard}" \
    --concurrency 2 \
    --timeout-seconds 600 \
    --maximum-attempts 5 \
    --stream-transport; do
    sleep 60
  done
done
