#!/bin/bash
#SBATCH --job-name=sai-nous-label-worker
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --no-requeue

set -euo pipefail
umask 077

: "${PYTHON:?PYTHON is required}"
: "${RUNTIME_ROOT:?RUNTIME_ROOT is required}"
: "${CANDIDATES:?CANDIDATES is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${NOUS_API_KEY_FILE:?NOUS_API_KEY_FILE is required}"
: "${WORKER_INDEX:?WORKER_INDEX is required}"
: "${PHYSICAL_WORKERS:?PHYSICAL_WORKERS is required}"
: "${LOGICAL_SHARDS:=1000}"
: "${CONCURRENCY:=4}"
: "${MODEL:=stealth/ox-alpha}"

for value in "$PYTHON" "$CANDIDATES" "$NOUS_API_KEY_FILE"; do
  test -f "$value"
  test ! -L "$value"
  test "$(stat -c %h -- "$value")" = 1
done
test -d "$RUNTIME_ROOT"
test ! -L "$RUNTIME_ROOT"
test "$(stat -c %u -- "$NOUS_API_KEY_FILE")" = "$(id -u)"
case "$(stat -c %a -- "$NOUS_API_KEY_FILE")" in
  400|600) ;;
  *) echo "sai: NOUS_API_KEY_FILE must be mode 0400 or 0600" >&2; exit 2 ;;
esac
case "$WORKER_INDEX:$PHYSICAL_WORKERS:$LOGICAL_SHARDS:$CONCURRENCY" in
  *[!0-9:]*|:*|*::*) echo "sai: worker geometry differs" >&2; exit 2 ;;
esac
(( PHYSICAL_WORKERS >= 1 && PHYSICAL_WORKERS <= 250 ))
(( WORKER_INDEX >= 0 && WORKER_INDEX < PHYSICAL_WORKERS ))
(( LOGICAL_SHARDS >= PHYSICAL_WORKERS && LOGICAL_SHARDS <= 10000 ))
(( CONCURRENCY >= 1 && CONCURRENCY <= 16 ))

IFS= read -r NOUS_API_KEY < "$NOUS_API_KEY_FILE"
test -n "$NOUS_API_KEY"

for (( shard=WORKER_INDEX; shard<LOGICAL_SHARDS; shard+=PHYSICAL_WORKERS )); do
  env -i \
    PATH=/usr/bin:/bin \
    PYTHONPATH="$RUNTIME_ROOT/src" \
    PYTHONDONTWRITEBYTECODE=1 \
    NOUS_API_KEY="$NOUS_API_KEY" \
    "$PYTHON" -P -s -B -m sai.data.nous_label_worker \
      --candidates "$CANDIDATES" \
      --output-root "$OUTPUT_ROOT" \
      --model "$MODEL" \
      --logical-shards "$LOGICAL_SHARDS" \
      --shard-index "$shard" \
      --concurrency "$CONCURRENCY"
done
