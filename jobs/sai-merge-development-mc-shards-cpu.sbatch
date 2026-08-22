#!/usr/bin/env bash
#SBATCH --job-name=sai-merge-devmc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail

: "${SAI_ROOT:?SAI_ROOT is required}"
: "${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"
: "${PYTHON:?PYTHON is required}"
: "${MANIFEST:?MANIFEST is required}"
: "${RESULT_PATHS:?RESULT_PATHS is required}"
: "${EXPECTED_RESULT_COUNT:?EXPECTED_RESULT_COUNT is required}"
: "${OUTPUT:?OUTPUT is required}"

test ! -L "$SAI_ROOT"
test "$(git -C "$SAI_ROOT" rev-parse --is-inside-work-tree)" = "true"
test "$(git -C "$SAI_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SAI_ROOT" status --short)"
test -f "$MANIFEST"
test ! -e "$OUTPUT"

case "${CUDA_VISIBLE_DEVICES:-}" in
  ""|NoDevFiles) ;;
  *) echo "CPU merge was exposed to a GPU" >&2; exit 2 ;;
esac

IFS=: read -r -a results <<< "$RESULT_PATHS"
test "${#results[@]}" = "$EXPECTED_RESULT_COUNT"
arguments=()
for result in "${results[@]}"; do
  test -f "$result"
  arguments+=(--result "$result")
done

export PYTHONPATH="$SAI_ROOT/src"
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON" -P -s -B -m sai.evaluation.development_mc_shards merge \
  --manifest "$MANIFEST" \
  "${arguments[@]}" \
  --output "$OUTPUT"
