#!/bin/bash

set -euo pipefail

sai_root=$(cd "$(dirname "$0")/.." && pwd)
cd "${sai_root}"

set -a
source .env
set +a
export PYTHONPATH=src

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is required}"

# Five total in-flight calls (three local plus the two-call Stokes Books worker)
# eliminated the avoidable retry load still observed at six.
export SAI_OPENROUTER_SHARED_PROVIDER_CONCURRENCY=3

sai_candidates=artifacts/sai_pdr_representation_population_20260826_r2/candidates.jsonl
sai_output=artifacts/sai_pdr_grounded_representation_generation_20260826_r2/judgments
sai_logical_shards=128
sai_lanes=8
sai_expected_records=758

[[ -f "${sai_candidates}" ]] || {
  echo "PDR grounded-representation candidates are missing" >&2
  exit 1
}
mkdir -p "${sai_output}"

run_lane() {
  local sai_lane=$1
  local sai_shard
  local sai_summary
  local sai_attempt
  for sai_shard in $(seq "${sai_lane}" "${sai_lanes}" $((sai_logical_shards - 1))); do
    sai_summary=$(printf '%s/shard_%05d.summary.json' "${sai_output}" "${sai_shard}")
    [[ -f "${sai_summary}" ]] && continue
    sai_attempt=0
    while true; do
      sai_attempt=$((sai_attempt + 1))
      if python3 -m sai.data.nous_grounded_representation_worker \
        --candidates "${sai_candidates}" \
        --output-root "${sai_output}" \
        --model stealth/ox-alpha \
        --base-url https://openrouter.ai/api/v1 \
        --api-key-env OPENROUTER_API_KEY \
        --logical-shards "${sai_logical_shards}" \
        --shard-index "${sai_shard}" \
        --concurrency 2 \
        --timeout-seconds 600 \
        --maximum-attempts 5 \
        --stream-transport; then
        break
      fi
      printf '{"event":"pdr_grounded_representation_retry","lane":%s,"shard_index":%s,"attempt":%s}\n' \
        "${sai_lane}" "${sai_shard}" "${sai_attempt}" >&2
      sleep 30
    done
  done
}

sai_pids=()
for sai_lane in $(seq 0 $((sai_lanes - 1))); do
  run_lane "${sai_lane}" &
  sai_pids+=("$!")
done
for sai_pid in "${sai_pids[@]}"; do
  wait "${sai_pid}"
done

sai_records=$(find "${sai_output}" -maxdepth 1 -type f \
  -name '*.grounded-representation.json' | wc -l | tr -d ' ')
sai_summaries=$(find "${sai_output}" -maxdepth 1 -type f \
  -name 'shard_*.summary.json' | wc -l | tr -d ' ')
if [[ "${sai_records}" != "${sai_expected_records}" \
      || "${sai_summaries}" != "${sai_logical_shards}" ]]; then
  echo "PDR grounded-representation generation custody is incomplete" >&2
  exit 1
fi

printf '{"event":"pdr_grounded_representation_generation_complete","records":%s,"summaries":%s}\n' \
  "${sai_records}" "${sai_summaries}"
