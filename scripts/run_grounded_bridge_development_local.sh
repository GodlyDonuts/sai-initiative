#!/bin/zsh

set -euo pipefail

sai_root=${0:A:h:h}
cd "${sai_root}"

export PYTHONPATH=src
export SAI_NOUS_LOOPBACK_KEY=local-proxy

sai_candidates=artifacts/sai_grounded_bridge_development_population_20260825_r2/candidates.jsonl
sai_output_root=artifacts/sai_grounded_bridge_development_population_20260825_r2/judgments
sai_frontier_dependency=artifacts/sai_frontier_source_audit_aggregate_20260825_r1.json
sai_main_dependency=artifacts/sai_common_pile_pilot_compiler_aggregate_20260825_r1.json

if [[ ! -f "${sai_candidates}" ]]; then
  echo 'grounded bridge candidates are absent' >&2
  exit 1
fi

# Preserve two sustainable Nous lanes: begin only after either active source
# compiler closes and releases one lane.
while [[ ! -f "${sai_frontier_dependency}" && ! -f "${sai_main_dependency}" ]]; do
  sleep 10
done

for shard_index in {0..63}; do
  summary=$(printf '%s/shard_%05d.summary.json' "${sai_output_root}" "${shard_index}")
  if [[ -f "${summary}" ]]; then
    continue
  fi
  attempt=0
  while true; do
    attempt=$((attempt + 1))
    if python3 -m sai.data.nous_grounded_bridge_worker \
      --candidates "${sai_candidates}" \
      --output-root "${sai_output_root}" \
      --model stealth/ox-alpha \
      --base-url http://127.0.0.1:8645/v1 \
      --api-key-env SAI_NOUS_LOOPBACK_KEY \
      --logical-shards 64 \
      --shard-index "${shard_index}" \
      --concurrency 2 \
      --timeout-seconds 600 \
      --maximum-attempts 5 \
      --stream-transport; then
      break
    fi
    printf '{"event":"grounded_bridge_retry","shard_index":%s,"attempt":%s}\n' \
      "${shard_index}" "${attempt}"
    sleep 60
  done
done

printf '{"event":"grounded_bridge_development_complete","logical_shards":64}\n'
