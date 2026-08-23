#!/bin/zsh

set -euo pipefail

sai_root=${0:A:h:h}
cd "${sai_root}"

export PYTHONPATH=src
export SAI_NOUS_LOOPBACK_KEY=local-proxy

sai_main_aggregate=artifacts/sai_common_pile_pilot_compiler_aggregate_20260825_r1.json
sai_frontier_aggregate=artifacts/sai_frontier_source_audit_aggregate_20260825_r1.json
sai_population_root=artifacts/sai_compiler_prerequisite_edge_population_20260826_r1
sai_judgments_root=${sai_population_root}/judgments
sai_output_root=artifacts/sai_compiler_prerequisite_edge_verification_aggregate_20260826_r1
sai_expected=192
sai_logical_shards=64

if [[ -e "${sai_output_root}" ]]; then
  echo 'prerequisite edge aggregate already exists; refusing duplicate' >&2
  exit 1
fi

for sai_wait_index in {1..34560}; do
  if [[ -f "${sai_main_aggregate}" && -f "${sai_frontier_aggregate}" ]]; then
    break
  fi
  sleep 10
done

if [[ ! -f "${sai_main_aggregate}" || ! -f "${sai_frontier_aggregate}" ]]; then
  echo 'compiler aggregates did not complete within ninety-six hours' >&2
  exit 1
fi

if [[ ! -e "${sai_population_root}" ]]; then
  python3 -m sai.data.compiler_prerequisite_edge_population \
    --candidates artifacts/sai_common_pile_pilot_compiler_20260825_r1/candidates.jsonl \
    --judgments artifacts/sai_common_pile_pilot_compiler_20260825_r1/judgments \
    --logical-shards 128 \
    --candidates artifacts/sai_frontier_source_audit_20260824_r1/candidates.jsonl \
    --judgments artifacts/sai_frontier_source_audit_20260824_r1/judgments \
    --logical-shards 64 \
    --output-root "${sai_population_root}" \
    --target-edges "${sai_expected}"
fi

if [[ ! -f "${sai_population_root}/receipt.json" ]]; then
  echo 'prerequisite edge population is incomplete' >&2
  exit 1
fi

for shard_index in {0..63}; do
  summary=$(printf '%s/shard_%05d.summary.json' "${sai_judgments_root}" "${shard_index}")
  if [[ -f "${summary}" ]]; then
    continue
  fi
  attempt=0
  while true; do
    attempt=$((attempt + 1))
    if python3 -m sai.data.nous_compiler_prerequisite_edge_verifier \
      --candidates "${sai_population_root}/candidates.jsonl" \
      --output-root "${sai_judgments_root}" \
      --model stealth/ox-alpha \
      --base-url http://127.0.0.1:8645/v1 \
      --api-key-env SAI_NOUS_LOOPBACK_KEY \
      --logical-shards "${sai_logical_shards}" \
      --shard-index "${shard_index}" \
      --concurrency 2 \
      --timeout-seconds 600 \
      --maximum-attempts 5 \
      --stream-transport; then
      break
    fi
    printf '{"event":"prerequisite_edge_retry","shard_index":%s,"attempt":%s}\n' \
      "${shard_index}" "${attempt}"
    sleep 60
  done
done

sai_receipts=$(find "${sai_judgments_root}" -maxdepth 1 -type f -name '*.prerequisite-edge-verification.json' | wc -l | tr -d ' ')
sai_summaries=$(find "${sai_judgments_root}" -maxdepth 1 -type f -name 'shard_*.summary.json' | wc -l | tr -d ' ')
if [[ "${sai_receipts}" != "${sai_expected}" || "${sai_summaries}" != "${sai_logical_shards}" ]]; then
  echo 'prerequisite edge verification custody is incomplete' >&2
  exit 1
fi

python3 -m sai.data.compiler_prerequisite_edge_aggregate \
  --population-root "${sai_population_root}" \
  --judgments-root "${sai_judgments_root}" \
  --output-root "${sai_output_root}" \
  --logical-shards "${sai_logical_shards}"
