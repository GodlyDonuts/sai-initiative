#!/bin/zsh

set -euo pipefail

sai_root=${0:A:h:h}
cd "${sai_root}"

export PYTHONPATH=src
export SAI_NOUS_LOOPBACK_KEY=local-proxy

sai_main_dependency=artifacts/sai_common_pile_pilot_compiler_aggregate_20260825_r1.json
sai_frontier_dependency=artifacts/sai_frontier_source_audit_aggregate_20260825_r1.json
sai_population_root=artifacts/sai_opencoder_code_web_audit_20260826_r1
sai_candidates=${sai_population_root}/candidates.jsonl
sai_judgments=${sai_population_root}/judgments
sai_aggregate=${sai_population_root}/aggregate.json
sai_decision=${sai_population_root}/source_decision.json

wait_for_file() {
  local sai_dependency_path=$1
  for sai_wait_index in {1..34560}; do
    if [[ -f "${sai_dependency_path}" ]]; then
      return 0
    fi
    sleep 10
  done
  printf 'dependency did not complete within ninety-six hours: %s\n' \
    "${sai_dependency_path}" >&2
  return 1
}

run_range() {
  local sai_first=$1
  local sai_last=$2
  for shard_index in $(seq "${sai_first}" "${sai_last}"); do
    local sai_summary
    sai_summary=$(printf '%s/shard_%05d.summary.json' "${sai_judgments}" "${shard_index}")
    if [[ -f "${sai_summary}" ]]; then
      continue
    fi
    local sai_attempt=0
    while true; do
      sai_attempt=$((sai_attempt + 1))
      if python3 -m sai.data.nous_compiler_worker \
        --candidates "${sai_candidates}" \
        --output-root "${sai_judgments}" \
        --model stealth/ox-alpha \
        --base-url http://127.0.0.1:8645/v1 \
        --api-key-env SAI_NOUS_LOOPBACK_KEY \
        --logical-shards 128 \
        --shard-index "${shard_index}" \
        --concurrency 4 \
        --timeout-seconds 600 \
        --maximum-attempts 5 \
        --stream-transport; then
        break
      fi
      printf '{"event":"opencoder_code_audit_retry","shard_index":%s,"attempt":%s}\n' \
        "${shard_index}" "${sai_attempt}"
      sleep 60
    done
  done
}

wait_for_file "${sai_main_dependency}"
wait_for_file "${sai_frontier_dependency}"

if [[ ! -f "${sai_candidates}" ]]; then
  echo 'OpenCoder code audit population is missing' >&2
  exit 1
fi

run_range 0 63 &
sai_first_pid=$!
run_range 64 127 &
sai_second_pid=$!
wait "${sai_first_pid}"
wait "${sai_second_pid}"

sai_receipts=$(find "${sai_judgments}" -maxdepth 1 -type f -name '*.compiler.json' | wc -l | tr -d ' ')
sai_summaries=$(find "${sai_judgments}" -maxdepth 1 -type f -name 'shard_*.summary.json' | wc -l | tr -d ' ')
if [[ "${sai_receipts}" != 2048 || "${sai_summaries}" != 128 ]]; then
  printf 'OpenCoder audit custody incomplete: receipts=%s summaries=%s\n' \
    "${sai_receipts}" "${sai_summaries}" >&2
  exit 1
fi

if [[ ! -e "${sai_aggregate}" ]]; then
  python3 -m sai.data.reservoir_audit_aggregate \
    --population-root "${sai_population_root}" \
    --judgments-root "${sai_judgments}" \
    --output "${sai_aggregate}"
fi
if [[ ! -e "${sai_decision}" ]]; then
  python3 -m sai.data.reservoir_audit_decision \
    --aggregate "${sai_aggregate}" \
    --output "${sai_decision}"
fi

printf '{"event":"opencoder_code_audit_complete","rows":2048}\n'
