#!/bin/zsh

set -euo pipefail

sai_root=${0:A:h:h}
cd "${sai_root}"

export PYTHONPATH=src
export SAI_NOUS_LOOPBACK_KEY=local-proxy

sai_main_dependency=artifacts/sai_common_pile_pilot_compiler_aggregate_20260825_r1.json
sai_frontier_dependency=artifacts/sai_frontier_source_audit_aggregate_20260825_r1.json
sai_weighted_root=artifacts/sai_reservoir_audit_weighted_20260824_r1
sai_pubmed_root=artifacts/sai_pubmed_fulltext_audit_clean_20260825_r1

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

run_compiler_range() {
  local sai_candidates=$1
  local sai_judgments=$2
  local sai_logical_shards=$3
  local sai_first=$4
  local sai_last=$5
  local sai_event=$6
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
        --logical-shards "${sai_logical_shards}" \
        --shard-index "${shard_index}" \
        --concurrency 2 \
        --timeout-seconds 600 \
        --maximum-attempts 5 \
        --stream-transport; then
        break
      fi
      printf '{"event":"%s_retry","shard_index":%s,"attempt":%s}\n' \
        "${sai_event}" "${shard_index}" "${sai_attempt}"
      sleep 60
    done
  done
}

seal_population() {
  local sai_population_root=$1
  local sai_expected_receipts=$2
  local sai_expected_summaries=$3
  local sai_aggregate=$4
  local sai_decision=$5
  local sai_receipts
  local sai_summaries
  sai_receipts=$(find "${sai_population_root}/judgments" -maxdepth 1 -type f -name '*.compiler.json' | wc -l | tr -d ' ')
  sai_summaries=$(find "${sai_population_root}/judgments" -maxdepth 1 -type f -name 'shard_*.summary.json' | wc -l | tr -d ' ')
  if [[ "${sai_receipts}" != "${sai_expected_receipts}" || "${sai_summaries}" != "${sai_expected_summaries}" ]]; then
    printf 'teacher audit custody incomplete: %s receipts=%s summaries=%s\n' \
      "${sai_population_root}" "${sai_receipts}" "${sai_summaries}" >&2
    return 1
  fi
  if [[ ! -e "${sai_aggregate}" ]]; then
    python3 -m sai.data.reservoir_audit_aggregate \
      --population-root "${sai_population_root}" \
      --judgments-root "${sai_population_root}/judgments" \
      --output "${sai_aggregate}"
  fi
  if [[ ! -e "${sai_decision}" ]]; then
    python3 -m sai.data.reservoir_audit_decision \
      --aggregate "${sai_aggregate}" \
      --output "${sai_decision}"
  fi
}

wait_for_file "${sai_main_dependency}"
wait_for_file "${sai_frontier_dependency}"

run_compiler_range \
  "${sai_weighted_root}/candidates.jsonl" \
  "${sai_weighted_root}/judgments" 64 18 40 weighted_teacher &
sai_weighted_a_pid=$!
run_compiler_range \
  "${sai_weighted_root}/candidates.jsonl" \
  "${sai_weighted_root}/judgments" 64 41 63 weighted_teacher &
sai_weighted_b_pid=$!
run_compiler_range \
  "${sai_pubmed_root}/candidates.jsonl" \
  "${sai_pubmed_root}/judgments" 128 0 63 pubmed_teacher &
sai_pubmed_a_pid=$!
run_compiler_range \
  "${sai_pubmed_root}/candidates.jsonl" \
  "${sai_pubmed_root}/judgments" 128 64 127 pubmed_teacher &
sai_pubmed_b_pid=$!

wait "${sai_weighted_a_pid}"
wait "${sai_weighted_b_pid}"
seal_population \
  "${sai_weighted_root}" 1024 64 \
  "${sai_weighted_root}/aggregate.json" \
  "${sai_weighted_root}/source_decision.json" &
sai_weighted_seal_pid=$!

wait "${sai_pubmed_a_pid}"
wait "${sai_pubmed_b_pid}"
seal_population \
  "${sai_pubmed_root}" 1007 128 \
  "${sai_pubmed_root}/aggregate.json" \
  "${sai_pubmed_root}/source_decision.json" &
sai_pubmed_seal_pid=$!

wait "${sai_weighted_seal_pid}"
wait "${sai_pubmed_seal_pid}"
printf '{"event":"broad_hermes_teacher_audits_complete","rows":2031}\n'
