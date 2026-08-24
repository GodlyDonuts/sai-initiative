#!/bin/bash

set -euo pipefail

sai_candidates=artifacts/sai_finemath_semantic_audit_population_20260826_r1/candidates.jsonl
sai_output=artifacts/sai_finemath_semantic_audit_judgments_20260826_r1
sai_log_root=artifacts/sai_finemath_semantic_audit_logs_20260826_r1
sai_logical_shards=64

if [[ ! -f "${sai_candidates}" ]]; then
  echo "FineMath semantic candidate population is missing" >&2
  exit 1
fi

mkdir -p "${sai_output}" "${sai_log_root}"
export PYTHONPATH=src
export SAI_NOUS_LOOPBACK_KEY=local-proxy

run_segment() {
  local sai_first=$1
  local sai_last=$2
  local sai_shard
  local sai_summary
  local sai_attempt
  for sai_shard in $(seq "${sai_first}" "${sai_last}"); do
    sai_summary=$(printf '%s/shard_%05d.summary.json' "${sai_output}" "${sai_shard}")
    [[ -f "${sai_summary}" ]] && continue
    sai_attempt=0
    while true; do
      sai_attempt=$((sai_attempt + 1))
      if python3 -m sai.data.nous_label_worker \
        --candidates "${sai_candidates}" \
        --output-root "${sai_output}" \
        --model stealth/ox-alpha \
        --base-url http://127.0.0.1:8645/v1 \
        --api-key-env SAI_NOUS_LOOPBACK_KEY \
        --logical-shards "${sai_logical_shards}" \
        --shard-index "${sai_shard}" \
        --concurrency 3 \
        --timeout-seconds 600 \
        --maximum-attempts 5 \
        --judgments-per-candidate 3; then
        break
      fi
      printf '{"event":"finemath_semantic_retry","shard_index":%s,"attempt":%s}\n' \
        "${sai_shard}" "${sai_attempt}" >&2
      sleep 30
    done
  done
}

run_segment 0 15 >"${sai_log_root}/segment-00-15.log" 2>&1 &
sai_a=$!
run_segment 16 31 >"${sai_log_root}/segment-16-31.log" 2>&1 &
sai_b=$!
run_segment 32 47 >"${sai_log_root}/segment-32-47.log" 2>&1 &
sai_c=$!
run_segment 48 63 >"${sai_log_root}/segment-48-63.log" 2>&1 &
sai_d=$!

wait "${sai_a}"
wait "${sai_b}"
wait "${sai_c}"
wait "${sai_d}"

printf '{"event":"finemath_semantic_audit_complete","logical_shards":%s}\n' \
  "${sai_logical_shards}"
