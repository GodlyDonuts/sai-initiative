#!/bin/zsh

set -euo pipefail

sai_root=${0:A:h:h}
cd "${sai_root}"

export PYTHONPATH=src
export SAI_NOUS_LOOPBACK_KEY=local-proxy

sai_frontier_aggregate=artifacts/sai_frontier_source_audit_aggregate_20260825_r1.json
sai_bridge_candidates=artifacts/sai_grounded_bridge_development_population_20260825_r2/candidates.jsonl
sai_bridge_judgments=artifacts/sai_grounded_bridge_development_population_20260825_r2/judgments
sai_book_candidates=artifacts/institutional_books_pilot_20260823_r2/candidates.jsonl
sai_book_judgments=artifacts/institutional_books_pilot_20260823_r2/judgments_v2
sai_prerequisite_candidates=artifacts/sai_compiler_prerequisite_edge_population_20260826_r1/candidates.jsonl
sai_prerequisite_judgments=artifacts/sai_compiler_prerequisite_edge_population_20260826_r1/judgments
sai_bridge_verifier_candidates=artifacts/sai_grounded_bridge_verification_population_20260826_r1/candidates.jsonl
sai_bridge_verifier_judgments=artifacts/sai_grounded_bridge_verification_population_20260826_r1/judgments

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

run_bridge_range() {
  local first=$1
  local last=$2
  for shard_index in $(seq "${first}" "${last}"); do
    local summary
    summary=$(printf '%s/shard_%05d.summary.json' "${sai_bridge_judgments}" "${shard_index}")
    if [[ -f "${summary}" ]]; then
      continue
    fi
    while ! python3 -m sai.data.nous_grounded_bridge_worker \
      --candidates "${sai_bridge_candidates}" \
      --output-root "${sai_bridge_judgments}" \
      --model stealth/ox-alpha \
      --base-url http://127.0.0.1:8645/v1 \
      --api-key-env SAI_NOUS_LOOPBACK_KEY \
      --logical-shards 64 \
      --shard-index "${shard_index}" \
      --concurrency 4 \
      --timeout-seconds 600 \
      --maximum-attempts 5 \
      --stream-transport; do
      sleep 60
    done
  done
}

run_book_lane() {
  local lane=$1
  local shards
  shards=$(python3 - "${lane}" <<'PY'
import json
import sys
from pathlib import Path

lane = int(sys.argv[1])
values = set()
with Path("artifacts/institutional_books_pilot_20260823_r2/candidates.jsonl").open() as handle:
    for line in handle:
        values.add(int(json.loads(line)["candidate_identity_sha256"], 16) % 10_000)
print(" ".join(str(value) for index, value in enumerate(sorted(values)) if index % 2 == lane))
PY
)
  for shard_index in $=shards; do
    local summary
    summary=$(printf '%s/shard_%05d.summary.json' "${sai_book_judgments}" "${shard_index}")
    if [[ -f "${summary}" ]]; then
      continue
    fi
    while ! python3 -m sai.data.nous_book_compiler_worker \
      --candidates "${sai_book_candidates}" \
      --output-root "${sai_book_judgments}" \
      --model stealth/ox-alpha \
      --base-url http://127.0.0.1:8645/v1 \
      --api-key-env SAI_NOUS_LOOPBACK_KEY \
      --logical-shards 10000 \
      --shard-index "${shard_index}" \
      --concurrency 2 \
      --timeout-seconds 600 \
      --maximum-attempts 5 \
      --stream-transport; do
      sleep 60
    done
  done
}

run_prerequisite_range() {
  local first=$1
  local last=$2
  wait_for_file "${sai_prerequisite_candidates}"
  for shard_index in $(seq "${first}" "${last}"); do
    local summary
    summary=$(printf '%s/shard_%05d.summary.json' "${sai_prerequisite_judgments}" "${shard_index}")
    if [[ -f "${summary}" ]]; then
      continue
    fi
    while ! python3 -m sai.data.nous_compiler_prerequisite_edge_verifier \
      --candidates "${sai_prerequisite_candidates}" \
      --output-root "${sai_prerequisite_judgments}" \
      --model stealth/ox-alpha \
      --base-url http://127.0.0.1:8645/v1 \
      --api-key-env SAI_NOUS_LOOPBACK_KEY \
      --logical-shards 64 \
      --shard-index "${shard_index}" \
      --concurrency 4 \
      --timeout-seconds 600 \
      --maximum-attempts 5 \
      --stream-transport; do
      sleep 60
    done
  done
}

run_bridge_verifier_range() {
  local first=$1
  local last=$2
  wait_for_file "${sai_bridge_verifier_candidates}"
  for shard_index in $(seq "${first}" "${last}"); do
    local summary
    summary=$(printf '%s/shard_%05d.summary.json' "${sai_bridge_verifier_judgments}" "${shard_index}")
    if [[ -f "${summary}" ]]; then
      continue
    fi
    while ! python3 -m sai.data.nous_grounded_bridge_verifier \
      --candidates "${sai_bridge_verifier_candidates}" \
      --output-root "${sai_bridge_verifier_judgments}" \
      --model stealth/ox-alpha \
      --base-url http://127.0.0.1:8645/v1 \
      --api-key-env SAI_NOUS_LOOPBACK_KEY \
      --logical-shards 64 \
      --shard-index "${shard_index}" \
      --concurrency 4 \
      --timeout-seconds 600 \
      --maximum-attempts 5 \
      --stream-transport; do
      sleep 60
    done
  done
}

wait_for_file "${sai_frontier_aggregate}"

run_bridge_range 21 42 &
sai_bridge_a_pid=$!
run_bridge_range 43 63 &
sai_bridge_b_pid=$!
run_book_lane 0 &
sai_book_a_pid=$!
run_book_lane 1 &
sai_book_b_pid=$!
run_prerequisite_range 21 42 &
sai_prerequisite_a_pid=$!
run_prerequisite_range 43 63 &
sai_prerequisite_b_pid=$!
run_bridge_verifier_range 21 42 &
sai_verifier_a_pid=$!
run_bridge_verifier_range 43 63 &
sai_verifier_b_pid=$!

wait "${sai_bridge_a_pid}"
wait "${sai_bridge_b_pid}"
wait "${sai_book_a_pid}"
wait "${sai_book_b_pid}"
wait "${sai_prerequisite_a_pid}"
wait "${sai_prerequisite_b_pid}"
wait "${sai_verifier_a_pid}"
wait "${sai_verifier_b_pid}"

printf '{"event":"hermes_capacity_fanout_complete"}\n'
