#!/bin/bash

set -euo pipefail

sai_population=artifacts/sai_pdr_representation_population_20260826_r2
sai_generator_judgments=artifacts/sai_pdr_grounded_representation_generation_20260826_r2/judgments
sai_generated_aggregate=artifacts/sai_pdr_grounded_representation_aggregate_20260826_r2
sai_boundary=artifacts/sai_official_benchmark_boundary_index_20260824_r2
sai_decontamination=artifacts/sai_pdr_grounded_representation_decontamination_20260826_r2
sai_verification_population=artifacts/sai_pdr_grounded_representation_verification_population_20260826_r2
sai_verification_judgments=artifacts/sai_pdr_grounded_representation_verification_judgments_20260826_r2
sai_verification_aggregate=artifacts/sai_pdr_grounded_representation_verification_aggregate_20260826_r2
sai_independent_judgments=artifacts/sai_pdr_grounded_representation_independent_nemotron_20260826_r2/judgments
sai_cross_model_aggregate=artifacts/sai_pdr_grounded_representation_cross_model_aggregate_20260826_r2
sai_log_root=artifacts/sai_pdr_grounded_representation_verification_logs_20260826_r2
sai_logical_shards=128
sai_lanes=8

for sai_required in \
  "${sai_population}/receipt.json" \
  "${sai_population}/candidates.jsonl" \
  "${sai_boundary}/receipt.json"; do
  if [[ ! -f "${sai_required}" ]]; then
    echo "required PDR representation input is missing: ${sai_required}" >&2
    exit 1
  fi
done

export PYTHONPATH=src
export SAI_NOUS_LOOPBACK_KEY=local-proxy

while true; do
  sai_summaries=$(find "${sai_generator_judgments}" -maxdepth 1 -type f \
    -name 'shard_*.summary.json' | wc -l | tr -d ' ')
  if [[ "${sai_summaries}" = "${sai_logical_shards}" ]]; then
    break
  fi
  sleep 30
done

if [[ ! -e "${sai_generated_aggregate}" ]]; then
  python3 -m sai.data.grounded_representation_aggregate \
    --population-root "${sai_population}" \
    --judgments-root "${sai_generator_judgments}" \
    --output-root "${sai_generated_aggregate}"
fi

if [[ ! -e "${sai_decontamination}" ]]; then
  python3 -m sai.data.grounded_representation_decontamination \
    --aggregate-root "${sai_generated_aggregate}" \
    --boundary-index "${sai_boundary}" \
    --output-root "${sai_decontamination}"
fi

if [[ ! -e "${sai_verification_population}" ]]; then
  python3 -m sai.data.grounded_representation_verification_population \
    --source-population-root "${sai_population}" \
    --generator-judgments-root "${sai_generator_judgments}" \
    --generated-aggregate-root "${sai_generated_aggregate}" \
    --decontamination-root "${sai_decontamination}" \
    --output-root "${sai_verification_population}"
fi

mkdir -p "${sai_verification_judgments}" "${sai_log_root}"

run_lane() {
  local sai_lane=$1
  local sai_shard
  local sai_summary
  local sai_attempt
  for sai_shard in $(seq "${sai_lane}" "${sai_lanes}" 127); do
    sai_summary=$(printf '%s/shard_%05d.summary.json' \
      "${sai_verification_judgments}" "${sai_shard}")
    [[ -f "${sai_summary}" ]] && continue
    sai_attempt=0
    while true; do
      sai_attempt=$((sai_attempt + 1))
      if python3 -m sai.data.nous_grounded_representation_verifier \
        --candidates "${sai_verification_population}/candidates.jsonl" \
        --output-root "${sai_verification_judgments}" \
        --model stealth/ox-alpha \
        --base-url http://127.0.0.1:8645/v1 \
        --api-key-env SAI_NOUS_LOOPBACK_KEY \
        --logical-shards "${sai_logical_shards}" \
        --shard-index "${sai_shard}" \
        --concurrency 1 \
        --timeout-seconds 600 \
        --maximum-attempts 5 \
        --stream-transport; then
        break
      fi
      printf '{"event":"pdr_representation_verification_retry","shard_index":%s,"attempt":%s}\n' \
        "${sai_shard}" "${sai_attempt}" >&2
      sleep 30
    done
  done
}

sai_pids=()
for sai_lane in $(seq 0 $((sai_lanes - 1))); do
  run_lane "${sai_lane}" \
    >"${sai_log_root}/lane-$(printf '%02d' "${sai_lane}").log" 2>&1 &
  sai_pids+=("$!")
done
for sai_pid in "${sai_pids[@]}"; do
  wait "${sai_pid}"
done

if [[ ! -e "${sai_verification_aggregate}" ]]; then
  python3 -m sai.data.grounded_representation_verification_aggregate \
    --population-root "${sai_verification_population}" \
    --judgments-root "${sai_verification_judgments}" \
    --output-root "${sai_verification_aggregate}"
fi

set -a
source .env
set +a
mkdir -p "${sai_independent_judgments}"

run_independent_lane() {
  local sai_lane=$1
  local sai_shard
  local sai_summary
  local sai_attempt
  for sai_shard in $(seq "${sai_lane}" "${sai_lanes}" 127); do
    sai_summary=$(printf '%s/shard_%05d.summary.json' \
      "${sai_independent_judgments}" "${sai_shard}")
    [[ -f "${sai_summary}" ]] && continue
    sai_attempt=0
    while true; do
      sai_attempt=$((sai_attempt + 1))
      if python3 -m sai.data.nemotron_grounded_representation_verifier \
        --candidates "${sai_verification_population}/candidates.jsonl" \
        --output-root "${sai_independent_judgments}" \
        --logical-shards "${sai_logical_shards}" \
        --shard-index "${sai_shard}" \
        --concurrency 1 \
        --timeout-seconds 600 \
        --maximum-attempts 5 \
        --stream-transport; then
        break
      fi
      printf '{"event":"pdr_independent_representation_retry","shard_index":%s,"attempt":%s}\n' \
        "${sai_shard}" "${sai_attempt}" >&2
      sleep 30
    done
  done
}

sai_pids=()
for sai_lane in $(seq 0 $((sai_lanes - 1))); do
  run_independent_lane "${sai_lane}" \
    >"${sai_log_root}/independent-lane-$(printf '%02d' "${sai_lane}").log" 2>&1 &
  sai_pids+=("$!")
done
for sai_pid in "${sai_pids[@]}"; do
  wait "${sai_pid}"
done

if [[ ! -e "${sai_cross_model_aggregate}" ]]; then
  python3 -m sai.data.nemotron_grounded_representation_verification_aggregate \
    --population-root "${sai_verification_population}" \
    --same-family-judgments-root "${sai_verification_judgments}" \
    --independent-judgments-root "${sai_independent_judgments}" \
    --output-root "${sai_cross_model_aggregate}"
fi

printf '{"event":"pdr_grounded_representation_cross_model_pipeline_complete","logical_shards":%s}\n' \
  "${sai_logical_shards}"
