#!/bin/zsh

set -euo pipefail

sai_root=${0:A:h:h}
cd "${sai_root}"

set -a
source .env
set +a
export PYTHONPATH=src

if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
  echo 'NVIDIA_API_KEY is required' >&2
  exit 1
fi

sai_population=artifacts/sai_grounded_bridge_verification_population_20260826_r1
sai_candidates=${sai_population}/candidates.jsonl
sai_same_family=artifacts/sai_grounded_bridge_verification_aggregate_20260826_r1
sai_judgments=artifacts/sai_grounded_bridge_independent_nemotron_20260826_r1/judgments
sai_aggregate=artifacts/sai_grounded_bridge_independent_nemotron_aggregate_20260826_r1
sai_decontamination=artifacts/sai_grounded_bridge_decontamination_20260826_r1
sai_boundary=artifacts/sai_official_benchmark_boundary_index_20260824_r2
sai_logical_shards=64
sai_lanes=12
sai_expected=512

if [[ ! -f "${sai_candidates}" ]]; then
  echo 'grounded bridge verification population is missing' >&2
  exit 1
fi
mkdir -p "${sai_judgments}"

run_lane() {
  local lane=$1
  local shard_index
  for shard_index in $(seq "${lane}" "${sai_lanes}" $((sai_logical_shards - 1))); do
    local summary
    summary=$(printf '%s/shard_%05d.summary.json' "${sai_judgments}" "${shard_index}")
    if [[ -f "${summary}" ]]; then
      continue
    fi
    while ! python3 -m sai.data.nemotron_grounded_bridge_verifier \
      --candidates "${sai_candidates}" \
      --output-root "${sai_judgments}" \
      --model nvidia/nemotron-3-ultra-550b-a55b \
      --base-url https://integrate.api.nvidia.com/v1 \
      --api-key-env NVIDIA_API_KEY \
      --logical-shards "${sai_logical_shards}" \
      --shard-index "${shard_index}" \
      --concurrency 1 \
      --timeout-seconds 600 \
      --maximum-attempts 5 \
      --stream-transport; do
      printf '{"event":"nemotron_bridge_retry","lane":%s,"shard_index":%s}\n' \
        "${lane}" "${shard_index}"
      sleep 60
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

sai_receipts=$(find "${sai_judgments}" -maxdepth 1 -type f \
  -name '*.grounded-bridge-independent-model-family-verification.json' \
  | wc -l | tr -d ' ')
sai_summaries=$(find "${sai_judgments}" -maxdepth 1 -type f \
  -name 'shard_*.summary.json' | wc -l | tr -d ' ')
if [[ "${sai_receipts}" != "${sai_expected}" || \
      "${sai_summaries}" != "${sai_logical_shards}" ]]; then
  echo 'independent bridge verification custody is incomplete' >&2
  exit 1
fi

if [[ ! -e "${sai_aggregate}" ]]; then
  python3 -m sai.data.nemotron_grounded_bridge_verification_aggregate \
    --population-root "${sai_population}" \
    --same-family-aggregate-root "${sai_same_family}" \
    --judgments-root "${sai_judgments}" \
    --output-root "${sai_aggregate}" \
    --logical-shards "${sai_logical_shards}"
fi

if [[ ! -e "${sai_decontamination}" ]]; then
  python3 -m sai.data.grounded_bridge_decontamination \
    --aggregate-root "${sai_aggregate}" \
    --boundary-index "${sai_boundary}" \
    --output-root "${sai_decontamination}"
fi

printf '{"event":"nemotron_grounded_bridge_verification_complete","receipts":%s,"summaries":%s}\n' \
  "${sai_receipts}" "${sai_summaries}"
