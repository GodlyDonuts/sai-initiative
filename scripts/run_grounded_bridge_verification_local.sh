#!/bin/zsh

set -euo pipefail

sai_root=${0:A:h:h}
cd "${sai_root}"

export PYTHONPATH=src
export SAI_NOUS_LOOPBACK_KEY=local-proxy

sai_source_population=artifacts/sai_grounded_bridge_development_population_20260825_r2
sai_generator_judgments=${sai_source_population}/judgments
sai_generated_aggregate=artifacts/sai_grounded_bridge_development_aggregate_20260826_r1
sai_verification_population=artifacts/sai_grounded_bridge_verification_population_20260826_r1
sai_judgments_root=${sai_verification_population}/judgments
sai_output_root=artifacts/sai_grounded_bridge_verification_aggregate_20260826_r1
sai_expected=512
sai_logical_shards=64

if [[ -e "${sai_output_root}" ]]; then
  echo 'grounded bridge verification aggregate already exists; refusing duplicate' >&2
  exit 1
fi

for sai_wait_index in {1..34560}; do
  if [[ -f "${sai_generated_aggregate}/receipt.json" ]]; then
    break
  fi
  sleep 10
done

if [[ ! -f "${sai_generated_aggregate}/receipt.json" ]]; then
  echo 'grounded bridge aggregate did not complete within ninety-six hours' >&2
  exit 1
fi

if [[ ! -e "${sai_verification_population}" ]]; then
  python3 -m sai.data.grounded_bridge_verification_population \
    --source-population-root "${sai_source_population}" \
    --generator-judgments-root "${sai_generator_judgments}" \
    --generated-aggregate-root "${sai_generated_aggregate}" \
    --output-root "${sai_verification_population}"
fi

if [[ ! -f "${sai_verification_population}/receipt.json" ]]; then
  echo 'grounded bridge verification population is incomplete' >&2
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
    if python3 -m sai.data.nous_grounded_bridge_verifier \
      --candidates "${sai_verification_population}/candidates.jsonl" \
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
    printf '{"event":"grounded_bridge_verification_retry","shard_index":%s,"attempt":%s}\n' \
      "${shard_index}" "${attempt}"
    sleep 60
  done
done

sai_receipts=$(find "${sai_judgments_root}" -maxdepth 1 -type f -name '*.grounded-bridge-verification.json' | wc -l | tr -d ' ')
sai_summaries=$(find "${sai_judgments_root}" -maxdepth 1 -type f -name 'shard_*.summary.json' | wc -l | tr -d ' ')
if [[ "${sai_receipts}" != "${sai_expected}" || "${sai_summaries}" != "${sai_logical_shards}" ]]; then
  echo 'grounded bridge verification custody is incomplete' >&2
  exit 1
fi

python3 -m sai.data.grounded_bridge_verification_aggregate \
  --population-root "${sai_verification_population}" \
  --judgments-root "${sai_judgments_root}" \
  --output-root "${sai_output_root}" \
  --logical-shards "${sai_logical_shards}"
