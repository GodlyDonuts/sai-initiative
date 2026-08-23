#!/bin/zsh

set -euo pipefail

sai_root=${0:A:h:h}
cd "${sai_root}"

export PYTHONPATH=src

sai_population_root=artifacts/sai_grounded_bridge_development_population_20260825_r2
sai_judgments_root=${sai_population_root}/judgments
sai_output_root=artifacts/sai_grounded_bridge_development_aggregate_20260826_r1
sai_expected=512
sai_logical_shards=64

if [[ -e "${sai_output_root}" ]]; then
  echo 'grounded bridge aggregate already exists; refusing duplicate' >&2
  exit 1
fi

for sai_wait_index in {1..34560}; do
  if [[ -d "${sai_judgments_root}" ]]; then
    sai_receipts=$(find "${sai_judgments_root}" -maxdepth 1 -type f -name '*.grounded-bridge.json' | wc -l | tr -d ' ')
    sai_summaries=$(find "${sai_judgments_root}" -maxdepth 1 -type f -name 'shard_*.summary.json' | wc -l | tr -d ' ')
  else
    sai_receipts=0
    sai_summaries=0
  fi
  if [[ "${sai_receipts}" = "${sai_expected}" && "${sai_summaries}" = "${sai_logical_shards}" ]]; then
    python3 -m sai.data.grounded_bridge_aggregate \
      --population-root "${sai_population_root}" \
      --judgments-root "${sai_judgments_root}" \
      --output-root "${sai_output_root}" \
      --logical-shards "${sai_logical_shards}"
    exit 0
  fi
  sleep 10
done

echo 'grounded bridge synthesis did not complete within ninety-six hours' >&2
exit 1
