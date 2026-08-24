#!/bin/bash

set -euo pipefail

sai_root=$(cd "$(dirname "$0")/.." && pwd)
cd "${sai_root}"
export PYTHONPATH=src

sai_decontamination=artifacts/sai_grounded_bridge_decontamination_20260826_r1
sai_anchor_population=artifacts/sai_grounded_bridge_development_population_20260825_r2
sai_output=artifacts/sai_grounded_bridge_curriculum_candidates_20260826_r1
sai_evidence=artifacts/sai_grounded_bridge_curriculum_candidates_evidence_20260826_r1.json

while [[ ! -f "${sai_decontamination}/receipt.json" ]]; do
  sleep 30
done

[[ ! -e "${sai_output}" && ! -e "${sai_evidence}" ]]
python3 -m sai.data.grounded_bridge_curriculum_candidates \
  --decontamination-root "${sai_decontamination}" \
  --anchor-population "${sai_anchor_population}/candidates.jsonl" \
  --anchor-population-receipt "${sai_anchor_population}/receipt.json" \
  --output-root "${sai_output}" \
  --durable-receipt "${sai_evidence}"
