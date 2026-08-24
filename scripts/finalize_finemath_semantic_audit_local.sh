#!/bin/bash

set -euo pipefail

sai_population=artifacts/sai_finemath_semantic_audit_population_20260826_r1
sai_judgments=artifacts/sai_finemath_semantic_audit_judgments_20260826_r1
sai_output=artifacts/sai_finemath_semantic_audit_aggregate_20260826_r1
sai_logical_shards=64

while true; do
  sai_summaries=$(find "${sai_judgments}" -maxdepth 1 -type f \
    -name 'shard_*.summary.json' | wc -l | tr -d ' ')
  if [[ "${sai_summaries}" = "${sai_logical_shards}" ]]; then
    break
  fi
  sleep 30
done

if [[ ! -e "${sai_output}" ]]; then
  PYTHONPATH=src python3 -m sai.data.semantic_audit_aggregate \
    --candidates "${sai_population}/candidates.jsonl" \
    --population-receipt "${sai_population}/source-receipt.json" \
    --judgments-root "${sai_judgments}" \
    --output-root "${sai_output}" \
    --expected-model stealth/ox-alpha \
    --logical-shards "${sai_logical_shards}"
fi
