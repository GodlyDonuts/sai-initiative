#!/bin/zsh

set -euo pipefail

sai_root=${0:A:h:h}
cd "${sai_root}"

export PYTHONPATH=src

sai_quality=artifacts/sai_source_mechanical_quality_gate_publication_20260826_r3.json
sai_output=artifacts/sai_hermes_teacher_census_20260826_r1.json
sai_aggregates=(
  artifacts/sai_institutional_books_compiler_aggregate_20260825_r1.json
  artifacts/sai_arxiv_abstracts_audit_clean_aggregate_20260825_r2.json
  artifacts/sai_common_pile_audit_20260824_r1/aggregate.json
  artifacts/sai_common_pile_confirmation_20260824_r1/aggregate.json
  artifacts/sai_common_pile_pep_compiler_aggregate_20260825_r1.json
  artifacts/sai_common_pile_pilot_compiler_aggregate_20260825_r1.json
  artifacts/sai_frontier_source_audit_aggregate_20260825_r1.json
  artifacts/sai_frontier_source_audit_expansion_20260824_r1/aggregate.json
  artifacts/sai_pubmed_fulltext_audit_clean_20260825_r1/aggregate.json
  artifacts/sai_reservoir_audit_20260823_r2/aggregate.json
  artifacts/sai_reservoir_audit_weighted_20260824_r1/aggregate.json
  artifacts/sai_ultradata_math_tier_audit_clean_aggregate_20260825_r1.json
  artifacts/sai_opencoder_code_web_audit_20260826_r1/aggregate.json
)

for sai_wait_index in {1..69120}; do
  sai_missing=0
  for sai_path in "${sai_aggregates[@]}"; do
    if [[ ! -f "${sai_path}" ]]; then
      sai_missing=$((sai_missing + 1))
    fi
  done
  if [[ "${sai_missing}" = 0 ]]; then
    break
  fi
  sleep 10
done

for sai_path in "${sai_aggregates[@]}"; do
  if [[ ! -f "${sai_path}" ]]; then
    printf 'Hermès teacher aggregate is missing: %s\n' "${sai_path}" >&2
    exit 1
  fi
done
if [[ -e "${sai_output}" ]]; then
  echo 'Hermès teacher census already exists; refusing duplicate' >&2
  exit 1
fi

sai_command=(python3 -m sai.data.hermes_teacher_census --quality-publication "${sai_quality}")
for sai_path in "${sai_aggregates[@]}"; do
  sai_command+=(--aggregate "${sai_path}")
done
sai_command+=(--output "${sai_output}")
"${sai_command[@]}"
