#!/bin/zsh

set -euo pipefail

sai_root=${0:A:h:h}
cd "${sai_root}"

sai_remote=stokes
sai_remote_root=/lustre/fs1/home/sa305415/sai_data_sources
sai_terminal_receipt=${sai_remote_root}/institutional-books-full-decontamination-20260826-r1/receipt.json

for sai_wait_index in {1..34560}; do
  if ssh -o BatchMode=yes "${sai_remote}" "test -f '${sai_terminal_receipt}'"; then
    break
  fi
  if [[ "${sai_wait_index}" = 34560 ]]; then
    printf 'book quality graph did not close within ninety-six hours\n' >&2
    exit 1
  fi
  sleep 10
done

copy_safe_file() {
  local sai_remote_path=$1
  local sai_local_path=$2
  mkdir -p "${sai_local_path:h}"
  scp -q "${sai_remote}:${sai_remote_path}" "${sai_local_path}"
}

copy_safe_file \
  "${sai_remote_root}/institutional-books-strict-english-materialized-20260826-r1/aggregate.json" \
  artifacts/sai_institutional_books_materialized_aggregate_20260826_r1.json
copy_safe_file \
  "${sai_remote_root}/institutional-books-mechanical-gate-20260826-r1/aggregate.json" \
  artifacts/sai_institutional_books_mechanical_gate_aggregate_20260826_r1.json
copy_safe_file \
  "${sai_remote_root}/institutional-books-mechanical-filtered-20260826-r1/aggregate.json" \
  artifacts/sai_institutional_books_mechanical_filter_aggregate_20260826_r1.json
copy_safe_file \
  "${sai_remote_root}/institutional-books-semantic-population-20260826-r1/receipt.json" \
  artifacts/sai_institutional_books_semantic_population_20260826_r1/receipt.json
copy_safe_file \
  "${sai_remote_root}/institutional-books-semantic-judgments-20260826-r1/aggregate.json" \
  artifacts/sai_institutional_books_semantic_aggregate_20260826_r1.json
copy_safe_file \
  "${sai_remote_root}/institutional-books-semantic-decisions-20260826-r1/receipt.json" \
  artifacts/sai_institutional_books_semantic_decisions_20260826_r1/receipt.json
copy_safe_file \
  "${sai_remote_root}/institutional-books-semantic-decisions-20260826-r1/decisions.jsonl" \
  artifacts/sai_institutional_books_semantic_decisions_20260826_r1/decisions.jsonl
copy_safe_file \
  "${sai_remote_root}/institutional-books-independent-population-20260826-r1/receipt.json" \
  artifacts/sai_institutional_books_independent_population_20260826_r1/receipt.json
copy_safe_file \
  "${sai_remote_root}/institutional-books-independent-nemotron-20260826-r1/aggregate.json" \
  artifacts/sai_institutional_books_independent_nemotron_aggregate_20260826_r1.json
copy_safe_file \
  "${sai_remote_root}/institutional-books-independent-agreement-20260826-r1/receipt.json" \
  artifacts/sai_institutional_books_independent_agreement_20260826_r1/receipt.json
copy_safe_file \
  "${sai_remote_root}/institutional-books-independent-agreement-20260826-r1/agreement.jsonl" \
  artifacts/sai_institutional_books_independent_agreement_20260826_r1/agreement.jsonl
copy_safe_file \
  "${sai_remote_root}/institutional-books-full-decontamination-20260826-r1/receipt.json" \
  artifacts/sai_institutional_books_full_decontamination_20260826_r1/receipt.json
copy_safe_file \
  "${sai_remote_root}/institutional-books-full-decontamination-20260826-r1/decisions.jsonl" \
  artifacts/sai_institutional_books_full_decontamination_20260826_r1/decisions.jsonl
copy_safe_file \
  "${sai_remote_root}/institutional-books-full-decontamination-20260826-r1/benchmark_disjoint_books.jsonl" \
  artifacts/sai_institutional_books_full_decontamination_20260826_r1/benchmark_disjoint_books.jsonl

python3 - <<'PY'
import json
from pathlib import Path

roots = [
    Path("artifacts/sai_institutional_books_semantic_population_20260826_r1"),
    Path("artifacts/sai_institutional_books_semantic_decisions_20260826_r1"),
    Path("artifacts/sai_institutional_books_independent_population_20260826_r1"),
    Path("artifacts/sai_institutional_books_independent_agreement_20260826_r1"),
    Path("artifacts/sai_institutional_books_full_decontamination_20260826_r1"),
]
for root in roots:
    receipt = json.loads((root / "receipt.json").read_text())
    if receipt.get("training_ready") is not False:
        raise SystemExit(f"unsafe copied receipt: {root}")
print('{"event":"institutional_books_quality_evidence_copied","source_text_copied":false}')
PY
