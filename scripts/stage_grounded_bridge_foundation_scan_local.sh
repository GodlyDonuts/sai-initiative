#!/bin/bash

set -euo pipefail

sai_root=$(cd "$(dirname "$0")/.." && pwd)
cd "${sai_root}"
export PYTHONPATH=src

sai_commit=$(git rev-parse HEAD)
sai_short=${sai_commit:0:7}
sai_candidates=artifacts/sai_grounded_bridge_curriculum_candidates_20260826_r1
sai_query=artifacts/sai_grounded_bridge_foundation_query_20260826_r1
sai_candidate_evidence=artifacts/sai_grounded_bridge_curriculum_candidates_evidence_20260826_r1.json
sai_query_evidence=artifacts/sai_grounded_bridge_foundation_query_evidence_20260826_r1.json
sai_remote_root=/lustre/fs1/home/sa305415/sai_data_sources
sai_remote_candidates=${sai_remote_root}/grounded-bridge-curriculum-candidates-20260826-r1
sai_remote_query=${sai_remote_root}/grounded-bridge-foundation-query-20260826-r1
sai_remote_evidence=${sai_remote_root}/sai_evidence/grounded-bridge-foundation-scan/20260826-r1
sai_runtime=/lustre/fs1/home/sa305415/sai-initiative-${sai_short}

while [[ ! -f "${sai_query}/receipt.json" ]]; do
  sleep 30
done

python3 -c \
  'from pathlib import Path; from sai.data.grounded_bridge_foundation_scan import QueryBoundary; QueryBoundary(Path("artifacts/sai_grounded_bridge_foundation_query_20260826_r1"))'

[[ -d "${sai_candidates}" && -f "${sai_candidate_evidence}" && -f "${sai_query_evidence}" ]]

sai_candidate_stage=${sai_remote_candidates}.partial.${sai_commit}
sai_query_stage=${sai_remote_query}.partial.${sai_commit}
ssh -o BatchMode=yes stokes \
  "test ! -e '${sai_remote_candidates}' && test ! -e '${sai_remote_query}' && test ! -e '${sai_candidate_stage}' && test ! -e '${sai_query_stage}'"

rsync -a "${sai_candidates}/" "stokes:${sai_candidate_stage}/"
rsync -a "${sai_query}/" "stokes:${sai_query_stage}/"

sai_query_receipt=$(python3 -c \
  'import json; print(json.load(open("artifacts/sai_grounded_bridge_foundation_query_20260826_r1/receipt.json"))["receipt_sha256"])')
sai_candidate_receipt=$(python3 -c \
  'import json; print(json.load(open("artifacts/sai_grounded_bridge_curriculum_candidates_20260826_r1/receipt.json"))["receipt_sha256"])')

ssh -o BatchMode=yes stokes bash -s -- \
  "${sai_commit}" \
  "${sai_runtime}" \
  "${sai_candidate_stage}" \
  "${sai_remote_candidates}" \
  "${sai_query_stage}" \
  "${sai_remote_query}" \
  "${sai_remote_evidence}" \
  "${sai_candidate_receipt}" \
  "${sai_query_receipt}" <<'REMOTE'
set -euo pipefail

sai_commit=$1
sai_runtime=$2
sai_candidate_stage=$3
sai_remote_candidates=$4
sai_query_stage=$5
sai_remote_query=$6
sai_remote_evidence=$7
sai_candidate_receipt=$8
sai_query_receipt=$9
sai_repo=/lustre/fs1/home/sa305415/sai-initiative

[[ ! -e "${sai_remote_candidates}" && ! -e "${sai_remote_query}" ]]
mv "${sai_candidate_stage}" "${sai_remote_candidates}"
mv "${sai_query_stage}" "${sai_remote_query}"

if [[ ! -e "${sai_runtime}" ]]; then
  git clone --quiet --no-hardlinks "${sai_repo}" "${sai_runtime}"
  git -C "${sai_runtime}" checkout --quiet --detach "${sai_commit}"
  [[ -z "$(git -C "${sai_runtime}" status --porcelain)" ]]
  [[ "$(git -C "${sai_runtime}" rev-parse HEAD)" == "${sai_commit}" ]]
  [[ -z "$(find "${sai_runtime}" -type l -print -quit)" ]]
  chmod -R a-w "${sai_runtime}"
fi

[[ "$(git -C "${sai_runtime}" rev-parse HEAD)" == "${sai_commit}" ]]
[[ -z "$(find "${sai_runtime}" -type l -print -quit)" ]]
[[ -z "$(find "${sai_runtime}" -type f -perm /222 -print -quit)" ]]
[[ -z "$(squeue -u sa305415 -h -o '%j' | grep -E '^sai-(pleias|book)-bridge-scan$|^sai-bridge-scan-aggregate$' || true)" ]]

sai_pleias_job=$(sbatch --parsable \
  --dependency=afterok:818642 \
  --export="ALL,SAI_RUNTIME_ROOT=${sai_runtime}" \
  "${sai_runtime}/scripts/scan_pleias_grounded_bridge_foundation_stokes.sbatch")
sai_book_job=$(sbatch --parsable \
  --dependency=afterok:818644 \
  --export="ALL,SAI_RUNTIME_ROOT=${sai_runtime}" \
  "${sai_runtime}/scripts/scan_institutional_books_grounded_bridge_foundation_stokes.sbatch")
sai_aggregate_job=$(sbatch --parsable \
  --dependency="afterok:${sai_pleias_job}:${sai_book_job}" \
  --export="ALL,SAI_RUNTIME_ROOT=${sai_runtime}" \
  "${sai_runtime}/scripts/aggregate_grounded_bridge_foundation_scan_stokes.sbatch")

mkdir -p "${sai_remote_evidence}"
SAI_COMMIT="${sai_commit}" \
SAI_RUNTIME="${sai_runtime}" \
SAI_CANDIDATE_RECEIPT="${sai_candidate_receipt}" \
SAI_QUERY_RECEIPT="${sai_query_receipt}" \
SAI_PLEIAS_JOB="${sai_pleias_job}" \
SAI_BOOK_JOB="${sai_book_job}" \
SAI_AGGREGATE_JOB="${sai_aggregate_job}" \
SAI_EVIDENCE="${sai_remote_evidence}/launch-receipt.json" \
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

payload = {
    "schema": "sai-grounded-bridge-foundation-scan-launch-v1",
    "commit": os.environ["SAI_COMMIT"],
    "immutable_runtime": os.environ["SAI_RUNTIME"],
    "candidate_receipt_sha256": os.environ["SAI_CANDIDATE_RECEIPT"],
    "query_receipt_sha256": os.environ["SAI_QUERY_RECEIPT"],
    "foundation_dependencies": {
        "pleias_final_aggregate_job": "818642",
        "institutional_books_final_aggregate_job": "818644",
    },
    "jobs": {
        "pleias_scan": os.environ["SAI_PLEIAS_JOB"],
        "institutional_books_scan": os.environ["SAI_BOOK_JOB"],
        "aggregate": os.environ["SAI_AGGREGATE_JOB"],
    },
    "source_text_persisted_in_launch_receipt": False,
    "training_ready": False,
    "four_b_training_authorized": False,
}
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
path = Path(os.environ["SAI_EVIDENCE"])
path.open("x").write(json.dumps(payload, sort_keys=True) + "\n")
print(json.dumps(payload["jobs"], sort_keys=True))
PY
REMOTE
