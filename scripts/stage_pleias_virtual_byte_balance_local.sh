#!/bin/bash

set -euo pipefail

sai_root=$(cd "$(dirname "$0")/.." && pwd)
cd "${sai_root}"

git diff --quiet
git diff --cached --quiet
sai_commit=$(git rev-parse HEAD)
sai_short=${sai_commit:0:7}
sai_runtime=/lustre/fs1/home/sa305415/sai-initiative-${sai_short}

ssh -o BatchMode=yes stokes bash -s -- "${sai_commit}" "${sai_runtime}" <<'REMOTE'
set -euo pipefail

sai_commit=$1
sai_runtime=$2
sai_repo=/lustre/fs1/home/sa305415/sai-initiative
sai_root=/lustre/fs1/home/sa305415/sai_data_sources/pleias-virtual-byte-balance-20260826-r1

[[ "$(git -C "${sai_repo}" rev-parse HEAD)" == "${sai_commit}" ]]
[[ -z "$(git -C "${sai_repo}" status --porcelain)" ]]
[[ ! -e "${sai_root}" ]]
[[ -z "$(squeue -u sa305415 -h -o '%j' | grep -E '^sai-pleias-byte-(allocate|select|aggregate)$' || true)" ]]

if [[ ! -e "${sai_runtime}" ]]; then
  git -C "${sai_repo}" worktree add --quiet --detach "${sai_runtime}" "${sai_commit}"
  [[ -z "$(git -C "${sai_runtime}" status --porcelain)" ]]
  [[ -z "$(find "${sai_runtime}" -type l -print -quit)" ]]
  chmod -R a-w "${sai_runtime}"
fi
[[ "$(git -C "${sai_runtime}" rev-parse HEAD)" == "${sai_commit}" ]]

mkdir -p "${sai_root}"
sai_allocate=$(sbatch --parsable \
  --dependency=afterok:818642:818644 \
  --export="ALL,SAI_RUNTIME_ROOT=${sai_runtime}" \
  "${sai_runtime}/scripts/allocate_pleias_virtual_byte_balance_stokes.sbatch")
sai_select=$(sbatch --parsable \
  --dependency="afterok:${sai_allocate}" \
  --export="ALL,SAI_RUNTIME_ROOT=${sai_runtime}" \
  "${sai_runtime}/scripts/select_pleias_virtual_byte_balance_stokes.sbatch")
sai_aggregate=$(sbatch --parsable \
  --dependency="afterok:${sai_select}" \
  --export="ALL,SAI_RUNTIME_ROOT=${sai_runtime}" \
  "${sai_runtime}/scripts/aggregate_pleias_virtual_byte_balance_stokes.sbatch")

# These consumers are dependency-pending and now read the byte-balanced view.
scontrol update JobId=818645 Dependency="afterok:${sai_aggregate}"
scontrol update JobId=818720 Dependency="afterok:${sai_aggregate}"
scontrol update JobId=818732 Dependency="afterok:${sai_aggregate}"

SAI_COMMIT="${sai_commit}" \
SAI_RUNTIME="${sai_runtime}" \
SAI_ALLOCATE="${sai_allocate}" \
SAI_SELECT="${sai_select}" \
SAI_AGGREGATE="${sai_aggregate}" \
SAI_OUTPUT="${sai_root}/launch-receipt.json" \
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

payload = {
    "schema": "sai-pleias-virtual-byte-balance-launch-v1",
    "runtime_commit": os.environ["SAI_COMMIT"],
    "immutable_runtime": os.environ["SAI_RUNTIME"],
    "dependencies": {
        "pleias_final_aggregate_job": "818642",
        "institutional_books_final_aggregate_job": "818644",
    },
    "jobs": {
        "allocation": os.environ["SAI_ALLOCATE"],
        "shard_selection": os.environ["SAI_SELECT"],
        "aggregate": os.environ["SAI_AGGREGATE"],
    },
    "rewired_pending_consumers": ["818645", "818720", "818732"],
    "source_text_persisted": False,
    "training_ready": False,
    "four_b_training_authorized": False,
}
payload["receipt_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
Path(os.environ["SAI_OUTPUT"]).open("x").write(
    json.dumps(payload, sort_keys=True) + "\n"
)
print(json.dumps(payload["jobs"], sort_keys=True))
PY
REMOTE
