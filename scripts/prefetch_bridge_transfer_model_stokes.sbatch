#!/usr/bin/env bash
#SBATCH --job-name=sai-prefetch-bridge-model
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --exclude=ec65
#SBATCH --no-requeue
#SBATCH --output=/lustre/fs1/home/sa305415/sai_data_sources/logs/prefetch-bridge-model-%j.out
#SBATCH --error=/lustre/fs1/home/sa305415/sai_data_sources/logs/prefetch-bridge-model-%j.err

set -euo pipefail

: "${SAI_RUNTIME_ROOT:?immutable Sai runtime root is required}"
: "${SAI_RUNTIME_COMMIT:?immutable Sai runtime commit is required}"
[[ "$(git -C "${SAI_RUNTIME_ROOT}" rev-parse HEAD)" == "${SAI_RUNTIME_COMMIT}" ]]
[[ -z "$(git -C "${SAI_RUNTIME_ROOT}" status --porcelain)" ]]

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${SAI_RUNTIME_ROOT}/src"
export HF_HOME=/lustre/fs1/home/sa305415/sai_data_sources/hf-cache/bridge-transfer-smollm2-360m

source /lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603/bin/activate
set -a
source /lustre/fs1/home/sa305415/sai-initiative/.env
set +a

sai_output=/lustre/fs1/home/sa305415/sai_data_sources/sai_evidence/bridge-transfer-model-prefetch/20260826-r1/receipt.json
[[ ! -e "${sai_output}" ]]
mkdir -p "$(dirname "${sai_output}")"

python - "${sai_output}" "${SAI_RUNTIME_COMMIT}" <<'PY'
import json
import os
import sys
from pathlib import Path

import huggingface_hub
from huggingface_hub import snapshot_download

from sai.data.bridge_transfer_screen import MODEL_REPOSITORY, MODEL_REVISION
from sai.data.token_stream import canonical_sha256, sha256_file

destination, runtime_commit = sys.argv[1:]
allowed = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]
token = os.environ.get("HF_TOKEN", "")
if not token:
    raise SystemExit("HF token is unavailable")
snapshot = Path(
    snapshot_download(
        MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        token=token,
        allow_patterns=allowed,
    )
)
files = []
for name in allowed:
    path = snapshot / name
    if not path.is_file():
        raise SystemExit(f"model snapshot member is missing: {name}")
    files.append(
        {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    )
if {path.name for path in snapshot.iterdir() if path.is_file()} != set(allowed):
    raise SystemExit("model snapshot file geometry differs")
payload = {
    "schema": "sai-bridge-transfer-model-prefetch-v1",
    "status": "complete_bridge_transfer_model_prefetch",
    "repository": MODEL_REPOSITORY,
    "revision": MODEL_REVISION,
    "snapshot": str(snapshot),
    "runtime_commit": runtime_commit,
    "huggingface_hub_version": huggingface_hub.__version__,
    "files": files,
    "file_count": len(files),
    "bytes": sum(row["bytes"] for row in files),
    "ordered_files_sha256": canonical_sha256(files),
    "gpu_requested": False,
    "training_ready": False,
    "four_b_training_authorized": False,
}
payload["receipt_sha256"] = canonical_sha256(payload)
path = Path(destination)
temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
with temporary.open("x", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
print(json.dumps({"bytes": payload["bytes"], "files": len(files)}, sort_keys=True))
PY
