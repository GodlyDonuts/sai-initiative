#!/bin/bash
#SBATCH --job-name=sai-acquire-finemath-4plus
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --no-requeue

set -euo pipefail
umask 077

: "${PYTHON:?PYTHON is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

DATASET=HuggingFaceTB/finemath
REVISION=e92b25a616738fe95dc186b64dfb19f9c8525594
FINAL="$OUTPUT_ROOT/finemath-4plus-e92b25a-r1"
STAGE="$OUTPUT_ROOT/.finemath-4plus-e92b25a-r1.stage"

test -x "$PYTHON"
test ! -L "$PYTHON"
test ! -e "$FINAL"
mkdir -p "$STAGE" "${SLURM_TMPDIR:?}/hf-cache"

env -i \
  PATH=/usr/bin:/bin \
  HF_HOME="$SLURM_TMPDIR/hf-cache" \
  PYTHONDONTWRITEBYTECODE=1 \
  DATASET="$DATASET" \
  REVISION="$REVISION" \
  STAGE="$STAGE" \
  "$PYTHON" -P -s -B - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

dataset = os.environ["DATASET"]
revision = os.environ["REVISION"]
stage = Path(os.environ["STAGE"])
api = HfApi()
rows = sorted(
    (
        row
        for row in api.list_repo_tree(
            dataset,
            repo_type="dataset",
            revision=revision,
            recursive=True,
            expand=True,
        )
        if getattr(row, "path", "").startswith("finemath-4plus/")
        and row.path.endswith(".parquet")
    ),
    key=lambda row: row.path,
)
if len(rows) != 64 or any(row.lfs is None for row in rows):
    raise RuntimeError("FineMath 4plus inventory differs")
if sum(row.size for row in rows) != 18_365_184_633:
    raise RuntimeError("FineMath 4plus byte total differs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_rows = []
for index, row in enumerate(rows, start=1):
    relative = Path(row.path).relative_to("finemath-4plus")
    if relative.is_absolute() or len(relative.parts) != 1:
        raise RuntimeError("FineMath member path differs")
    target = stage / relative
    expected_sha256 = row.lfs.sha256
    if target.exists():
        if (
            not target.is_file()
            or target.is_symlink()
            or target.stat().st_size != row.size
            or sha256_file(target) != expected_sha256
        ):
            raise RuntimeError("partial FineMath member differs")
    else:
        source = Path(
            hf_hub_download(
                dataset,
                filename=row.path,
                repo_type="dataset",
                revision=revision,
            )
        )
        temporary = stage / f".{relative.name}.tmp"
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 << 20)
            writer.flush()
            os.fsync(writer.fileno())
        if temporary.stat().st_size != row.size or sha256_file(temporary) != expected_sha256:
            raise RuntimeError("downloaded FineMath member differs")
        os.replace(temporary, target)
    manifest_rows.append(
        {"path": relative.as_posix(), "bytes": row.size, "sha256": expected_sha256}
    )
    print(f"verified {index:02d}/{len(rows):02d} {relative}", flush=True)

manifest = {
    "schema": "sai-hf-source-manifest-v1",
    "status": "download_complete_candidate_only",
    "dataset": dataset,
    "revision": revision,
    "subset": "finemath-4plus",
    "files": manifest_rows,
    "file_count": len(manifest_rows),
    "total_bytes": sum(row["bytes"] for row in manifest_rows),
    "source_admitted": False,
    "training_authorized": False,
}
encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
manifest["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
temporary = stage / ".source-manifest.json.tmp"
with temporary.open("x") as handle:
    json.dump(manifest, handle, ensure_ascii=False, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, stage / "source-manifest.json")
PY

find "$STAGE" -type f -exec chmod 0444 -- {} +
find "$STAGE" -type d -exec chmod 0555 -- {} +
mv -- "$STAGE" "$FINAL"
printf 'sai: FineMath 4plus acquisition complete: %s\n' "$FINAL"
