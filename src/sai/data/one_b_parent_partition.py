"""Repartition Sai 1B remote curriculum locators by pinned source parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.one_b_curriculum_index import SHARD_SCHEMA as INDEX_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-1b-parent-partition-shard-v1"
PARTITION_COUNT = 128


class OneBParentPartitionError(RuntimeError):
    """A curriculum index, parent partition, or identity differs."""


def _load_index(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBParentPartitionError("curriculum index receipt differs") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or value.get("schema") != INDEX_SCHEMA
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise OneBParentPartitionError("curriculum index receipt differs")
    return value


def _bucket(path: str) -> int:
    value = int.from_bytes(hashlib.sha256(path.encode()).digest()[:8], "big")
    return value % PARTITION_COUNT


def partition(index_root: Path, output_root: Path, component: str) -> dict[str, Any]:
    """Write train-only, source-text-free locators into stable parent buckets."""

    if (
        component not in {"pleias", "code"}
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise OneBParentPartitionError("parent partition arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise OneBParentPartitionError("pyarrow is required") from error
    receipt = _load_index(index_root / "receipt.json")
    descriptor = receipt.get("output", {})
    source = index_root / descriptor.get("path", "")
    if (
        receipt.get("component") != component
        or not source.is_file()
        or source.is_symlink()
        or source.stat().st_size != descriptor.get("bytes")
        or sha256_file(source) != descriptor.get("sha256")
    ):
        raise OneBParentPartitionError("curriculum index source differs")
    table = pq.read_table(source)
    train_rows = [row for row in table.to_pylist() if row["split"] == "train"]
    for row in train_rows:
        row["parent_bucket"] = _bucket(row["source_path"])
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    try:
        if train_rows:
            partitioned = pa.Table.from_pylist(train_rows)
            pq.write_to_dataset(
                partitioned,
                root_path=stage,
                partition_cols=["parent_bucket"],
                compression="zstd",
                use_threads=False,
            )
        files = []
        output_rows = 0
        for path in sorted(stage.glob("parent_bucket=*/*.parquet")):
            rows = pq.ParquetFile(path).metadata.num_rows
            output_rows += rows
            files.append(
                {
                    "path": str(path.relative_to(stage)),
                    "rows": rows,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        if output_rows != len(train_rows):
            raise OneBParentPartitionError("parent partition coverage differs")
        counts: Counter[str] = Counter()
        ordered = hashlib.sha256()
        for row in train_rows:
            counts["rows"] += 1
            counts["text_utf8_bytes"] += row["text_utf8_bytes"]
            counts["source_token_estimate"] += row["source_token_estimate"]
            counts[f"band::{row['curriculum_band']}::rows"] += 1
            ordered.update(bytes.fromhex(row["document_identity_sha256"]))
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_1b_parent_partition_shard",
            "component": component,
            "source_shard": receipt["source_shard"],
            "source_index_receipt_sha256": receipt["receipt_sha256"],
            "partition_count": PARTITION_COUNT,
            "partition_function": "sha256-source-path-first8-big-endian-mod-128",
            "development_rows_excluded": True,
            "counts": dict(sorted(counts.items())),
            "ordered_train_identities_sha256": ordered.hexdigest(),
            "files": files,
            "files_sha256": canonical_sha256(files),
            "source_text_persisted": False,
            "model_training_started": False,
            "one_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage / "receipt.json", payload)
        os.replace(stage, output_root)
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--component", choices=("pleias", "code"), required=True)
    args = parser.parse_args()
    result = partition(args.index_root, args.output_root, args.component)
    print(json.dumps({"receipt_sha256": result["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
