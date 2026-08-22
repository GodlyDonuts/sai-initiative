"""Freeze an exact no-download Hugging Face dataset file inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-hf-dataset-inventory-v1"


class HFDatasetInventoryError(RuntimeError):
    """The exact dataset API response or inventory differs."""


def _hex(value: object, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise HFDatasetInventoryError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise HFDatasetInventoryError(f"{field} differs") from error
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HFDatasetInventoryError(f"{field} differs")
    return value


def _safe_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise HFDatasetInventoryError("dataset member path differs")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise HFDatasetInventoryError("dataset member path is unsafe")
    return value


def build_inventory(
    payload: object,
    *,
    dataset: str,
    revision: str,
    api_response_sha256: str,
) -> dict[str, Any]:
    """Validate an exact Hub API response without downloading dataset members."""

    if not isinstance(dataset, str) or dataset.count("/") != 1:
        raise HFDatasetInventoryError("dataset identity differs")
    revision = _hex(revision, 40, "dataset revision")
    api_response_sha256 = _hex(api_response_sha256, 64, "API response")
    if (
        not isinstance(payload, dict)
        or payload.get("id") != dataset
        or payload.get("sha") != revision
        or payload.get("private") is not False
        or payload.get("gated") not in {False, None}
    ):
        raise HFDatasetInventoryError("dataset API identity differs")
    siblings = payload.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise HFDatasetInventoryError("dataset API members differ")

    files = []
    seen = set()
    component_counts: Counter[str] = Counter()
    component_bytes: Counter[str] = Counter()
    data_files = data_bytes = 0
    for member in siblings:
        if not isinstance(member, dict):
            raise HFDatasetInventoryError("dataset API member differs")
        path = _safe_path(member.get("rfilename"))
        if path in seen:
            raise HFDatasetInventoryError("dataset member paths are duplicated")
        seen.add(path)
        size = _positive_int(member.get("size"), f"{path} size")
        blob_id = _hex(member.get("blobId"), 40, f"{path} blob")
        lfs = member.get("lfs")
        row: dict[str, Any] = {"path": path, "bytes": size, "blob_id": blob_id}
        if path.startswith("data/"):
            parts = PurePosixPath(path).parts
            if (
                len(parts) != 3
                or not path.endswith(".jsonl.zst")
                or not isinstance(lfs, dict)
                or lfs.get("size") != size
            ):
                raise HFDatasetInventoryError("dataset data member differs")
            row["sha256"] = _hex(lfs.get("sha256"), 64, f"{path} LFS SHA256")
            component = parts[1]
            component_counts[component] += 1
            component_bytes[component] += size
            data_files += 1
            data_bytes += size
        elif lfs is not None:
            if not isinstance(lfs, dict) or lfs.get("size") != size:
                raise HFDatasetInventoryError("dataset metadata LFS member differs")
            row["sha256"] = _hex(lfs.get("sha256"), 64, f"{path} LFS SHA256")
        else:
            row["sha256"] = None
        files.append(row)
    if data_files <= 0 or data_files == len(files):
        raise HFDatasetInventoryError("dataset data/metadata boundary differs")
    files.sort(key=lambda row: row["path"])
    components = [
        {
            "component": component,
            "files": component_counts[component],
            "compressed_bytes": component_bytes[component],
        }
        for component in sorted(component_counts)
    ]
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "metadata_inventory_complete_content_not_acquired",
        "training_authorized": False,
        "source_admitted": False,
        "content_downloaded": False,
        "dataset": dataset,
        "revision": revision,
        "api_response_sha256": api_response_sha256,
        "file_count": len(files),
        "data_file_count": data_files,
        "data_compressed_bytes": data_bytes,
        "component_partition_count": len(components),
        "files_sha256": canonical_sha256(files),
        "component_partitions_sha256": canonical_sha256(components),
        "files": files,
        "component_partitions": components,
        "checks": {
            "exact_revision": True,
            "unique_safe_paths": True,
            "lfs_sha256_and_size_bound": True,
            "no_dataset_content_opened": True,
            "license_quality_decontamination_pending": True,
            "no_training_or_source_admission": True,
        },
    }
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def inventory_bytes(encoded: bytes, *, dataset: str, revision: str) -> dict[str, Any]:
    if not encoded:
        raise HFDatasetInventoryError("dataset API response is empty")
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFDatasetInventoryError("dataset API response is malformed") from error
    return build_inventory(
        payload,
        dataset=dataset,
        revision=revision,
        api_response_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def validate_inventory(payload: object) -> dict[str, Any]:
    """Replay a generated no-download dataset inventory."""

    expected = {
        "schema",
        "status",
        "training_authorized",
        "source_admitted",
        "content_downloaded",
        "dataset",
        "revision",
        "api_response_sha256",
        "file_count",
        "data_file_count",
        "data_compressed_bytes",
        "component_partition_count",
        "files_sha256",
        "component_partitions_sha256",
        "files",
        "component_partitions",
        "checks",
        "receipt_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected
        or payload.get("schema") != SCHEMA
        or payload.get("status") != "metadata_inventory_complete_content_not_acquired"
        or payload.get("training_authorized") is not False
        or payload.get("source_admitted") is not False
        or payload.get("content_downloaded") is not False
        or not isinstance(payload.get("dataset"), str)
        or payload["dataset"].count("/") != 1
    ):
        raise HFDatasetInventoryError("dataset inventory contract differs")
    _hex(payload["revision"], 40, "inventory revision")
    _hex(payload["api_response_sha256"], 64, "inventory API response")
    files = payload.get("files")
    components = payload.get("component_partitions")
    if not isinstance(files, list) or not files or not isinstance(components, list):
        raise HFDatasetInventoryError("dataset inventory members differ")
    paths = []
    data_files = data_bytes = 0
    component_counts: Counter[str] = Counter()
    component_bytes: Counter[str] = Counter()
    for row in files:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "bytes",
            "blob_id",
            "sha256",
        }:
            raise HFDatasetInventoryError("dataset inventory member differs")
        path = _safe_path(row["path"])
        paths.append(path)
        size = _positive_int(row["bytes"], f"{path} inventory size")
        _hex(row["blob_id"], 40, f"{path} inventory blob")
        if path.startswith("data/"):
            if len(PurePosixPath(path).parts) != 3 or not path.endswith(".jsonl.zst"):
                raise HFDatasetInventoryError("dataset inventory data member differs")
            _hex(row["sha256"], 64, f"{path} inventory SHA256")
            component = PurePosixPath(path).parts[1]
            component_counts[component] += 1
            component_bytes[component] += size
            data_files += 1
            data_bytes += size
        elif row["sha256"] is not None:
            _hex(row["sha256"], 64, f"{path} inventory SHA256")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise HFDatasetInventoryError("dataset inventory paths differ")
    rebuilt_components = [
        {
            "component": component,
            "files": component_counts[component],
            "compressed_bytes": component_bytes[component],
        }
        for component in sorted(component_counts)
    ]
    if (
        components != rebuilt_components
        or payload.get("file_count") != len(files)
        or payload.get("data_file_count") != data_files
        or payload.get("data_compressed_bytes") != data_bytes
        or payload.get("component_partition_count") != len(components)
        or payload.get("files_sha256") != canonical_sha256(files)
        or payload.get("component_partitions_sha256") != canonical_sha256(components)
        or payload.get("checks")
        != {
            "exact_revision": True,
            "unique_safe_paths": True,
            "lfs_sha256_and_size_bound": True,
            "no_dataset_content_opened": True,
            "license_quality_decontamination_pending": True,
            "no_training_or_source_admission": True,
        }
        or payload.get("receipt_sha256")
        != canonical_sha256(
            {key: value for key, value in payload.items() if key != "receipt_sha256"}
        )
    ):
        raise HFDatasetInventoryError("dataset inventory evidence differs")
    return payload


def _write(output: Path, payload: dict[str, Any]) -> None:
    if output.exists() or output.is_symlink():
        raise HFDatasetInventoryError("dataset inventory output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    try:
        if output.is_symlink() or output.stat().st_nlink != 1:
            raise HFDatasetInventoryError("dataset inventory output is unsafe")
        validate_inventory(json.loads(output.read_bytes()))
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--api-response", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.api_response == "-":
        encoded = sys.stdin.buffer.read()
    else:
        path = Path(args.api_response)
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise HFDatasetInventoryError("dataset API response is missing or unsafe")
        encoded = path.read_bytes()
    payload = inventory_bytes(encoded, dataset=args.dataset, revision=args.revision)
    _write(args.output, payload)
    print(
        json.dumps(
            {
                "receipt_sha256": payload["receipt_sha256"],
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
