"""Merge sealed, text-free quarantine manifests into one materialization deny list."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-quarantine-exclusion-registry-receipt-v1"
RECORD_SCHEMA = "sai-quarantine-exclusion-registry-record-v1"
SUPPORTED_RECEIPT_SCHEMAS = {
    "sai-audit-quarantine-manifest-receipt-v1",
    "sai-institutional-books-quarantine-manifest-receipt-v1",
}
FORBIDDEN_TEXT_KEYS = {"text", "content", "source_text", "text_excerpt", "excerpt"}


class QuarantineExclusionRegistryError(RuntimeError):
    """A source manifest, exclusion identity, or registry output differs."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise QuarantineExclusionRegistryError("quarantine evidence is missing or unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise QuarantineExclusionRegistryError("quarantine evidence is invalid") from error
    if not isinstance(value, dict):
        raise QuarantineExclusionRegistryError("quarantine evidence is invalid")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _contains_forbidden_text(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_TEXT_KEYS or _contains_forbidden_text(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_text(item) for item in value)
    return False


def _load_manifest(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise QuarantineExclusionRegistryError("quarantine manifest root is unsafe")
    receipt_path = root / "receipt.json"
    receipt = _load_json(receipt_path)
    claimed = receipt.get("receipt_sha256")
    unsigned = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("manifest")
    if (
        receipt.get("schema") not in SUPPORTED_RECEIPT_SCHEMAS
        or not _is_sha256(claimed)
        or claimed != canonical_sha256(unsigned)
        or receipt.get("source_text_persisted") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
        or descriptor.get("path") != "quarantine_exclusions.jsonl"
        or not isinstance(descriptor.get("rows"), int)
        or isinstance(descriptor.get("rows"), bool)
        or descriptor["rows"] < 0
    ):
        raise QuarantineExclusionRegistryError("quarantine manifest receipt differs")
    manifest_path = root / descriptor["path"]
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path.stat().st_nlink != 1
        or manifest_path.stat().st_size != descriptor.get("bytes")
        or sha256_file(manifest_path) != descriptor.get("sha256")
    ):
        raise QuarantineExclusionRegistryError("quarantine manifest bytes differ")
    rows = []
    with manifest_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise QuarantineExclusionRegistryError(
                    "quarantine manifest row is invalid"
                ) from error
            if not isinstance(row, dict):
                raise QuarantineExclusionRegistryError(
                    "quarantine manifest row is invalid"
                )
            record_hash = row.get("record_sha256")
            row_unsigned = {
                key: item for key, item in row.items() if key != "record_sha256"
            }
            if (
                not _is_sha256(row.get("candidate_identity_sha256"))
                or not _is_sha256(row.get("source_content_sha256"))
                or not _is_sha256(record_hash)
                or record_hash != canonical_sha256(row_unsigned)
                or row.get("route") != "quarantine"
                or row.get("dataset_materialization_allowed") is not False
                or row.get("source_text_persisted") is not False
                or _contains_forbidden_text(row)
            ):
                raise QuarantineExclusionRegistryError(
                    "quarantine manifest record differs"
                )
            rows.append(row)
    if len(rows) != descriptor["rows"]:
        raise QuarantineExclusionRegistryError("quarantine manifest row coverage differs")
    return rows, {
        "root_name": root.name,
        "schema": receipt["schema"],
        "receipt_file_sha256": sha256_file(receipt_path),
        "receipt_sha256": claimed,
        "manifest_sha256": descriptor["sha256"],
        "rows": len(rows),
    }


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    stage = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with stage.open("x") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, path)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise


def build_registry(manifest_roots: list[Path], output_root: Path) -> dict[str, Any]:
    """Build a deterministic deny registry from complete exclusion manifests."""

    if (
        not manifest_roots
        or len(manifest_roots) != len(set(manifest_roots))
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise QuarantineExclusionRegistryError("quarantine registry inputs differ")
    source_descriptors = []
    by_identity: dict[str, dict[str, Any]] = {}
    source_counts: Counter[str] = Counter()
    for root in manifest_roots:
        rows, source = _load_manifest(root)
        source_descriptors.append(source)
        source_counts[source["schema"]] += len(rows)
        for row in rows:
            identity = row["candidate_identity_sha256"]
            if identity in by_identity:
                raise QuarantineExclusionRegistryError(
                    "quarantine candidate identity is duplicated"
                )
            registry_row = {
                "schema": RECORD_SCHEMA,
                "candidate_identity_sha256": identity,
                "source_content_sha256": row["source_content_sha256"],
                "source_manifest_receipt_sha256": source["receipt_sha256"],
                "source_record_sha256": row["record_sha256"],
                "route": "quarantine",
                "dataset_materialization_allowed": False,
                "source_text_persisted": False,
            }
            registry_row["record_sha256"] = canonical_sha256(registry_row)
            by_identity[identity] = registry_row
    rows = [by_identity[key] for key in sorted(by_identity)]
    output_root.mkdir(parents=True)
    try:
        manifest_path = output_root / "quarantine_registry.jsonl"
        _atomic_jsonl(manifest_path, rows)
        payload = {
            "schema": SCHEMA,
            "status": "complete_quarantine_exclusion_registry",
            "sources": source_descriptors,
            "source_manifest_count": len(source_descriptors),
            "source_rows": sum(source["rows"] for source in source_descriptors),
            "unique_quarantine_rows": len(rows),
            "rows_by_source_schema": dict(sorted(source_counts.items())),
            "registry": {
                "path": manifest_path.name,
                "rows": len(rows),
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in rows]
                ),
            },
            "dataset_materialization_allowed": False,
            "source_text_persisted": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_registry(args.manifest_root, args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
