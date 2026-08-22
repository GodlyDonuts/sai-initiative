"""Audit one inventory-bound compressed Hugging Face JSONL shard."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.hf_dataset_inventory import (
    HFDatasetInventoryError,
    validate_inventory,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-hf-compressed-shard-audit-v1"


class HFShardAuditError(RuntimeError):
    """The inventory, compressed member, or row population differs."""


def _regular(path: Path, field: str) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise HFShardAuditError(f"{field} is missing or unsafe")


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _rows(path: Path):
    try:
        import zstandard
    except ImportError as error:
        raise HFShardAuditError(
            "zstandard is required to audit compressed shards"
        ) from error
    try:
        with path.open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text:
                    yield from text
    except (OSError, UnicodeDecodeError, zstandard.ZstdError) as error:
        raise HFShardAuditError("compressed shard cannot be decoded") from error


def audit_shard(
    inventory_path: Path, compressed_path: Path, *, member_path: str
) -> dict[str, Any]:
    """Reopen an inventory and audit exact duplicate/provenance metadata."""

    _regular(inventory_path, "dataset inventory")
    _regular(compressed_path, "compressed shard")
    inventory_file_sha256 = sha256_file(inventory_path)
    try:
        inventory = validate_inventory(json.loads(inventory_path.read_bytes()))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        HFDatasetInventoryError,
    ) as error:
        raise HFShardAuditError("dataset inventory differs") from error
    members = [row for row in inventory["files"] if row["path"] == member_path]
    if len(members) != 1 or not member_path.startswith("data/"):
        raise HFShardAuditError("compressed shard is absent from inventory")
    member = members[0]
    compressed_sha256 = sha256_file(compressed_path)
    if (
        compressed_path.stat().st_size != member["bytes"]
        or compressed_sha256 != member["sha256"]
    ):
        raise HFShardAuditError("compressed shard bytes differ")

    id_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()
    id_text: dict[str, str] = {}
    sources: Counter[str] = Counter()
    license_types: Counter[str] = Counter()
    integer_scores: Counter[str] = Counter()
    metadata_keys: Counter[str] = Counter()
    lengths: list[int] = []
    empty_text_rows = 0
    ordered_identity = []
    for line_number, line in enumerate(_rows(compressed_path), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise HFShardAuditError(f"shard row {line_number} is malformed") from error
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or not isinstance(row.get("text"), str)
            or not isinstance(row.get("metadata", {}), dict)
            or row.get("source") is not None
            and not isinstance(row.get("source"), str)
        ):
            raise HFShardAuditError(f"shard row {line_number} contract differs")
        identity = row["id"]
        text = row["text"]
        empty_text_rows += int(not text)
        text_sha256 = hashlib.sha256(text.encode()).hexdigest()
        if identity in id_text and id_text[identity] != text_sha256:
            raise HFShardAuditError("one document identity maps to multiple texts")
        id_text[identity] = text_sha256
        id_counts[identity] += 1
        text_counts[text_sha256] += 1
        source = row.get("source")
        sources["<null>" if source is None else source] += 1
        metadata = row.get("metadata", {})
        license_type = metadata.get("license_type")
        license_types["<missing>" if license_type is None else str(license_type)] += 1
        score = metadata.get("int_score")
        integer_scores["<missing>" if score is None else str(score)] += 1
        metadata_keys.update(metadata.keys())
        lengths.append(len(text))
        ordered_identity.append({"id": identity, "text_sha256": text_sha256})
    if not lengths:
        raise HFShardAuditError("compressed shard contains no rows")
    rows = len(lengths)
    unique_ids = len(id_counts)
    unique_texts = len(text_counts)
    multiplicities = Counter(id_counts.values())
    source_rows = sorted(sources.items(), key=lambda item: (-item[1], item[0]))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "diagnostic_complete_source_not_admitted",
        "training_authorized": False,
        "source_admitted": False,
        "rows_selected_for_training": 0,
        "dataset_inventory": {
            "file_sha256": inventory_file_sha256,
            "receipt_sha256": inventory["receipt_sha256"],
            "dataset": inventory["dataset"],
            "revision": inventory["revision"],
        },
        "member": {
            "path": member_path,
            "compressed_bytes": member["bytes"],
            "compressed_sha256": compressed_sha256,
        },
        "population": {
            "rows": rows,
            "unique_document_ids": unique_ids,
            "unique_texts": unique_texts,
            "duplicate_document_id_rows": rows - unique_ids,
            "duplicate_text_rows": rows - unique_texts,
            "duplicate_document_id_fraction": (rows - unique_ids) / rows,
            "duplicate_text_fraction": (rows - unique_texts) / rows,
            "distinct_ids_sharing_text": unique_ids - unique_texts,
            "empty_text_rows": empty_text_rows,
            "max_document_id_multiplicity": max(id_counts.values()),
            "document_id_multiplicity_histogram": {
                str(value): count for value, count in sorted(multiplicities.items())
            },
            "text_length_characters": {
                "minimum": min(lengths),
                "median": _percentile(lengths, 0.5),
                "p95": _percentile(lengths, 0.95),
                "maximum": max(lengths),
            },
            "ordered_identity_sha256": canonical_sha256(ordered_identity),
        },
        "metadata": {
            "distinct_sources": len(sources),
            "source_counts_sha256": canonical_sha256(sorted(sources.items())),
            "top_source_counts": [
                {"source": source, "rows": count} for source, count in source_rows[:20]
            ],
            "license_type_counts": dict(sorted(license_types.items())),
            "integer_score_counts": dict(sorted(integer_scores.items())),
            "metadata_key_row_counts": dict(sorted(metadata_keys.items())),
        },
        "checks": {
            "inventory_replayed": True,
            "compressed_size_and_sha256_match": True,
            "all_rows_parsed": True,
            "identity_text_mapping_consistent": True,
            "duplicates_measured_not_silently_expanded": True,
            "license_metadata_reported_not_inferred_from_dataset_wrapper": True,
            "diagnostic_sample_not_population_estimate": True,
            "no_source_admission_or_training": True,
        },
    }
    if (
        sha256_file(inventory_path) != inventory_file_sha256
        or sha256_file(compressed_path) != compressed_sha256
    ):
        raise HFShardAuditError("audit input changed while reading")
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def audit_to_file(
    inventory_path: Path,
    compressed_path: Path,
    output_path: Path,
    *,
    member_path: str,
) -> dict[str, Any]:
    if output_path.exists() or output_path.is_symlink():
        raise HFShardAuditError("shard audit output already exists")
    payload = audit_shard(inventory_path, compressed_path, member_path=member_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--compressed-shard", type=Path, required=True)
    parser.add_argument("--member-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_to_file(
        args.inventory,
        args.compressed_shard,
        args.output,
        member_path=args.member_path,
    )
    print(
        json.dumps(
            {"receipt_sha256": payload["receipt_sha256"], "status": payload["status"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
