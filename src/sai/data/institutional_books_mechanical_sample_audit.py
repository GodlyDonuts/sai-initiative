"""Audit the mechanical gate against real, hash-bound private book rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books_materializer import (
    OUTPUT_SCHEMA,
    PARENT_SCHEMA,
    SHARD_SCHEMA,
    _load_json,
    _valid_receipt,
    _validate_parent_files,
)
from sai.data.source_quality_gate import POLICY_SHA256, mechanical_quality_evidence
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-institutional-books-mechanical-real-text-audit-v1"
ROW_SCHEMA = "sai-institutional-books-mechanical-real-text-audit-row-v1"


class InstitutionalBooksMechanicalSampleAuditError(RuntimeError):
    """Complete shard custody, sampled source row, or audit output differs."""


def _sample_positions(size: int, requested: int) -> list[int]:
    if size <= 0 or requested <= 0:
        return []
    count = min(size, requested)
    return sorted({index * size // count for index in range(count)})


def _read_row(path: Path, identity: str) -> tuple[int, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksMechanicalSampleAuditError(
            "pyarrow is required"
        ) from error
    parquet = pq.ParquetFile(path)
    rows = parquet.metadata.num_rows
    if rows <= 0:
        raise InstitutionalBooksMechanicalSampleAuditError(
            "sampled parent output is empty"
        )
    target = int(identity, 16) % rows
    offset = 0
    columns = [
        "schema",
        "barcode_src",
        "source_content_sha256",
        "text",
        "training_ready",
    ]
    for row_group in range(parquet.metadata.num_row_groups):
        count = parquet.metadata.row_group(row_group).num_rows
        if target < offset + count:
            table = parquet.read_row_group(
                row_group, columns=columns, use_threads=False
            )
            return target, table.slice(target - offset, 1).to_pylist()[0]
        offset += count
    raise InstitutionalBooksMechanicalSampleAuditError("sampled row differs")


def build_audit(
    materialized_root: Path,
    output: Path,
    *,
    logical_shards: int = 64,
    rows_per_complete_shard: int = 2,
) -> dict[str, Any]:
    """Audit deterministic real rows from every shard complete at execution."""

    if (
        output.exists()
        or output.is_symlink()
        or logical_shards <= 0
        or rows_per_complete_shard <= 0
    ):
        raise InstitutionalBooksMechanicalSampleAuditError("audit arguments differ")
    audited_rows = []
    shard_receipts = []
    counts: Counter[str] = Counter()
    for shard_index in range(logical_shards):
        root = materialized_root / "shards" / f"shard_{shard_index:05d}"
        receipt_path = root / "receipt.json"
        if not receipt_path.exists():
            continue
        shard = _load_json(receipt_path)
        if (
            not _valid_receipt(shard, SHARD_SCHEMA)
            or shard.get("logical_shards") != logical_shards
            or shard.get("shard_index") != shard_index
        ):
            raise InstitutionalBooksMechanicalSampleAuditError(
                "complete shard receipt differs"
            )
        shard_receipts.append(shard["receipt_sha256"])
        parents = []
        for path in sorted((root / "parents").glob("parent_*.json")):
            parent = _load_json(path)
            if not _valid_receipt(parent, PARENT_SCHEMA):
                raise InstitutionalBooksMechanicalSampleAuditError(
                    "parent receipt differs"
                )
            if parent.get("output") is not None:
                parents.append((path, parent))
        for position in _sample_positions(len(parents), rows_per_complete_shard):
            parent_path, parent = parents[position]
            _validate_parent_files(root, parent)
            descriptor = parent["output"]
            source_path = root / descriptor["path"]
            selection_identity = canonical_sha256(
                {
                    "schema": ROW_SCHEMA,
                    "shard_receipt_sha256": shard["receipt_sha256"],
                    "parent_receipt_sha256": parent["receipt_sha256"],
                    "mechanical_policy_sha256": POLICY_SHA256,
                }
            )
            source_row_index, row = _read_row(source_path, selection_identity)
            text = row.get("text")
            content_sha256 = row.get("source_content_sha256")
            if (
                row.get("schema") != OUTPUT_SCHEMA
                or row.get("training_ready") is not False
                or not isinstance(row.get("barcode_src"), str)
                or not isinstance(text, str)
                or not isinstance(content_sha256, str)
                or hashlib.sha256(text.encode()).hexdigest() != content_sha256
            ):
                raise InstitutionalBooksMechanicalSampleAuditError(
                    "sampled materialized row differs"
                )
            evidence = mechanical_quality_evidence(text)
            audit_row = {
                "schema": ROW_SCHEMA,
                "shard_index": shard_index,
                "shard_receipt_sha256": shard["receipt_sha256"],
                "parent_receipt_path_name": parent_path.name,
                "parent_receipt_sha256": parent["receipt_sha256"],
                "parent_output_sha256": descriptor["sha256"],
                "source_row_index": source_row_index,
                "source_content_sha256": content_sha256,
                "source_barcode_sha256": hashlib.sha256(
                    row["barcode_src"].encode()
                ).hexdigest(),
                "mechanical_policy_sha256": POLICY_SHA256,
                "decision": evidence["decision"],
                "reasons": evidence["reasons"],
                "measurements": evidence["measurements"],
                "flags": evidence["flags"],
                "source_text_persisted": False,
                "training_ready": False,
            }
            audit_row["row_sha256"] = canonical_sha256(audit_row)
            audited_rows.append(audit_row)
            counts[f"decision::{evidence['decision']}"] += 1
            for reason in evidence["reasons"]:
                counts[f"reason::{reason}"] += 1
    if not shard_receipts or not audited_rows:
        raise InstitutionalBooksMechanicalSampleAuditError(
            "no complete book shards were auditable"
        )
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_real_book_mechanical_sample_audit",
        "materialized_root_name": materialized_root.name,
        "logical_shards": logical_shards,
        "complete_shards_at_execution": len(shard_receipts),
        "ordered_complete_shard_receipts_sha256": canonical_sha256(shard_receipts),
        "rows_per_complete_shard": rows_per_complete_shard,
        "sampled_rows": len(audited_rows),
        "counts": dict(sorted(counts.items())),
        "rows": audited_rows,
        "ordered_rows_sha256": canonical_sha256(
            [row["row_sha256"] for row in audited_rows]
        ),
        "mechanical_policy_sha256": POLICY_SHA256,
        "sample_is_full_population_audit": False,
        "sample_is_acceptance_rate_estimate": False,
        "source_text_persisted": False,
        "changes_training_admission": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, default=64)
    parser.add_argument("--rows-per-complete-shard", type=int, default=2)
    args = parser.parse_args()
    result = build_audit(
        args.materialized_root,
        args.output,
        logical_shards=args.logical_shards,
        rows_per_complete_shard=args.rows_per_complete_shard,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
