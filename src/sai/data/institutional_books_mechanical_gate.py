"""Mechanically gate every privately materialized Institutional Books row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books_materializer import (
    AGGREGATE_SCHEMA as MATERIALIZER_AGGREGATE_SCHEMA,
)
from sai.data.institutional_books_materializer import (
    OUTPUT_SCHEMA,
    PARENT_SCHEMA,
    _load_json,
    _valid_receipt,
)
from sai.data.institutional_books_materializer import (
    SHARD_SCHEMA as MATERIALIZER_SHARD_SCHEMA,
)
from sai.data.source_quality_gate import POLICY_SHA256, mechanical_quality_evidence
from sai.data.token_stream import canonical_sha256, sha256_file

DECISION_SCHEMA = "sai-institutional-books-mechanical-decision-v1"
SHARD_SCHEMA = "sai-institutional-books-mechanical-gate-shard-v1"
AGGREGATE_SCHEMA = "sai-institutional-books-mechanical-gate-aggregate-v1"


class InstitutionalBooksMechanicalGateError(RuntimeError):
    """Materialized book custody or mechanical decisions differ."""


def _atomic_jsonl(path: Path, emit) -> None:
    if path.exists() or path.is_symlink():
        raise InstitutionalBooksMechanicalGateError("mechanical output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial.{uuid.uuid4().hex}"
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
        )
        with os.fdopen(descriptor, "w") as handle:
            emit(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _decision(row: dict[str, Any], source_path: str) -> dict[str, Any]:
    text = row.get("text")
    barcode = row.get("barcode_src")
    content_sha256 = row.get("source_content_sha256")
    if (
        row.get("schema") != OUTPUT_SCHEMA
        or not isinstance(barcode, str)
        or not barcode
        or not isinstance(text, str)
        or len(text.encode()) < 200
        or not isinstance(content_sha256, str)
        or hashlib.sha256(text.encode()).hexdigest() != content_sha256
        or row.get("training_ready") is not False
    ):
        raise InstitutionalBooksMechanicalGateError("materialized book row differs")
    evidence = mechanical_quality_evidence(text)
    decision = {
        "schema": DECISION_SCHEMA,
        "barcode_src": barcode,
        "source_content_sha256": content_sha256,
        "source_path": source_path,
        "policy_sha256": POLICY_SHA256,
        "decision": evidence["decision"],
        "reasons": evidence["reasons"],
        "flags": evidence["flags"],
        "semantic_admission_complete": False,
        "training_ready": False,
    }
    decision["decision_sha256"] = canonical_sha256(decision)
    return decision


def run_shard(
    materialized_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    """Verify and scan one complete materializer shard without copying text."""

    if not 0 <= shard_index < logical_shards:
        raise InstitutionalBooksMechanicalGateError("mechanical shard differs")
    source_root = materialized_root / "shards" / f"shard_{shard_index:05d}"
    source_receipt = _load_json(source_root / "receipt.json")
    if (
        not _valid_receipt(source_receipt, MATERIALIZER_SHARD_SCHEMA)
        or source_receipt.get("logical_shards") != logical_shards
        or source_receipt.get("shard_index") != shard_index
    ):
        raise InstitutionalBooksMechanicalGateError("materializer shard differs")
    root = output_root / "shards" / f"shard_{shard_index:05d}"
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        existing = _load_json(receipt_path)
        if (
            not _valid_receipt(existing, SHARD_SCHEMA)
            or existing.get("source", {}).get("receipt_sha256")
            != source_receipt["receipt_sha256"]
        ):
            raise InstitutionalBooksMechanicalGateError(
                "existing mechanical shard differs"
            )
        return existing
    decision_path = root / "decisions.jsonl"
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    seen_barcodes: set[str] = set()
    ordered_decisions: list[str] = []
    verified_parent_outputs: list[dict[str, Any]] = []

    def emit(handle) -> None:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise InstitutionalBooksMechanicalGateError(
                "pyarrow is required"
            ) from error
        for parent_path in sorted((source_root / "parents").glob("parent_*.json")):
            parent = _load_json(parent_path)
            if not _valid_receipt(parent, PARENT_SCHEMA):
                raise InstitutionalBooksMechanicalGateError(
                    "materialized parent differs"
                )
            output = parent.get("output")
            if output is None:
                continue
            parquet_path = source_root / output["path"]
            if (
                not parquet_path.is_file()
                or parquet_path.is_symlink()
                or parquet_path.stat().st_nlink != 1
                or parquet_path.stat().st_size != output.get("bytes")
                or sha256_file(parquet_path) != output.get("sha256")
            ):
                raise InstitutionalBooksMechanicalGateError(
                    "materialized parent output differs"
                )
            parent_rows = 0
            source_path = parent.get("source", {}).get("path")
            if not isinstance(source_path, str) or not source_path:
                raise InstitutionalBooksMechanicalGateError(
                    "materialized parent identity differs"
                )
            parquet = pq.ParquetFile(parquet_path)
            columns = [
                "schema",
                "barcode_src",
                "text",
                "source_content_sha256",
                "training_ready",
            ]
            if any(column not in parquet.schema_arrow.names for column in columns):
                raise InstitutionalBooksMechanicalGateError(
                    "materialized book columns differ"
                )
            for batch in parquet.iter_batches(
                batch_size=32, columns=columns, use_threads=False
            ):
                for row in batch.to_pylist():
                    decision = _decision(row, source_path)
                    barcode = decision["barcode_src"]
                    if barcode in seen_barcodes:
                        raise InstitutionalBooksMechanicalGateError(
                            "materialized barcode overlaps"
                        )
                    seen_barcodes.add(barcode)
                    decision_counts[decision["decision"]] += 1
                    reason_counts.update(decision["reasons"])
                    ordered_decisions.append(decision["decision_sha256"])
                    handle.write(
                        json.dumps(decision, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    parent_rows += 1
            if parent_rows != output.get("rows"):
                raise InstitutionalBooksMechanicalGateError(
                    "materialized parent row coverage differs"
                )
            verified_parent_outputs.append(
                {
                    "parent_receipt_sha256": parent["receipt_sha256"],
                    "output_sha256": output["sha256"],
                    "rows": parent_rows,
                }
            )

    _atomic_jsonl(decision_path, emit)
    expected_rows = source_receipt.get("counts", {}).get("materialized_rows")
    if len(seen_barcodes) != expected_rows:
        raise InstitutionalBooksMechanicalGateError(
            "mechanical shard row coverage differs"
        )
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_private_book_mechanical_gate_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "materialized_root": str(materialized_root.resolve()),
            "receipt_sha256": source_receipt["receipt_sha256"],
            "rows": expected_rows,
            "verified_parent_outputs": len(verified_parent_outputs),
            "ordered_parent_outputs_sha256": canonical_sha256(
                verified_parent_outputs
            ),
        },
        "policy_sha256": POLICY_SHA256,
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "ordered_decisions_sha256": canonical_sha256(ordered_decisions),
        "decisions": {
            "path": str(decision_path.relative_to(root)),
            "rows": len(seen_barcodes),
            "bytes": decision_path.stat().st_size,
            "sha256": sha256_file(decision_path),
        },
        "all_nonpass_rows_excluded_from_direct_admission": True,
        "source_text_persisted": False,
        "semantic_admission_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(receipt_path, payload)
    return payload


def aggregate(
    materialized_root: Path,
    output_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Verify all shard decisions against the complete materializer aggregate."""

    if output.exists() or output.is_symlink():
        raise InstitutionalBooksMechanicalGateError("mechanical aggregate exists")
    materializer = _load_json(materialized_root / "aggregate.json")
    if not _valid_receipt(materializer, MATERIALIZER_AGGREGATE_SCHEMA):
        raise InstitutionalBooksMechanicalGateError(
            "materializer aggregate differs"
        )
    seen_barcodes: set[str] = set()
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    ordered_shards: list[str] = []
    for shard_index in range(logical_shards):
        root = output_root / "shards" / f"shard_{shard_index:05d}"
        receipt = _load_json(root / "receipt.json")
        source = _load_json(
            materialized_root
            / "shards"
            / f"shard_{shard_index:05d}"
            / "receipt.json"
        )
        if (
            not _valid_receipt(receipt, SHARD_SCHEMA)
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source", {}).get("receipt_sha256")
            != source.get("receipt_sha256")
        ):
            raise InstitutionalBooksMechanicalGateError(
                "mechanical shard receipt differs"
            )
        decisions = root / receipt["decisions"]["path"]
        if (
            not decisions.is_file()
            or decisions.is_symlink()
            or decisions.stat().st_nlink != 1
            or decisions.stat().st_size != receipt["decisions"]["bytes"]
            or sha256_file(decisions) != receipt["decisions"]["sha256"]
        ):
            raise InstitutionalBooksMechanicalGateError(
                "mechanical decision stream differs"
            )
        rows = 0
        local_counts: Counter[str] = Counter()
        local_reasons: Counter[str] = Counter()
        local_hashes: list[str] = []
        with decisions.open() as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise InstitutionalBooksMechanicalGateError(
                        "mechanical decision is invalid"
                    ) from error
                unsigned = {
                    key: value
                    for key, value in row.items()
                    if key != "decision_sha256"
                }
                barcode = row.get("barcode_src")
                if (
                    row.get("schema") != DECISION_SCHEMA
                    or row.get("policy_sha256") != POLICY_SHA256
                    or row.get("decision_sha256") != canonical_sha256(unsigned)
                    or not isinstance(barcode, str)
                    or barcode in seen_barcodes
                    or row.get("training_ready") is not False
                ):
                    raise InstitutionalBooksMechanicalGateError(
                        "mechanical decision differs"
                    )
                seen_barcodes.add(barcode)
                rows += 1
                local_counts[row["decision"]] += 1
                local_reasons.update(row["reasons"])
                local_hashes.append(row["decision_sha256"])
        if (
            rows != receipt["decisions"]["rows"]
            or dict(sorted(local_counts.items())) != receipt["decision_counts"]
            or dict(sorted(local_reasons.items())) != receipt["reason_counts"]
            or canonical_sha256(local_hashes)
            != receipt["ordered_decisions_sha256"]
        ):
            raise InstitutionalBooksMechanicalGateError(
                "mechanical shard accounting differs"
            )
        ordered_shards.append(receipt["receipt_sha256"])
        decision_counts.update(local_counts)
        reason_counts.update(local_reasons)
    expected_rows = materializer.get("counts", {}).get("materialized_rows")
    if len(seen_barcodes) != expected_rows:
        raise InstitutionalBooksMechanicalGateError(
            "mechanical aggregate row coverage differs"
        )
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_private_book_mechanical_gate",
        "source": {
            "materializer_receipt_sha256": materializer["receipt_sha256"],
            "rows": expected_rows,
        },
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(ordered_shards),
        },
        "policy_sha256": POLICY_SHA256,
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "all_rows_accounted": True,
        "all_nonpass_rows_excluded_from_direct_admission": True,
        "source_text_persisted": False,
        "semantic_admission_complete": False,
        "benchmark_decontamination_complete": False,
        "global_semantic_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    shard = children.add_parser("shard")
    aggregate_parser = children.add_parser("aggregate")
    for child in (shard, aggregate_parser):
        child.add_argument("--materialized-root", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        run_shard(
            args.materialized_root,
            args.output_root,
            args.logical_shards,
            args.shard_index,
        )
        if args.command == "shard"
        else aggregate(
            args.materialized_root,
            args.output_root,
            args.logical_shards,
            args.output,
        )
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
