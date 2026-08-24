"""Materialize only mechanically clean private Institutional Books rows."""

from __future__ import annotations

import argparse
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
    PARENT_SCHEMA,
    _load_json,
    _valid_receipt,
)
from sai.data.institutional_books_materializer import (
    SHARD_SCHEMA as MATERIALIZER_SHARD_SCHEMA,
)
from sai.data.institutional_books_mechanical_gate import (
    AGGREGATE_SCHEMA as MECHANICAL_AGGREGATE_SCHEMA,
)
from sai.data.institutional_books_mechanical_gate import (
    DECISION_SCHEMA,
)
from sai.data.institutional_books_mechanical_gate import (
    SHARD_SCHEMA as MECHANICAL_SHARD_SCHEMA,
)
from sai.data.source_quality_gate import POLICY_SHA256
from sai.data.token_stream import canonical_sha256, sha256_file

SHARD_SCHEMA = "sai-institutional-books-mechanical-filter-shard-v1"
AGGREGATE_SCHEMA = "sai-institutional-books-mechanical-filter-aggregate-v1"
PASS_DECISION = "pass_mechanical_gate"


class InstitutionalBooksMechanicalFilterError(RuntimeError):
    """Private source, decision, or filtered output custody differs."""


def _decisions(root: Path, receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    descriptor = receipt.get("decisions")
    if not isinstance(descriptor, dict):
        raise InstitutionalBooksMechanicalFilterError(
            "mechanical decisions differ"
        )
    path = root / descriptor.get("path", "")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise InstitutionalBooksMechanicalFilterError(
            "mechanical decision stream differs"
        )
    rows: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    ordered = []
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise InstitutionalBooksMechanicalFilterError(
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
                or barcode in rows
                or row.get("training_ready") is not False
            ):
                raise InstitutionalBooksMechanicalFilterError(
                    "mechanical decision differs"
                )
            rows[barcode] = row
            counts[row["decision"]] += 1
            ordered.append(row["decision_sha256"])
    if (
        len(rows) != descriptor.get("rows")
        or dict(sorted(counts.items())) != receipt.get("decision_counts")
        or canonical_sha256(ordered) != receipt.get("ordered_decisions_sha256")
    ):
        raise InstitutionalBooksMechanicalFilterError(
            "mechanical decision accounting differs"
        )
    return rows


def _atomic_filtered_parquet(
    source_root: Path,
    decisions: dict[str, dict[str, Any]],
    output: Path,
) -> tuple[int, int, list[dict[str, Any]]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksMechanicalFilterError(
            "pyarrow is required"
        ) from error
    if output.exists() or output.is_symlink():
        raise InstitutionalBooksMechanicalFilterError("filtered output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.partial.{uuid.uuid4().hex}"
    writer = None
    seen: set[str] = set()
    retained = 0
    sources = []
    try:
        for parent_path in sorted((source_root / "parents").glob("parent_*.json")):
            parent = _load_json(parent_path)
            if not _valid_receipt(parent, PARENT_SCHEMA):
                raise InstitutionalBooksMechanicalFilterError(
                    "materialized parent differs"
                )
            descriptor = parent.get("output")
            if descriptor is None:
                continue
            path = source_root / descriptor["path"]
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or path.stat().st_size != descriptor.get("bytes")
                or sha256_file(path) != descriptor.get("sha256")
            ):
                raise InstitutionalBooksMechanicalFilterError(
                    "materialized parent output differs"
                )
            source_rows = 0
            retained_before = retained
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=32, use_threads=False):
                barcodes = batch.column(
                    batch.schema.get_field_index("barcode_src")
                ).to_pylist()
                mask = []
                for barcode in barcodes:
                    if not isinstance(barcode, str) or barcode in seen:
                        raise InstitutionalBooksMechanicalFilterError(
                            "materialized barcode overlaps"
                        )
                    decision = decisions.get(barcode)
                    if decision is None:
                        raise InstitutionalBooksMechanicalFilterError(
                            "materialized barcode lacks decision"
                        )
                    seen.add(barcode)
                    source_rows += 1
                    keep = decision["decision"] == PASS_DECISION
                    mask.append(keep)
                    retained += keep
                filtered = batch.filter(pa.array(mask))
                if filtered.num_rows:
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary, filtered.schema, compression="zstd"
                        )
                    elif filtered.schema != writer.schema:
                        raise InstitutionalBooksMechanicalFilterError(
                            "filtered schema differs"
                        )
                    writer.write_batch(filtered)
            sources.append(
                {
                    "parent_receipt_sha256": parent["receipt_sha256"],
                    "output_sha256": descriptor["sha256"],
                    "source_rows": source_rows,
                    "retained_rows": retained - retained_before,
                }
            )
        if writer is not None:
            writer.close()
            writer = None
            os.replace(temporary, output)
        elif temporary.exists():
            temporary.unlink()
        if seen != set(decisions):
            raise InstitutionalBooksMechanicalFilterError(
                "filtered decision coverage differs"
            )
        return len(seen), retained, sources
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise


def run_shard(
    materialized_root: Path,
    mechanical_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    """Filter one exact materializer shard using its complete decisions."""

    if not 0 <= shard_index < logical_shards:
        raise InstitutionalBooksMechanicalFilterError("filter shard differs")
    source_root = materialized_root / "shards" / f"shard_{shard_index:05d}"
    decision_root = mechanical_root / "shards" / f"shard_{shard_index:05d}"
    materializer = _load_json(source_root / "receipt.json")
    mechanical = _load_json(decision_root / "receipt.json")
    if (
        not _valid_receipt(materializer, MATERIALIZER_SHARD_SCHEMA)
        or not _valid_receipt(mechanical, MECHANICAL_SHARD_SCHEMA)
        or materializer.get("logical_shards") != logical_shards
        or materializer.get("shard_index") != shard_index
        or mechanical.get("logical_shards") != logical_shards
        or mechanical.get("shard_index") != shard_index
        or mechanical.get("source", {}).get("receipt_sha256")
        != materializer["receipt_sha256"]
    ):
        raise InstitutionalBooksMechanicalFilterError("filter inputs differ")
    root = output_root / "shards" / f"shard_{shard_index:05d}"
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        existing = _load_json(receipt_path)
        descriptor = existing.get("output")
        output_valid = descriptor is None
        if isinstance(descriptor, dict):
            existing_output = root / descriptor.get("path", "")
            output_valid = (
                existing_output.is_file()
                and not existing_output.is_symlink()
                and existing_output.stat().st_nlink == 1
                and existing_output.stat().st_size == descriptor.get("bytes")
                and sha256_file(existing_output) == descriptor.get("sha256")
            )
        if (
            not _valid_receipt(existing, SHARD_SCHEMA)
            or existing.get("mechanical_receipt_sha256")
            != mechanical["receipt_sha256"]
            or not output_valid
        ):
            raise InstitutionalBooksMechanicalFilterError(
                "existing filter shard differs"
            )
        return existing
    decisions = _decisions(decision_root, mechanical)
    output_path = root / "data.parquet"
    source_rows, retained_rows, sources = _atomic_filtered_parquet(
        source_root, decisions, output_path
    )
    expected_retained = mechanical.get("decision_counts", {}).get(PASS_DECISION, 0)
    if source_rows != len(decisions) or retained_rows != expected_retained:
        raise InstitutionalBooksMechanicalFilterError(
            "filtered shard accounting differs"
        )
    payload = {
        "schema": SHARD_SCHEMA,
        "status": "complete_nontraining_private_book_mechanical_filter_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "materializer_receipt_sha256": materializer["receipt_sha256"],
        "mechanical_receipt_sha256": mechanical["receipt_sha256"],
        "policy_sha256": POLICY_SHA256,
        "source_rows": source_rows,
        "retained_rows": retained_rows,
        "excluded_rows": source_rows - retained_rows,
        "ordered_parent_outputs_sha256": canonical_sha256(sources),
        "output": (
            {
                "path": output_path.name,
                "rows": retained_rows,
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
            }
            if retained_rows
            else None
        ),
        "nonpass_source_rows_copied": False,
        "source_text_persisted_in_private_output": retained_rows > 0,
        "benchmark_decontamination_complete": False,
        "global_semantic_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(receipt_path, payload)
    return payload


def aggregate(
    materialized_root: Path,
    mechanical_root: Path,
    output_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Verify complete filtered row and byte custody across all shards."""

    if output.exists() or output.is_symlink():
        raise InstitutionalBooksMechanicalFilterError("filter aggregate exists")
    materializer = _load_json(materialized_root / "aggregate.json")
    mechanical = _load_json(mechanical_root / "aggregate.json")
    if (
        not _valid_receipt(materializer, MATERIALIZER_AGGREGATE_SCHEMA)
        or not _valid_receipt(mechanical, MECHANICAL_AGGREGATE_SCHEMA)
        or mechanical.get("source", {}).get("materializer_receipt_sha256")
        != materializer["receipt_sha256"]
        or mechanical.get("shards", {}).get("logical_shards")
        != logical_shards
    ):
        raise InstitutionalBooksMechanicalFilterError(
            "filter aggregate inputs differ"
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksMechanicalFilterError(
            "pyarrow is required"
        ) from error
    seen: set[str] = set()
    totals: Counter[str] = Counter()
    shard_receipts = []
    for shard_index in range(logical_shards):
        root = output_root / "shards" / f"shard_{shard_index:05d}"
        receipt = _load_json(root / "receipt.json")
        source_receipt = _load_json(
            materialized_root
            / "shards"
            / f"shard_{shard_index:05d}"
            / "receipt.json"
        )
        mechanical_receipt = _load_json(
            mechanical_root
            / "shards"
            / f"shard_{shard_index:05d}"
            / "receipt.json"
        )
        if (
            not _valid_receipt(receipt, SHARD_SCHEMA)
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or not _valid_receipt(source_receipt, MATERIALIZER_SHARD_SCHEMA)
            or not _valid_receipt(mechanical_receipt, MECHANICAL_SHARD_SCHEMA)
            or receipt.get("materializer_receipt_sha256")
            != source_receipt["receipt_sha256"]
            or receipt.get("mechanical_receipt_sha256")
            != mechanical_receipt["receipt_sha256"]
            or mechanical_receipt.get("source", {}).get("receipt_sha256")
            != source_receipt["receipt_sha256"]
        ):
            raise InstitutionalBooksMechanicalFilterError(
                "filter shard receipt differs"
            )
        shard_receipts.append(receipt["receipt_sha256"])
        descriptor = receipt.get("output")
        if descriptor is not None:
            path = root / descriptor["path"]
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_nlink != 1
                or path.stat().st_size != descriptor.get("bytes")
                or sha256_file(path) != descriptor.get("sha256")
            ):
                raise InstitutionalBooksMechanicalFilterError(
                    "filtered shard bytes differ"
                )
            parquet = pq.ParquetFile(path)
            rows = 0
            for batch in parquet.iter_batches(
                batch_size=1024, columns=["barcode_src"], use_threads=False
            ):
                for barcode in batch.column(0).to_pylist():
                    if not isinstance(barcode, str) or barcode in seen:
                        raise InstitutionalBooksMechanicalFilterError(
                            "filtered barcode overlaps"
                        )
                    seen.add(barcode)
                    rows += 1
            if rows != descriptor.get("rows"):
                raise InstitutionalBooksMechanicalFilterError(
                    "filtered shard row coverage differs"
                )
            totals["output_files"] += 1
            totals["output_bytes"] += descriptor["bytes"]
        totals["source_rows"] += receipt["source_rows"]
        totals["retained_rows"] += receipt["retained_rows"]
        totals["excluded_rows"] += receipt["excluded_rows"]
    expected_source = mechanical.get("source", {}).get("rows")
    expected_retained = mechanical.get("decision_counts", {}).get(
        PASS_DECISION, 0
    )
    if (
        totals["source_rows"] != expected_source
        or totals["retained_rows"] != expected_retained
        or len(seen) != expected_retained
        or totals["excluded_rows"] != expected_source - expected_retained
    ):
        raise InstitutionalBooksMechanicalFilterError(
            "filter aggregate accounting differs"
        )
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_private_book_mechanical_filter",
        "materializer_receipt_sha256": materializer["receipt_sha256"],
        "mechanical_receipt_sha256": mechanical["receipt_sha256"],
        "policy_sha256": POLICY_SHA256,
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(shard_receipts),
        },
        "counts": dict(sorted(totals.items())),
        "ordered_retained_barcodes_sha256": canonical_sha256(sorted(seen)),
        "nonpass_source_rows_copied": False,
        "source_text_persisted_in_private_output": bool(seen),
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
        child.add_argument("--mechanical-root", type=Path, required=True)
        child.add_argument("--output-root", type=Path, required=True)
        child.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        run_shard(
            args.materialized_root,
            args.mechanical_root,
            args.output_root,
            args.logical_shards,
            args.shard_index,
        )
        if args.command == "shard"
        else aggregate(
            args.materialized_root,
            args.mechanical_root,
            args.output_root,
            args.logical_shards,
            args.output,
        )
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
