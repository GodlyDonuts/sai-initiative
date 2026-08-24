"""Combine and verify all PleIAs parent-disjoint audit acquisition shards."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.pleias_parent_disjoint_audit_population import (
    EXPECTED_ROWS,
    SOURCE_ID,
)
from sai.data.reservoir_audit_population import SCHEMA, _write_jsonl
from sai.data.token_stream import canonical_sha256, sha256_file

AGGREGATE_SCHEMA = "sai-pleias-parent-disjoint-audit-aggregate-v1"


class PleiasParentDisjointAggregateError(RuntimeError):
    """Shard identity, file custody, or aggregate coverage differs."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PleiasParentDisjointAggregateError("shard JSON boundary differs")
    try:
        value = json.loads(path.read_text())
    except Exception as error:
        raise PleiasParentDisjointAggregateError("shard JSON differs") from error
    if not isinstance(value, dict):
        raise PleiasParentDisjointAggregateError("shard JSON is not an object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise PleiasParentDisjointAggregateError("shard JSONL boundary differs")
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as error:
                raise PleiasParentDisjointAggregateError(
                    f"shard JSONL row {line_number} differs"
                ) from error
            if not isinstance(row, dict):
                raise PleiasParentDisjointAggregateError(
                    "shard JSONL row is not an object"
                )
            rows.append(row)
    return rows


def build_aggregate(
    shards_root: Path,
    output_root: Path,
    *,
    logical_shards: int = 8,
) -> dict[str, Any]:
    """Replay each shard and emit one globally ordered population."""

    if (
        not shards_root.is_dir()
        or shards_root.is_symlink()
        or logical_shards != 8
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise PleiasParentDisjointAggregateError("aggregate boundary differs")
    candidates_by_identity: dict[str, dict[str, Any]] = {}
    lineage_by_ordinal: dict[int, dict[str, Any]] = {}
    shard_receipts = []
    parent_identities = set()
    for shard_index in range(logical_shards):
        root = shards_root / f"shard_{shard_index:03d}"
        receipt_path = root / "receipt.json"
        receipt = _load_json(receipt_path)
        unsigned = {
            key: value
            for key, value in receipt.items()
            if key != "receipt_sha256"
        }
        population = receipt.get("population", {})
        lineage_descriptor = receipt.get("lineage", {})
        if (
            receipt.get("schema") != SCHEMA
            or receipt.get("status") != "complete"
            or receipt.get("source_id") != SOURCE_ID
            or receipt.get("global_plan_rows") != EXPECTED_ROWS
            or receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("acquisition_mode") != "full_verified_parent"
            or receipt.get("maximum_simultaneous_parent_files") != 1
            or receipt.get("temporary_parent_removed_after_each_row") is not True
            or receipt.get("fully_verified_parent_files") != EXPECTED_ROWS // 8
            or receipt.get("benchmark_decontamination_complete") is not False
            or receipt.get("hermes_judgments_complete") is not False
            or receipt.get("training_ready") is not False
            or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
            or population.get("rows") != EXPECTED_ROWS // 8
            or lineage_descriptor.get("rows") != EXPECTED_ROWS // 8
        ):
            raise PleiasParentDisjointAggregateError("shard receipt differs")
        candidate_path = root / population.get("path", "")
        lineage_path = root / lineage_descriptor.get("path", "")
        if (
            sha256_file(candidate_path) != population.get("sha256")
            or candidate_path.stat().st_size != population.get("bytes")
            or sha256_file(lineage_path) != lineage_descriptor.get("sha256")
            or lineage_path.stat().st_size != lineage_descriptor.get("bytes")
        ):
            raise PleiasParentDisjointAggregateError("shard file identity differs")
        candidates = _jsonl(candidate_path)
        lineage = _jsonl(lineage_path)
        if (
            len(candidates) != EXPECTED_ROWS // 8
            or len(lineage) != EXPECTED_ROWS // 8
        ):
            raise PleiasParentDisjointAggregateError("shard row count differs")
        for candidate in candidates:
            identity = candidate.get("candidate_identity_sha256")
            if not isinstance(identity, str) or identity in candidates_by_identity:
                raise PleiasParentDisjointAggregateError(
                    "candidate identity custody differs"
                )
            candidates_by_identity[identity] = candidate
        for row in lineage:
            ordinal = row.get("ordinal")
            identity = row.get("candidate_identity_sha256")
            parent = (row.get("repository"), row.get("revision"), row.get("path"))
            if (
                not isinstance(ordinal, int)
                or ordinal % logical_shards != shard_index
                or ordinal in lineage_by_ordinal
                or identity not in candidates_by_identity
                or parent in parent_identities
                or row.get("full_file_content_verified") is not True
            ):
                raise PleiasParentDisjointAggregateError("lineage custody differs")
            lineage_by_ordinal[ordinal] = row
            parent_identities.add(parent)
        shard_receipts.append(
            {
                "shard_index": shard_index,
                "receipt_file_sha256": sha256_file(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
                "population_sha256": population["sha256"],
                "lineage_sha256": lineage_descriptor["sha256"],
            }
        )
    if (
        len(candidates_by_identity) != EXPECTED_ROWS
        or len(lineage_by_ordinal) != EXPECTED_ROWS
        or len(parent_identities) != EXPECTED_ROWS
        or sorted(lineage_by_ordinal) != list(range(EXPECTED_ROWS))
    ):
        raise PleiasParentDisjointAggregateError("global shard coverage differs")
    ordered_lineage = [lineage_by_ordinal[index] for index in range(EXPECTED_ROWS)]
    ordered_candidates = [
        candidates_by_identity[row["candidate_identity_sha256"]]
        for row in ordered_lineage
    ]
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise PleiasParentDisjointAggregateError("aggregate temporary path differs")
    temporary.mkdir(parents=True)
    try:
        candidate_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        receipt_path = temporary / "receipt.json"
        _write_jsonl(candidate_path, ordered_candidates)
        _write_jsonl(lineage_path, ordered_lineage)
        by_stratum = Counter(row["stratum"] for row in ordered_lineage)
        receipt = {
            "schema": AGGREGATE_SCHEMA,
            "status": "complete_parent_disjoint_acquisition",
            "source_id": SOURCE_ID,
            "logical_shards": logical_shards,
            "shards": shard_receipts,
            "population": {
                "path": candidate_path.name,
                "rows": EXPECTED_ROWS,
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
                "ordered_identities_sha256": canonical_sha256(
                    [row["candidate_identity_sha256"] for row in ordered_candidates]
                ),
            },
            "lineage": {
                "path": lineage_path.name,
                "rows": EXPECTED_ROWS,
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
                "ordered_rows_sha256": canonical_sha256(ordered_lineage),
            },
            "by_stratum": dict(sorted(by_stratum.items())),
            "unique_parent_files": EXPECTED_ROWS,
            "fully_verified_parent_files": EXPECTED_ROWS,
            "maximum_simultaneous_parent_files_per_shard": 1,
            "maximum_concurrent_parent_files": logical_shards,
            "benchmark_decontamination_complete": False,
            "hermes_judgments_complete": False,
            "source_wide_yield_established": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _write_jsonl(receipt_path, [receipt])
        os.replace(temporary, output_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, default=8)
    args = parser.parse_args()
    result = build_aggregate(
        args.shards_root,
        args.output_root,
        logical_shards=args.logical_shards,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
