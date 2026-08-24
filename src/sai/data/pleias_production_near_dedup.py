"""Discover and collapse high-confidence PleIAs document near duplicates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import struct
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_production_descriptor_census import (
    AGGREGATE_SCHEMA as DESCRIPTOR_AGGREGATE_SCHEMA,
)
from sai.data.pleias_production_normalized_exact_dedup import (
    SCHEMA as EXACT_SCHEMA,
)
from sai.data.pleias_production_normalized_exact_dedup import (
    _load_signed,
    _validated_shard,
    validate_descriptor,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pleias-production-high-precision-near-dedup-v1"
BAND_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
MINIMUM_SKETCH_VALUES = 16
MINIMUM_SHARED_PPM = 750_000
MINIMUM_LENGTH_RATIO_PPM = 800_000
MAXIMUM_BUCKET_MEMBERS = 64


class PleiasProductionNearDedupError(RuntimeError):
    """Exact custody, sketch geometry, or near-duplicate decision differs."""


def _pack_sketch(values: list[int]) -> bytes:
    return struct.pack(f">{len(values)}Q", *values)


def _unpack_sketch(value: bytes) -> tuple[int, ...]:
    if not value or len(value) % 8:
        raise PleiasProductionNearDedupError("stored sketch differs")
    return struct.unpack(f">{len(value) // 8}Q", value)


def band_keys(values: list[int]) -> list[bytes]:
    """Return four exact two-fingerprint LSH keys for a complete sketch."""

    if len(values) < MINIMUM_SKETCH_VALUES or values != sorted(set(values)):
        return []
    return [
        struct.pack(">QQ", values[first], values[second])
        for first, second in BAND_PAIRS
    ]


def high_confidence_near_duplicate(
    first: tuple[int, ...] | list[int],
    second: tuple[int, ...] | list[int],
    first_bytes: int,
    second_bytes: int,
) -> bool:
    """Apply a conservative KMV-overlap and document-length decision."""

    if (
        len(first) < MINIMUM_SKETCH_VALUES
        or len(second) < MINIMUM_SKETCH_VALUES
        or first_bytes <= 0
        or second_bytes <= 0
    ):
        return False
    length_ppm = (
        min(first_bytes, second_bytes) * 1_000_000 // max(first_bytes, second_bytes)
    )
    if length_ppm < MINIMUM_LENGTH_RATIO_PPM:
        return False
    shared = len(set(first).intersection(second))
    required = math.ceil(min(len(first), len(second)) * MINIMUM_SHARED_PPM / 1_000_000)
    return shared >= required


def _root(connection: sqlite3.Connection, identity: str) -> str:
    path = []
    current = identity
    while True:
        row = connection.execute(
            "SELECT parent FROM forest WHERE identity=?", (current,)
        ).fetchone()
        if row is None:
            connection.execute("INSERT INTO forest VALUES (?, ?)", (current, current))
            root = current
            break
        parent = row[0]
        if parent == current:
            root = current
            break
        path.append(current)
        current = parent
    for member in path:
        connection.execute(
            "UPDATE forest SET parent=? WHERE identity=?", (root, member)
        )
    return root


def union_lowest(
    connection: sqlite3.Connection, first: str, second: str
) -> tuple[str, bool]:
    """Union two disk-backed components under their lowest stable identity."""

    first_root = _root(connection, first)
    second_root = _root(connection, second)
    if first_root == second_root:
        return first_root, False
    keep = min(first_root, second_root)
    drop = max(first_root, second_root)
    connection.execute("UPDATE forest SET parent=? WHERE identity=?", (keep, drop))
    return keep, True


def union_preferred(
    connection: sqlite3.Connection, first: str, second: str
) -> tuple[str, bool]:
    """Union a component under its strongest measured stratum representative."""

    first_root = _root(connection, first)
    second_root = _root(connection, second)
    if first_root == second_root:
        return first_root, False
    rows = connection.execute(
        "SELECT identity, quality_floor, quality_mean FROM docs "
        "WHERE identity IN (?, ?)",
        (first_root, second_root),
    ).fetchall()
    if len(rows) != 2:
        raise PleiasProductionNearDedupError("near-dedup quality rank differs")
    keep = min(rows, key=lambda row: (-row[1], -row[2], row[0]))[0]
    drop = second_root if keep == first_root else first_root
    connection.execute("UPDATE forest SET parent=? WHERE identity=?", (keep, drop))
    return keep, True


def _validate_exact_database(exact_root: Path, receipt: dict[str, Any]) -> Path:
    path = exact_root / receipt.get("keep_database", {}).get("path", "")
    if (
        receipt.get("status")
        != "complete_nontraining_pleias_production_normalized_exact_dedup"
        or receipt.get("normalized_exact_deduplication_complete") is not True
        or receipt.get("decision_contains_source_text") is not False
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != receipt.get("keep_database", {}).get("bytes")
        or sha256_file(path) != receipt.get("keep_database", {}).get("sha256")
    ):
        raise PleiasProductionNearDedupError("normalized-exact decision differs")
    return path


def build_decision(
    descriptor_root: Path,
    descriptor_aggregate_path: Path,
    exact_root: Path,
    output_root: Path,
    logical_shards: int,
) -> dict[str, Any]:
    """Build conservative near-duplicate drops from exact-retained descriptors."""

    if output_root.exists() or output_root.is_symlink() or logical_shards <= 0:
        raise PleiasProductionNearDedupError("near-dedup arguments differ")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasProductionNearDedupError("pyarrow is required") from error
    descriptor_aggregate = _load_signed(
        descriptor_aggregate_path, DESCRIPTOR_AGGREGATE_SCHEMA
    )
    exact_receipt = _load_signed(exact_root / "receipt.json", EXACT_SCHEMA)
    exact_database = _validate_exact_database(exact_root, exact_receipt)
    if descriptor_aggregate.get("shards", {}).get(
        "logical_shards"
    ) != logical_shards or exact_receipt.get("source", {}).get(
        "descriptor_aggregate_receipt_sha256"
    ) != descriptor_aggregate.get("receipt_sha256"):
        raise PleiasProductionNearDedupError("near-dedup lineage differs")
    output_root.mkdir(parents=True)
    database_path = output_root / "high_precision_near_drops.sqlite3"
    temporary = output_root / f".near.partial.{uuid.uuid4().hex}.sqlite3"
    connection = sqlite3.connect(temporary)
    exact_path = str(exact_database.resolve())
    shard_receipts = []
    eligible_rows = 0
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("ATTACH DATABASE ? AS exact", (exact_path,))
        connection.execute(
            "CREATE TABLE docs ("
            "identity TEXT PRIMARY KEY, bytes INTEGER NOT NULL, sketch BLOB NOT NULL, "
            "quality_floor INTEGER NOT NULL, quality_mean INTEGER NOT NULL"
            ") WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE buckets ("
            "band_key BLOB NOT NULL, identity TEXT NOT NULL, "
            "PRIMARY KEY(band_key, identity)"
            ") WITHOUT ROWID"
        )
        for shard_index in range(logical_shards):
            data_path, receipt = _validated_shard(
                descriptor_root,
                descriptor_aggregate,
                logical_shards,
                shard_index,
            )
            parquet = pq.ParquetFile(data_path)
            rows = 0
            ordered = hashlib.sha256()
            for batch in parquet.iter_batches(batch_size=256, use_threads=False):
                for row in batch.to_pylist():
                    validate_descriptor(row)
                    ordered.update(bytes.fromhex(row["descriptor_sha256"]))
                    rows += 1
                    identity = row["source_row_identity_sha256"]
                    keep = connection.execute(
                        "SELECT 1 FROM exact.keep WHERE source_row_identity_sha256=?",
                        (identity,),
                    ).fetchone()
                    if keep is None:
                        continue
                    keys = band_keys(row["near_dedup_bottom_k_u64"])
                    if not keys:
                        continue
                    sketch = _pack_sketch(row["near_dedup_bottom_k_u64"])
                    connection.execute(
                        "INSERT INTO docs VALUES (?, ?, ?, ?, ?)",
                        (
                            identity,
                            row["text_utf8_bytes"],
                            sketch,
                            row["stratum_quality_floor_milli"],
                            row["stratum_quality_mean_milli"],
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO buckets VALUES (?, ?)",
                        [(key, identity) for key in keys],
                    )
                    eligible_rows += 1
            if rows != receipt.get("counts", {}).get(
                "production_candidate_descriptors"
            ) or ordered.hexdigest() != receipt.get(
                "ordered_descriptor_digests_sha256"
            ):
                raise PleiasProductionNearDedupError(
                    "near-dedup descriptor coverage differs"
                )
            connection.commit()
            shard_receipts.append(receipt["receipt_sha256"])
        connection.execute("DETACH DATABASE exact")
        if sha256_file(exact_database) != exact_receipt["keep_database"]["sha256"]:
            raise PleiasProductionNearDedupError(
                "normalized-exact database mutated during replay"
            )
        connection.execute(
            "CREATE TABLE forest (identity TEXT PRIMARY KEY, parent TEXT NOT NULL) "
            "WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE compared ("
            "first TEXT NOT NULL, second TEXT NOT NULL, "
            "PRIMARY KEY(first, second)"
            ") WITHOUT ROWID"
        )
        candidate_buckets = 0
        skipped_high_fanout_buckets = 0
        compared_pairs = 0
        high_confidence_edges = 0
        union_edges = 0
        buckets = connection.execute(
            "SELECT band_key, COUNT(*) FROM buckets "
            "GROUP BY band_key HAVING COUNT(*) >= 2"
        )
        for key, members_count in buckets:
            if members_count > MAXIMUM_BUCKET_MEMBERS:
                skipped_high_fanout_buckets += 1
                continue
            candidate_buckets += 1
            members = connection.execute(
                "SELECT d.identity, d.bytes, d.sketch FROM docs d "
                "JOIN buckets b ON b.identity=d.identity WHERE b.band_key=? "
                "ORDER BY d.identity",
                (key,),
            ).fetchall()
            for first_index, first in enumerate(members):
                for second in members[first_index + 1 :]:
                    inserted = connection.execute(
                        "INSERT OR IGNORE INTO compared VALUES (?, ?)",
                        (first[0], second[0]),
                    ).rowcount
                    if not inserted:
                        continue
                    compared_pairs += 1
                    if not high_confidence_near_duplicate(
                        _unpack_sketch(first[2]),
                        _unpack_sketch(second[2]),
                        first[1],
                        second[1],
                    ):
                        continue
                    high_confidence_edges += 1
                    _root_value, changed = union_preferred(
                        connection, first[0], second[0]
                    )
                    union_edges += int(changed)
            if candidate_buckets % 10_000 == 0:
                connection.commit()
        connection.execute(
            "CREATE TABLE drops ("
            "source_row_identity_sha256 TEXT PRIMARY KEY, "
            "representative_source_row_identity_sha256 TEXT NOT NULL"
            ") WITHOUT ROWID"
        )
        forest_members = [
            row[0] for row in connection.execute("SELECT identity FROM forest")
        ]
        for identity in forest_members:
            representative = _root(connection, identity)
            if representative != identity:
                connection.execute(
                    "INSERT INTO drops VALUES (?, ?)",
                    (identity, representative),
                )
        affected_rows = len(forest_members)
        dropped_rows = connection.execute("SELECT COUNT(*) FROM drops").fetchone()[0]
        clusters = affected_rows - dropped_rows
        connection.execute("DROP TABLE buckets")
        connection.execute("DROP TABLE docs")
        connection.execute("DROP TABLE compared")
        connection.execute("DROP TABLE forest")
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        os.replace(temporary, database_path)
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_pleias_high_precision_near_dedup",
        "source": {
            "descriptor_aggregate_file_sha256": sha256_file(descriptor_aggregate_path),
            "descriptor_aggregate_receipt_sha256": descriptor_aggregate[
                "receipt_sha256"
            ],
            "normalized_exact_receipt_sha256": exact_receipt["receipt_sha256"],
            "ordered_shard_receipts_sha256": canonical_sha256(shard_receipts),
        },
        "method": {
            "candidate_bands": [list(pair) for pair in BAND_PAIRS],
            "minimum_sketch_values": MINIMUM_SKETCH_VALUES,
            "minimum_shared_sketch_ppm": MINIMUM_SHARED_PPM,
            "minimum_length_ratio_ppm": MINIMUM_LENGTH_RATIO_PPM,
            "maximum_bucket_members": MAXIMUM_BUCKET_MEMBERS,
            "representative": (
                "highest_stratum_quality_floor_then_highest_stratum_quality_mean_"
                "then_lowest_source_row_identity_sha256"
            ),
            "high_fanout_buckets_fail_open_to_later_global_pass": True,
        },
        "counts": {
            "normalized_exact_rows_with_complete_sketch": eligible_rows,
            "candidate_buckets": candidate_buckets,
            "skipped_high_fanout_buckets": skipped_high_fanout_buckets,
            "unique_compared_pairs": compared_pairs,
            "high_confidence_edges": high_confidence_edges,
            "component_union_edges": union_edges,
            "affected_rows": affected_rows,
            "near_duplicate_clusters": clusters,
            "dropped_rows": dropped_rows,
        },
        "drop_database": {
            "path": database_path.name,
            "bytes": database_path.stat().st_size,
            "sha256": sha256_file(database_path),
            "rows": dropped_rows,
        },
        "decision_contains_source_text": False,
        "high_precision_near_duplicate_pass_complete": True,
        "all_candidate_buckets_within_fanout_cap": skipped_high_fanout_buckets == 0,
        "global_near_deduplication_complete": False,
        "cross_source_near_deduplication_complete": False,
        "benchmark_decontamination_complete": False,
        "production_selection_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor-root", type=Path, required=True)
    parser.add_argument("--descriptor-aggregate", type=Path, required=True)
    parser.add_argument("--exact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    args = parser.parse_args()
    result = build_decision(
        args.descriptor_root,
        args.descriptor_aggregate,
        args.exact_root,
        args.output_root,
        args.logical_shards,
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
