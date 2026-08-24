"""Build a disk-backed exact and normalized-exact PleIAs production decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_production_descriptor_census import (
    AGGREGATE_SCHEMA as CENSUS_AGGREGATE_SCHEMA,
)
from sai.data.pleias_production_descriptor_census import (
    DESCRIPTOR_SCHEMA,
    NEAR_BOTTOM_K,
)
from sai.data.pleias_production_descriptor_census import (
    SHARD_SCHEMA as CENSUS_SHARD_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pleias-production-normalized-exact-dedup-v1"


class PleiasProductionNormalizedExactDedupError(RuntimeError):
    """Descriptor custody, normalized identity, or SQLite decision differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasProductionNormalizedExactDedupError("signed input is unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasProductionNormalizedExactDedupError(
            "signed input is invalid"
        ) from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("training_ready") is not False
    ):
        raise PleiasProductionNormalizedExactDedupError("signed input differs")
    return value


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_descriptor(row: dict[str, Any]) -> None:
    """Fail closed on any mutated or incomplete production descriptor."""

    unsigned = {key: value for key, value in row.items() if key != "descriptor_sha256"}
    sketch = row.get("near_dedup_bottom_k_u64")
    if (
        row.get("schema") != DESCRIPTOR_SCHEMA
        or row.get("training_ready") is not False
        or not _sha256(row.get("source_parent_sha256"))
        or not _sha256(row.get("source_row_identity_sha256"))
        or not _sha256(row.get("identifier_sha256"))
        or not _sha256(row.get("content_sha256"))
        or not _sha256(row.get("normalized_content_sha256"))
        or not isinstance(row.get("source_path"), str)
        or not row["source_path"]
        or isinstance(row.get("source_row_index"), bool)
        or not isinstance(row.get("source_row_index"), int)
        or row["source_row_index"] < 0
        or not isinstance(row.get("stratum"), str)
        or not row["stratum"]
        or isinstance(row.get("stratum_quality_floor_milli"), bool)
        or not isinstance(row.get("stratum_quality_floor_milli"), int)
        or not 0 <= row["stratum_quality_floor_milli"] <= 5_000
        or isinstance(row.get("stratum_quality_mean_milli"), bool)
        or not isinstance(row.get("stratum_quality_mean_milli"), int)
        or not row["stratum_quality_floor_milli"]
        <= row["stratum_quality_mean_milli"]
        <= 5_000
        or isinstance(row.get("text_utf8_bytes"), bool)
        or not isinstance(row.get("text_utf8_bytes"), int)
        or row["text_utf8_bytes"] <= 0
        or isinstance(row.get("token_count"), bool)
        or not isinstance(row.get("token_count"), int)
        or row["token_count"] <= 0
        or not isinstance(sketch, list)
        or not 1 <= len(sketch) <= NEAR_BOTTOM_K
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 2**64
            for value in sketch
        )
        or sketch != sorted(set(sketch))
        or row.get("descriptor_sha256") != canonical_sha256(unsigned)
    ):
        raise PleiasProductionNormalizedExactDedupError("descriptor row differs")


def _validated_shard(
    descriptor_root: Path,
    aggregate: dict[str, Any],
    logical_shards: int,
    shard_index: int,
) -> tuple[Path, dict[str, Any]]:
    root = descriptor_root / "shards" / f"shard_{shard_index:05d}"
    receipt = _load_signed(root / "receipt.json", CENSUS_SHARD_SCHEMA)
    data_path = root / receipt.get("output", {}).get("path", "")
    if (
        receipt.get("status")
        != "complete_nontraining_pleias_production_descriptor_census_shard"
        or receipt.get("logical_shards") != logical_shards
        or receipt.get("shard_index") != shard_index
        or receipt.get("source", {}).get("manifest_sha256")
        != aggregate.get("source", {}).get("manifest_sha256")
        or receipt.get("source", {}).get("policy_receipt_sha256")
        != aggregate.get("source", {}).get("policy_receipt_sha256")
        or receipt.get("source", {}).get("semantic_decision_receipt_sha256")
        != aggregate.get("source", {}).get("semantic_decision_receipt_sha256")
        or receipt.get("source_text_persisted") is not False
        or not data_path.is_file()
        or data_path.is_symlink()
        or data_path.stat().st_nlink != 1
        or data_path.stat().st_size != receipt.get("output", {}).get("bytes")
        or sha256_file(data_path) != receipt.get("output", {}).get("sha256")
    ):
        raise PleiasProductionNormalizedExactDedupError("descriptor shard differs")
    return data_path, receipt


def build_decision(
    descriptor_root: Path,
    aggregate_path: Path,
    output_root: Path,
    logical_shards: int,
) -> dict[str, Any]:
    """Choose the lowest stable identity per normalized full-document content."""

    if output_root.exists() or output_root.is_symlink() or logical_shards <= 0:
        raise PleiasProductionNormalizedExactDedupError("decision arguments differ")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasProductionNormalizedExactDedupError(
            "pyarrow is required"
        ) from error
    aggregate = _load_signed(aggregate_path, CENSUS_AGGREGATE_SCHEMA)
    if (
        aggregate.get("status")
        != "complete_nontraining_pleias_production_descriptor_census"
        or aggregate.get("shards", {}).get("logical_shards") != logical_shards
        or aggregate.get("complete_source_parent_coverage") is not True
        or aggregate.get("source_text_persisted") is not False
    ):
        raise PleiasProductionNormalizedExactDedupError("census aggregate differs")
    output_root.mkdir(parents=True)
    database_path = output_root / "normalized_exact_keep.sqlite3"
    temporary = output_root / f".keep.partial.{uuid.uuid4().hex}.sqlite3"
    connection = sqlite3.connect(temporary)
    source_rows = 0
    source_bytes = 0
    source_tokens = 0
    shard_receipts = []
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE full_content (content_sha256 TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE keep ("
            "normalized_content_sha256 TEXT PRIMARY KEY, "
            "source_row_identity_sha256 TEXT NOT NULL UNIQUE, "
            "content_sha256 TEXT NOT NULL, "
            "source_path TEXT NOT NULL, "
            "source_parent_sha256 TEXT NOT NULL, "
            "source_row_index INTEGER NOT NULL, "
            "stratum TEXT NOT NULL, "
            "stratum_quality_floor_milli INTEGER NOT NULL, "
            "stratum_quality_mean_milli INTEGER NOT NULL, "
            "text_utf8_bytes INTEGER NOT NULL, "
            "token_count INTEGER NOT NULL, "
            "descriptor_sha256 TEXT NOT NULL"
            ") WITHOUT ROWID"
        )
        for shard_index in range(logical_shards):
            data_path, receipt = _validated_shard(
                descriptor_root, aggregate, logical_shards, shard_index
            )
            rows = 0
            ordered = hashlib.sha256()
            parquet = pq.ParquetFile(data_path)
            for batch in parquet.iter_batches(batch_size=256, use_threads=False):
                for row in batch.to_pylist():
                    validate_descriptor(row)
                    ordered.update(bytes.fromhex(row["descriptor_sha256"]))
                    connection.execute(
                        "INSERT OR IGNORE INTO full_content VALUES (?)",
                        (row["content_sha256"],),
                    )
                    connection.execute(
                        "INSERT INTO keep VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                        "ON CONFLICT(normalized_content_sha256) DO UPDATE SET "
                        "source_row_identity_sha256="
                        "excluded.source_row_identity_sha256, "
                        "content_sha256=excluded.content_sha256, "
                        "source_path=excluded.source_path, "
                        "source_parent_sha256=excluded.source_parent_sha256, "
                        "source_row_index=excluded.source_row_index, "
                        "stratum=excluded.stratum, "
                        "stratum_quality_floor_milli="
                        "excluded.stratum_quality_floor_milli, "
                        "stratum_quality_mean_milli="
                        "excluded.stratum_quality_mean_milli, "
                        "text_utf8_bytes=excluded.text_utf8_bytes, "
                        "token_count=excluded.token_count, "
                        "descriptor_sha256=excluded.descriptor_sha256 "
                        "WHERE excluded.stratum_quality_floor_milli > "
                        "keep.stratum_quality_floor_milli OR "
                        "(excluded.stratum_quality_floor_milli = "
                        "keep.stratum_quality_floor_milli AND "
                        "excluded.stratum_quality_mean_milli > "
                        "keep.stratum_quality_mean_milli) OR "
                        "(excluded.stratum_quality_floor_milli = "
                        "keep.stratum_quality_floor_milli AND "
                        "excluded.stratum_quality_mean_milli = "
                        "keep.stratum_quality_mean_milli AND "
                        "excluded.source_row_identity_sha256 < "
                        "keep.source_row_identity_sha256)",
                        (
                            row["normalized_content_sha256"],
                            row["source_row_identity_sha256"],
                            row["content_sha256"],
                            row["source_path"],
                            row["source_parent_sha256"],
                            row["source_row_index"],
                            row["stratum"],
                            row["stratum_quality_floor_milli"],
                            row["stratum_quality_mean_milli"],
                            row["text_utf8_bytes"],
                            row["token_count"],
                            row["descriptor_sha256"],
                        ),
                    )
                    rows += 1
                    source_rows += 1
                    source_bytes += row["text_utf8_bytes"]
                    source_tokens += row["token_count"]
            if rows != receipt.get("counts", {}).get(
                "production_candidate_descriptors"
            ) or ordered.hexdigest() != receipt.get(
                "ordered_descriptor_digests_sha256"
            ):
                raise PleiasProductionNormalizedExactDedupError(
                    "descriptor shard coverage differs"
                )
            connection.commit()
            shard_receipts.append(receipt["receipt_sha256"])
        unique_full = connection.execute(
            "SELECT COUNT(*) FROM full_content"
        ).fetchone()[0]
        unique_normalized = connection.execute("SELECT COUNT(*) FROM keep").fetchone()[
            0
        ]
        retained_bytes, retained_tokens = connection.execute(
            "SELECT COALESCE(SUM(text_utf8_bytes), 0), "
            "COALESCE(SUM(token_count), 0) FROM keep"
        ).fetchone()
        connection.execute("DROP TABLE full_content")
        connection.execute(
            "CREATE INDEX keep_identity ON keep(source_row_identity_sha256)"
        )
        connection.execute("CREATE INDEX keep_stratum ON keep(stratum)")
        connection.execute(
            "CREATE INDEX keep_source_path ON keep(source_path, source_row_index)"
        )
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        os.replace(temporary, database_path)
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    expected = aggregate.get("totals", {})
    if (
        source_rows != expected.get("production_candidate_descriptors")
        or source_bytes != expected.get("production_candidate_text_utf8_bytes")
        or source_tokens != expected.get("production_candidate_source_tokens")
    ):
        raise PleiasProductionNormalizedExactDedupError(
            "descriptor aggregate accounting differs"
        )
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_pleias_production_normalized_exact_dedup",
        "source": {
            "descriptor_aggregate_file_sha256": sha256_file(aggregate_path),
            "descriptor_aggregate_receipt_sha256": aggregate["receipt_sha256"],
            "ordered_shard_receipts_sha256": canonical_sha256(shard_receipts),
        },
        "method": {
            "exact_key": "full_content_sha256",
            "normalized_exact_key": "NFKC_casefold_whitespace_collapse_sha256",
            "representative": (
                "highest_stratum_quality_floor_then_highest_stratum_quality_mean_"
                "then_lowest_source_row_identity_sha256"
            ),
            "sqlite_journal_mode": "DELETE",
            "sqlite_synchronous": "FULL",
        },
        "counts": {
            "source_rows": source_rows,
            "source_text_utf8_bytes": source_bytes,
            "source_tokens": source_tokens,
            "unique_full_content_rows": unique_full,
            "full_content_exact_duplicate_rows": source_rows - unique_full,
            "unique_normalized_content_rows": unique_normalized,
            "additional_normalized_exact_duplicate_rows": unique_full
            - unique_normalized,
            "retained_text_utf8_bytes": retained_bytes,
            "retained_tokens": retained_tokens,
        },
        "keep_database": {
            "path": database_path.name,
            "bytes": database_path.stat().st_size,
            "sha256": sha256_file(database_path),
            "rows": unique_normalized,
        },
        "decision_contains_source_text": False,
        "full_content_exact_deduplication_complete": True,
        "normalized_exact_deduplication_complete": True,
        "global_near_deduplication_complete": False,
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
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    args = parser.parse_args()
    result = build_decision(
        args.descriptor_root,
        args.aggregate,
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
