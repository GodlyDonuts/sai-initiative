"""Globally exact-deduplicate decontaminated PleIAs candidate shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_full_candidate_decontamination import (
    AGGREGATE_SCHEMA as DECONTAMINATION_AGGREGATE_SCHEMA,
)
from sai.data.pleias_full_candidate_decontamination import (
    SHARD_SCHEMA as DECONTAMINATION_SHARD_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

DECISION_SCHEMA = "sai-pleias-global-exact-dedup-decision-v1"
FILTER_SHARD_SCHEMA = "sai-pleias-global-exact-dedup-filter-shard-v1"
AGGREGATE_SCHEMA = "sai-pleias-global-exact-dedup-aggregate-v1"


class PleiasGlobalExactDedupError(RuntimeError):
    """Decontamination custody, global hash index, or filter replay differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasGlobalExactDedupError("signed input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasGlobalExactDedupError("signed input is invalid") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("training_ready") is not False
    ):
        raise PleiasGlobalExactDedupError("signed input differs")
    return payload


def _validated_decontamination_shard(
    shards_root: Path,
    shard_index: int,
    logical_shards: int,
    aggregate: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    root = shards_root / f"shard_{shard_index:05d}"
    receipt = _load_signed(root / "receipt.json", DECONTAMINATION_SHARD_SCHEMA)
    output = root / receipt.get("output", {}).get("path", "")
    index = root / receipt.get("global_exact_dedup_index", {}).get("path", "")
    if (
        receipt.get("logical_shards") != logical_shards
        or receipt.get("shard_index") != shard_index
        or receipt.get("source", {}).get("bounded_aggregate_receipt_sha256")
        != aggregate.get("source", {}).get("bounded_aggregate_receipt_sha256")
        or not output.is_file()
        or output.is_symlink()
        or output.stat().st_nlink != 1
        or output.stat().st_size != receipt.get("output", {}).get("bytes")
        or sha256_file(output) != receipt.get("output", {}).get("sha256")
        or not index.is_file()
        or index.is_symlink()
        or index.stat().st_nlink != 1
        or index.stat().st_size
        != receipt.get("global_exact_dedup_index", {}).get("bytes")
        or sha256_file(index)
        != receipt.get("global_exact_dedup_index", {}).get("sha256")
    ):
        raise PleiasGlobalExactDedupError("decontamination shard differs")
    return output, index, receipt


def _valid_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def build_decision(
    decontamination_root: Path,
    decontamination_aggregate_path: Path,
    output_root: Path,
    logical_shards: int,
) -> dict[str, Any]:
    """Build a disk-backed one-content-hash/one-identity keep decision."""

    if output_root.exists() or output_root.is_symlink():
        raise PleiasGlobalExactDedupError("exact-dedup output exists")
    aggregate = _load_signed(
        decontamination_aggregate_path, DECONTAMINATION_AGGREGATE_SCHEMA
    )
    if (
        aggregate.get("status")
        != "complete_nontraining_pleias_full_candidate_decontamination"
        or aggregate.get("shards", {}).get("logical_shards") != logical_shards
        or aggregate.get("full_candidate_benchmark_decontamination_complete")
        is not True
        or aggregate.get("global_exact_deduplication_complete") is not False
    ):
        raise PleiasGlobalExactDedupError("decontamination aggregate differs")
    output_root.mkdir(parents=True)
    database_path = output_root / "global_exact_keep.sqlite3"
    temporary = output_root / f".keep.partial.{uuid.uuid4().hex}.sqlite3"
    connection = sqlite3.connect(temporary)
    source_rows = 0
    shard_receipts = []
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            "CREATE TABLE keep ("
            "content_sha256 TEXT PRIMARY KEY, "
            "source_row_identity_sha256 TEXT NOT NULL UNIQUE, "
            "shard_index INTEGER NOT NULL, "
            "source_row_index INTEGER NOT NULL, "
            "stratum TEXT NOT NULL"
            ") WITHOUT ROWID"
        )
        for shard_index in range(logical_shards):
            _output, index, receipt = _validated_decontamination_shard(
                decontamination_root / "shards",
                shard_index,
                logical_shards,
                aggregate,
            )
            digest = hashlib.sha256()
            rows = 0
            with index.open() as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise PleiasGlobalExactDedupError(
                            "exact-dedup index is invalid"
                        ) from error
                    if (
                        not isinstance(row, dict)
                        or set(row)
                        != {
                            "content_sha256",
                            "source_row_identity_sha256",
                            "shard_index",
                            "source_row_index",
                            "stratum",
                        }
                        or not _valid_hash(row["content_sha256"])
                        or not _valid_hash(row["source_row_identity_sha256"])
                        or row["shard_index"] != shard_index
                        or isinstance(row["source_row_index"], bool)
                        or not isinstance(row["source_row_index"], int)
                        or row["source_row_index"] < 0
                        or not isinstance(row["stratum"], str)
                        or not row["stratum"]
                    ):
                        raise PleiasGlobalExactDedupError(
                            "exact-dedup index row differs"
                        )
                    digest.update(bytes.fromhex(canonical_sha256(row)))
                    connection.execute(
                        "INSERT INTO keep VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(content_sha256) DO UPDATE SET "
                        "source_row_identity_sha256="
                        "excluded.source_row_identity_sha256, "
                        "shard_index=excluded.shard_index, "
                        "source_row_index=excluded.source_row_index, "
                        "stratum=excluded.stratum "
                        "WHERE excluded.source_row_identity_sha256 < "
                        "keep.source_row_identity_sha256",
                        (
                            row["content_sha256"],
                            row["source_row_identity_sha256"],
                            row["shard_index"],
                            row["source_row_index"],
                            row["stratum"],
                        ),
                    )
                    rows += 1
                    source_rows += 1
            descriptor = receipt["global_exact_dedup_index"]
            if (
                rows != descriptor["rows"]
                or digest.hexdigest() != descriptor["ordered_row_digests_sha256"]
            ):
                raise PleiasGlobalExactDedupError("exact-dedup index coverage differs")
            connection.commit()
            shard_receipts.append(receipt["receipt_sha256"])
        unique_rows = connection.execute("SELECT COUNT(*) FROM keep").fetchone()[0]
        connection.execute("CREATE INDEX keep_shard ON keep(shard_index)")
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        os.replace(temporary, database_path)
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    if source_rows != aggregate.get("totals", {}).get("retained_candidates"):
        raise PleiasGlobalExactDedupError("exact-dedup source coverage differs")
    payload = {
        "schema": DECISION_SCHEMA,
        "status": "complete_nontraining_pleias_global_exact_dedup_decision",
        "source": {
            "decontamination_aggregate_file_sha256": sha256_file(
                decontamination_aggregate_path
            ),
            "decontamination_aggregate_receipt_sha256": aggregate["receipt_sha256"],
            "ordered_shard_receipts_sha256": canonical_sha256(shard_receipts),
        },
        "method": {
            "duplicate_key": "full_content_sha256",
            "representative": "lowest_source_row_identity_sha256",
            "sqlite_journal_mode": "DELETE",
            "sqlite_synchronous": "FULL",
        },
        "counts": {
            "source_rows": source_rows,
            "unique_content_rows": unique_rows,
            "global_exact_duplicate_rows": source_rows - unique_rows,
        },
        "keep_database": {
            "path": database_path.name,
            "bytes": database_path.stat().st_size,
            "sha256": sha256_file(database_path),
            "rows": unique_rows,
        },
        "decision_contains_source_text": False,
        "global_exact_deduplication_decision_complete": True,
        "global_near_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def run_filter_shard(
    decontamination_root: Path,
    decontamination_aggregate_path: Path,
    decision_root: Path,
    output_root: Path,
    logical_shards: int,
    shard_index: int,
) -> dict[str, Any]:
    """Rewrite one shard to exactly the globally selected content identities."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not 0 <= shard_index < logical_shards
    ):
        raise PleiasGlobalExactDedupError("filter shard arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasGlobalExactDedupError("pyarrow is required") from error
    aggregate = _load_signed(
        decontamination_aggregate_path, DECONTAMINATION_AGGREGATE_SCHEMA
    )
    decision = _load_signed(decision_root / "receipt.json", DECISION_SCHEMA)
    database = decision_root / decision.get("keep_database", {}).get("path", "")
    if (
        decision.get("source", {}).get("decontamination_aggregate_receipt_sha256")
        != aggregate["receipt_sha256"]
        or not database.is_file()
        or database.is_symlink()
        or database.stat().st_nlink != 1
        or database.stat().st_size != decision["keep_database"]["bytes"]
        or sha256_file(database) != decision["keep_database"]["sha256"]
    ):
        raise PleiasGlobalExactDedupError("exact-dedup decision differs")
    source_path, _index, source_receipt = _validated_decontamination_shard(
        decontamination_root / "shards", shard_index, logical_shards, aggregate
    )
    uri = f"file:{database}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    keep = {
        row[0]
        for row in connection.execute(
            "SELECT source_row_identity_sha256 FROM keep WHERE shard_index=?",
            (shard_index,),
        )
    }
    connection.close()
    source = pq.ParquetFile(source_path)
    output_root.mkdir(parents=True)
    output_path = output_root / "exact_deduplicated_candidates.parquet"
    temporary = output_root / f".candidates.partial.{uuid.uuid4().hex}.parquet"
    writer = pq.ParquetWriter(temporary, source.schema_arrow, compression="zstd")
    counts = Counter()
    retained_hashes = []
    retained_text_bytes = 0
    try:
        for batch in source.iter_batches(batch_size=32, use_threads=False):
            rows = []
            for row in batch.to_pylist():
                counts["source_rows"] += 1
                identity = row.get("source_row_identity_sha256")
                if not _valid_hash(identity):
                    raise PleiasGlobalExactDedupError(
                        "exact-dedup source identity differs"
                    )
                if identity not in keep:
                    counts["global_exact_duplicate_rows"] += 1
                    continue
                keep.remove(identity)
                rows.append(row)
                counts["retained_rows"] += 1
                retained_hashes.append(identity)
                retained_text_bytes += len(row["text"].encode())
            if rows:
                writer.write_table(
                    pa.Table.from_pylist(rows, schema=source.schema_arrow)
                )
        if keep:
            raise PleiasGlobalExactDedupError("exact-dedup keep identities are missing")
        writer.close()
        os.replace(temporary, output_path)
    except BaseException:
        writer.close()
        temporary.unlink(missing_ok=True)
        raise
    payload = {
        "schema": FILTER_SHARD_SCHEMA,
        "status": "complete_nontraining_pleias_global_exact_dedup_filter_shard",
        "logical_shards": logical_shards,
        "shard_index": shard_index,
        "source": {
            "decontamination_shard_receipt_sha256": source_receipt["receipt_sha256"],
            "exact_dedup_decision_receipt_sha256": decision["receipt_sha256"],
        },
        "counts": dict(sorted(counts.items())),
        "retained_text_utf8_bytes": retained_text_bytes,
        "ordered_retained_identities_sha256": canonical_sha256(retained_hashes),
        "output": {
            "path": output_path.name,
            "rows": counts["retained_rows"],
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "global_exact_deduplication_complete": True,
        "global_near_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def aggregate_filters(
    decontamination_aggregate_path: Path,
    decision_root: Path,
    filters_root: Path,
    logical_shards: int,
    output: Path,
) -> dict[str, Any]:
    """Seal globally exact-deduplicated shard custody and row accounting."""

    if output.exists() or output.is_symlink():
        raise PleiasGlobalExactDedupError("filter aggregate output exists")
    source = _load_signed(
        decontamination_aggregate_path, DECONTAMINATION_AGGREGATE_SCHEMA
    )
    decision = _load_signed(decision_root / "receipt.json", DECISION_SCHEMA)
    totals: Counter[str] = Counter()
    receipts = []
    for shard_index in range(logical_shards):
        root = filters_root / f"shard_{shard_index:05d}"
        receipt = _load_signed(root / "receipt.json", FILTER_SHARD_SCHEMA)
        path = root / receipt.get("output", {}).get("path", "")
        if (
            receipt.get("logical_shards") != logical_shards
            or receipt.get("shard_index") != shard_index
            or receipt.get("source", {}).get("exact_dedup_decision_receipt_sha256")
            != decision["receipt_sha256"]
            or not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.stat().st_size != receipt["output"]["bytes"]
            or sha256_file(path) != receipt["output"]["sha256"]
        ):
            raise PleiasGlobalExactDedupError("filter shard differs")
        for key, value in receipt["counts"].items():
            totals[key] += value
        totals["retained_text_utf8_bytes"] += receipt["retained_text_utf8_bytes"]
        totals["output_bytes"] += receipt["output"]["bytes"]
        receipts.append(receipt["receipt_sha256"])
    if (
        totals["source_rows"] != source.get("totals", {}).get("retained_candidates")
        or totals["retained_rows"]
        != decision.get("counts", {}).get("unique_content_rows")
        or totals["global_exact_duplicate_rows"]
        != decision.get("counts", {}).get("global_exact_duplicate_rows")
    ):
        raise PleiasGlobalExactDedupError("filter aggregate accounting differs")
    payload = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_pleias_global_exact_dedup",
        "source": {
            "decontamination_aggregate_receipt_sha256": source["receipt_sha256"],
            "exact_dedup_decision_receipt_sha256": decision["receipt_sha256"],
        },
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
        },
        "totals": dict(sorted(totals.items())),
        "global_exact_deduplication_complete": True,
        "global_near_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    decide = commands.add_parser("decide")
    decide.add_argument("--decontamination-root", type=Path, required=True)
    decide.add_argument("--decontamination-aggregate", type=Path, required=True)
    decide.add_argument("--output-root", type=Path, required=True)
    decide.add_argument("--logical-shards", type=int, required=True)
    shard = commands.add_parser("filter-shard")
    shard.add_argument("--decontamination-root", type=Path, required=True)
    shard.add_argument("--decontamination-aggregate", type=Path, required=True)
    shard.add_argument("--decision-root", type=Path, required=True)
    shard.add_argument("--output-root", type=Path, required=True)
    shard.add_argument("--logical-shards", type=int, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    combine = commands.add_parser("aggregate")
    combine.add_argument("--decontamination-aggregate", type=Path, required=True)
    combine.add_argument("--decision-root", type=Path, required=True)
    combine.add_argument("--filters-root", type=Path, required=True)
    combine.add_argument("--logical-shards", type=int, required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "decide":
        result = build_decision(
            args.decontamination_root,
            args.decontamination_aggregate,
            args.output_root,
            args.logical_shards,
        )
    elif args.command == "filter-shard":
        result = run_filter_shard(
            args.decontamination_root,
            args.decontamination_aggregate,
            args.decision_root,
            args.output_root,
            args.logical_shards,
            args.shard_index,
        )
    else:
        result = aggregate_filters(
            args.decontamination_aggregate,
            args.decision_root,
            args.filters_root,
            args.logical_shards,
            args.output,
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
