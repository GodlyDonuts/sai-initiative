"""Exact-deduplicate and admit the practical PleIAs locator corpus."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books_practical_admission import (
    SCHEMA as BOOKS_ADMISSION_SCHEMA,
)
from sai.data.pleias_metadata_census import load_manifest, select_shard
from sai.data.pleias_practical_locator_scan import (
    LOCATOR_SCHEMA,
    SHARD_SCHEMA,
    _schema,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pleias-practical-admission-receipt-v1"
SQLITE_PAGE_BYTES = 65_536
SQLITE_CACHE_KIB = 4 * 1024 * 1024
# Stokes's SQLite build reports MAX_MMAP_SIZE=0x7fff0000; requesting its exact
# supported ceiling avoids pretending a larger mapping was admitted.
SQLITE_MMAP_BYTES = 0x7FFF0000
SQLITE_WORKER_THREADS = 4


class PleiasPracticalAdmissionError(RuntimeError):
    """Practical locator custody, deduplication, or byte accounting differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasPracticalAdmissionError("signed input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasPracticalAdmissionError("signed input is invalid") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise PleiasPracticalAdmissionError("signed input differs")
    return payload


def _valid_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_locator(row: dict[str, Any]) -> bool:
    return bool(
        row.get("schema") == LOCATOR_SCHEMA
        and row.get("source_id") == "pleias_common_corpus"
        and all(
            isinstance(row.get(key), str) and row[key]
            for key in (
                "source_repository",
                "source_revision",
                "source_path",
                "identifier",
                "collection",
                "open_type",
                "license",
            )
        )
        and row.get("language", "").strip().casefold() == "english"
        and _valid_hex(row.get("source_parent_sha256"))
        and _valid_hex(row.get("source_row_identity_sha256"))
        and _valid_hex(row.get("content_sha256"))
        and isinstance(row.get("source_row_index"), int)
        and row["source_row_index"] >= 0
        and isinstance(row.get("word_count"), int)
        and row["word_count"] > 0
        and isinstance(row.get("source_token_count"), int)
        and row["source_token_count"] > 0
        and isinstance(row.get("text_utf8_bytes"), int)
        and row["text_utf8_bytes"] > 0
    )


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    # Admission runs on node-local scratch with 64 GiB and four reserved CPUs.
    # Use a bounded fraction of that memory for the content-hash B-tree and let
    # SQLite parallelize eligible sort work. These pragmas change only physical
    # index execution; winner ordering and emitted scientific bytes are fixed by
    # the SQL statements below.
    connection.execute(f"PRAGMA page_size={SQLITE_PAGE_BYTES}")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(f"PRAGMA cache_size=-{SQLITE_CACHE_KIB}")
    connection.execute(f"PRAGMA mmap_size={SQLITE_MMAP_BYTES}")
    connection.execute(f"PRAGMA threads={SQLITE_WORKER_THREADS}")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute(
        """
        CREATE TABLE winners (
            content_sha256 TEXT PRIMARY KEY,
            identity_sha256 TEXT NOT NULL,
            output_shard INTEGER NOT NULL,
            text_utf8_bytes INTEGER NOT NULL,
            source_token_count INTEGER NOT NULL,
            license TEXT NOT NULL,
            row_json TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    return connection


_UPSERT = """
INSERT INTO winners (
    content_sha256, identity_sha256, output_shard, text_utf8_bytes,
    source_token_count, license, row_json
) VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(content_sha256) DO UPDATE SET
    identity_sha256=excluded.identity_sha256,
    output_shard=excluded.output_shard,
    text_utf8_bytes=excluded.text_utf8_bytes,
    source_token_count=excluded.source_token_count,
    license=excluded.license,
    row_json=excluded.row_json
WHERE excluded.identity_sha256 < winners.identity_sha256
"""


def _output_shard(source_path: str, output_shards: int) -> int:
    if not isinstance(source_path, str) or not source_path or output_shards < 1:
        raise PleiasPracticalAdmissionError("source-local output partition differs")
    return int(canonical_sha256({"source_path": source_path})[:16], 16) % output_shards


def build_admission(
    manifest_path: Path,
    scan_root: Path,
    books_receipt_path: Path,
    output_root: Path,
    logical_shards: int,
    total_text_byte_ceiling: int,
    output_shards: int = 128,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    """Verify all scans, exact-deduplicate locators, and fill the corpus ceiling."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or logical_shards < 1
        or not 1 <= output_shards <= 128
        or total_text_byte_ceiling <= 0
    ):
        raise PleiasPracticalAdmissionError("admission arguments differ")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise PleiasPracticalAdmissionError("pyarrow is required") from error

    books = _load_signed(books_receipt_path, BOOKS_ADMISSION_SCHEMA)
    books_bytes = books.get("counts", {}).get("admitted_text_utf8_bytes")
    if (
        books.get("practical_pretraining_ready") is not True
        or books.get("training_ready") is not True
        or isinstance(books_bytes, bool)
        or not isinstance(books_bytes, int)
        or books_bytes <= 0
        or books_bytes >= total_text_byte_ceiling
    ):
        raise PleiasPracticalAdmissionError("Books practical admission differs")
    maximum_pleias_bytes = total_text_byte_ceiling - books_bytes
    manifest = load_manifest(manifest_path)
    expected_paths = {row["source_path"] for row in manifest}
    seen_paths: set[str] = set()
    receipt_hashes = []
    scan_counts: Counter[str] = Counter()
    all_assigned_parents_scanned = True
    output_root.mkdir(parents=True)

    with tempfile.TemporaryDirectory(
        prefix="sai-pleias-practical-admission-", dir=scratch_root
    ) as temporary_directory:
        database_path = Path(temporary_directory) / "exact-dedup.sqlite3"
        database = _open_database(database_path)
        try:
            for shard_index in range(logical_shards):
                shard_root = scan_root / "shards" / f"shard_{shard_index:05d}"
                receipt = _load_signed(shard_root / "receipt.json", SHARD_SCHEMA)
                selected_parents = select_shard(manifest, logical_shards, shard_index)
                selected_paths = {row["source_path"] for row in selected_parents}
                descriptor = receipt.get("output", {})
                locator_path = shard_root / descriptor.get("path", "")
                scanned_parent_count = receipt.get("source", {}).get(
                    "scanned_parent_count"
                )
                if (
                    receipt.get("status")
                    != "complete_pleias_practical_locator_scan_shard"
                    or receipt.get("logical_shards") != logical_shards
                    or receipt.get("shard_index") != shard_index
                    or receipt.get("source", {}).get("manifest_sha256")
                    != sha256_file(manifest_path)
                    or receipt.get("source", {}).get("selected_paths_sha256")
                    != canonical_sha256(
                        [row["source_path"] for row in selected_parents]
                    )
                    or receipt.get("practical_candidate_complete") is not True
                    or receipt.get("deterministic_byte_cap_sampling_complete")
                    is not True
                    or isinstance(scanned_parent_count, bool)
                    or not isinstance(scanned_parent_count, int)
                    or not 1 <= scanned_parent_count <= len(selected_parents)
                    or receipt.get("training_ready") is not False
                    or receipt.get("byte_cap_respected") is not True
                    or not locator_path.is_file()
                    or locator_path.is_symlink()
                    or locator_path.stat().st_nlink != 1
                    or locator_path.stat().st_size != descriptor.get("bytes")
                    or sha256_file(locator_path) != descriptor.get("sha256")
                    or seen_paths.intersection(selected_paths)
                ):
                    raise PleiasPracticalAdmissionError("scan shard differs")
                seen_paths.update(selected_paths)
                all_assigned_parents_scanned = bool(
                    all_assigned_parents_scanned
                    and receipt.get("complete_assigned_parent_scan") is True
                )
                parquet = pq.ParquetFile(locator_path)
                observed_rows = 0
                observed_bytes = 0
                observed_tokens = 0
                for batch in parquet.iter_batches(batch_size=4096, use_threads=False):
                    rows = batch.to_pylist()
                    values = []
                    for row in rows:
                        if not _valid_locator(row):
                            raise PleiasPracticalAdmissionError("locator row differs")
                        observed_rows += 1
                        observed_bytes += row["text_utf8_bytes"]
                        observed_tokens += row["source_token_count"]
                        values.append(
                            (
                                row["content_sha256"],
                                row["source_row_identity_sha256"],
                                _output_shard(row["source_path"], output_shards),
                                row["text_utf8_bytes"],
                                row["source_token_count"],
                                row["license"],
                                json.dumps(row, sort_keys=True, separators=(",", ":")),
                            )
                        )
                    database.executemany(_UPSERT, values)
                selected = receipt.get("selected", {})
                if (
                    observed_rows != selected.get("rows")
                    or observed_bytes != selected.get("text_utf8_bytes")
                    or observed_tokens != selected.get("source_token_count")
                ):
                    raise PleiasPracticalAdmissionError("scan accounting differs")
                scan_counts.update(
                    {
                        "candidate_rows": observed_rows,
                        "candidate_text_utf8_bytes": observed_bytes,
                        "candidate_source_token_count": observed_tokens,
                        "locator_parquet_bytes": descriptor["bytes"],
                        "scanned_source_parents": scanned_parent_count,
                    }
                )
                receipt_hashes.append(receipt["receipt_sha256"])
                print(
                    json.dumps(
                        {
                            "event": "pleias_practical_admission_scan_progress",
                            "complete_scan_shards": shard_index + 1,
                            "remaining_scan_shards": logical_shards - shard_index - 1,
                            "candidate_rows": scan_counts["candidate_rows"],
                            "candidate_text_utf8_bytes": scan_counts[
                                "candidate_text_utf8_bytes"
                            ],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if seen_paths != expected_paths:
                raise PleiasPracticalAdmissionError("source parent coverage differs")
            database.commit()

            unique_rows = database.execute("SELECT COUNT(*) FROM winners").fetchone()[0]
            duplicate_rows = scan_counts["candidate_rows"] - unique_rows
            print(
                json.dumps(
                    {
                        "event": "pleias_practical_admission_exact_dedup_complete",
                        "candidate_rows": scan_counts["candidate_rows"],
                        "unique_candidate_rows": unique_rows,
                        "exact_duplicate_rows_excluded": duplicate_rows,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            output_parent = output_root / "shards"
            output_parent.mkdir()
            schema = _schema()
            descriptors = []
            current_index = None
            writer = None
            temporary_path = None
            output_path = None
            pending = []
            shard_rows = shard_bytes = shard_tokens = 0
            admitted_rows = admitted_bytes = admitted_tokens = 0
            byte_cap_excluded_rows = byte_cap_excluded_bytes = 0
            rights: Counter[str] = Counter()
            collections: Counter[str] = Counter()
            open_types: Counter[str] = Counter()

            def close_writer() -> None:
                nonlocal writer, temporary_path, output_path
                nonlocal pending, shard_rows, shard_bytes, shard_tokens
                if writer is None or temporary_path is None or output_path is None:
                    return
                if pending:
                    writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                    pending = []
                writer.close()
                os.replace(temporary_path, output_path)
                descriptors.append(
                    {
                        "shard_index": current_index,
                        "path": str(output_path.relative_to(output_root)),
                        "rows": shard_rows,
                        "text_utf8_bytes": shard_bytes,
                        "source_token_count": shard_tokens,
                        "bytes": output_path.stat().st_size,
                        "sha256": sha256_file(output_path),
                    }
                )
                writer = temporary_path = output_path = None
                shard_rows = shard_bytes = shard_tokens = 0

            cursor = database.execute(
                "SELECT output_shard, text_utf8_bytes, "
                "source_token_count, license, row_json FROM winners "
                "ORDER BY output_shard, content_sha256"
            )
            for (
                target,
                text_bytes,
                tokens,
                license_name,
                row_json,
            ) in cursor:
                if admitted_bytes + text_bytes > maximum_pleias_bytes:
                    byte_cap_excluded_rows += 1
                    byte_cap_excluded_bytes += text_bytes
                    continue
                if current_index != target:
                    close_writer()
                    current_index = target
                    shard_root = output_parent / f"shard_{target:05d}"
                    shard_root.mkdir()
                    output_path = shard_root / "locators.parquet"
                    temporary_path = shard_root / (
                        f".locators.partial.{uuid.uuid4().hex}.parquet"
                    )
                    writer = pq.ParquetWriter(
                        temporary_path, schema, compression="zstd"
                    )
                row = json.loads(row_json)
                pending.append(row)
                admitted_rows += 1
                admitted_bytes += text_bytes
                admitted_tokens += tokens
                shard_rows += 1
                shard_bytes += text_bytes
                shard_tokens += tokens
                rights[license_name] += 1
                collections[row["collection"]] += 1
                open_types[row["open_type"]] += 1
                if len(pending) >= 4096:
                    writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                    pending = []
            close_writer()
        finally:
            database.close()

    if not descriptors or admitted_rows <= 0:
        raise PleiasPracticalAdmissionError("practical admission is empty")
    payload = {
        "schema": SCHEMA,
        "status": "complete_practical_pleias_pretraining_admission",
        "source": {
            "manifest_sha256": sha256_file(manifest_path),
            "source_parent_count": len(manifest),
            "source_parent_bytes": sum(row["bytes"] for row in manifest),
            "scanned_source_parent_count": scan_counts["scanned_source_parents"],
            "scan_logical_shards": logical_shards,
            "ordered_scan_receipts_sha256": canonical_sha256(receipt_hashes),
            "books_admission_receipt_sha256": books["receipt_sha256"],
        },
        "policy": {
            "english_only": True,
            "explicit_reusable_rights_only": True,
            "mechanical_non_slop_gate_required": True,
            "exact_content_duplicate_policy": "smallest_identity_sha256_wins",
            "output_partition_policy": "canonical_source_path_sha256_modulo",
            "semantic_model_review_required": False,
            "total_books_plus_pleias_text_byte_ceiling": total_text_byte_ceiling,
            "reserved_books_text_utf8_bytes": books_bytes,
            "maximum_pleias_text_utf8_bytes": maximum_pleias_bytes,
        },
        "counts": {
            **dict(sorted(scan_counts.items())),
            "exact_duplicate_rows_excluded": duplicate_rows,
            "unique_candidate_rows": unique_rows,
            "byte_cap_excluded_rows": byte_cap_excluded_rows,
            "byte_cap_excluded_text_utf8_bytes": byte_cap_excluded_bytes,
            "admitted_rows": admitted_rows,
            "admitted_text_utf8_bytes": admitted_bytes,
            "admitted_source_token_count": admitted_tokens,
            "combined_books_plus_pleias_text_utf8_bytes": books_bytes + admitted_bytes,
            "rights": dict(sorted(rights.items())),
            "admitted_collection_count": len(collections),
            "collections": dict(sorted(collections.items())),
            "open_types": dict(sorted(open_types.items())),
        },
        "outputs": {
            "shards": len(descriptors),
            "descriptors": descriptors,
            "ordered_descriptors_sha256": canonical_sha256(descriptors),
        },
        "complete_source_identity_partition_coverage": True,
        "complete_source_parent_content_scan": all_assigned_parents_scanned,
        "deterministic_byte_cap_sampling_complete": True,
        "global_exact_content_deduplication_complete": True,
        "global_near_deduplication_complete": False,
        "official_benchmark_decontamination_complete": False,
        "evaluation_claims_allowed": False,
        "source_text_copied": False,
        "source_text_reconstructed_from_pinned_upstream": True,
        "practical_pretraining_ready": True,
        "training_ready": True,
        "four_b_training_authorized": False,
    }
    if (
        payload["counts"]["combined_books_plus_pleias_text_utf8_bytes"]
        > total_text_byte_ceiling
    ):
        raise PleiasPracticalAdmissionError("combined byte ceiling exceeded")
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--books-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--total-text-byte-ceiling", type=int, required=True)
    parser.add_argument("--output-shards", type=int, default=128)
    parser.add_argument("--scratch-root", type=Path)
    args = parser.parse_args()
    result = build_admission(
        args.manifest,
        args.scan_root,
        args.books_receipt,
        args.output_root,
        args.logical_shards,
        args.total_text_byte_ceiling,
        args.output_shards,
        args.scratch_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
