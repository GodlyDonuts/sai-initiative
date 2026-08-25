import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.institutional_books_practical_admission import (
    SCHEMA as BOOKS_ADMISSION_SCHEMA,
)
from sai.data.pleias_practical_admission import (
    _UPSERT,
    SQLITE_CACHE_KIB,
    SQLITE_PAGE_BYTES,
    SQLITE_WORKER_THREADS,
    PleiasPracticalAdmissionError,
    _load_quarantine_content_hashes,
    _open_database,
    _output_shard,
    _valid_locator,
    build_admission,
)
from sai.data.pleias_practical_locator_scan import (
    LOCATOR_SCHEMA,
    SHARD_SCHEMA,
    _ordered_parents,
    _schema,
)
from sai.data.quarantine_exclusion_registry import (
    RECORD_SCHEMA as QUARANTINE_RECORD_SCHEMA,
)
from sai.data.quarantine_exclusion_registry import (
    SCHEMA as QUARANTINE_REGISTRY_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _locator(identity: str = "1" * 64) -> dict:
    return {
        "schema": LOCATOR_SCHEMA,
        "source_id": "pleias_common_corpus",
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "source_path": "data/file.parquet",
        "source_parent_sha256": "2" * 64,
        "source_row_index": 4,
        "source_row_identity_sha256": identity,
        "identifier": "doc-4",
        "collection": "books",
        "open_type": "open",
        "license": "public domain",
        "language": "English",
        "word_count": 400,
        "source_token_count": 600,
        "text_utf8_bytes": 2400,
        "content_sha256": "3" * 64,
    }


def test_valid_locator_is_english_and_structurally_complete() -> None:
    assert _valid_locator(_locator())
    row = _locator()
    row["language"] = "French"
    assert not _valid_locator(row)


def test_sqlite_exact_dedup_keeps_smallest_identity(tmp_path: Path) -> None:
    connection = _open_database(tmp_path / "dedup.sqlite3")
    try:
        assert connection.execute("PRAGMA page_size").fetchone()[0] == SQLITE_PAGE_BYTES
        assert (
            connection.execute("PRAGMA cache_size").fetchone()[0] == -SQLITE_CACHE_KIB
        )
        assert (
            connection.execute("PRAGMA threads").fetchone()[0] == SQLITE_WORKER_THREADS
        )
        connection.execute(
            _UPSERT,
            ("3" * 64, "f" * 64, 1, 10, 20, "public domain", '{"winner":false}'),
        )
        connection.execute(
            _UPSERT,
            ("3" * 64, "0" * 64, 2, 11, 21, "cc0", '{"winner":true}'),
        )
        row = connection.execute(
            "SELECT identity_sha256, output_shard, text_utf8_bytes, row_json "
            "FROM winners"
        ).fetchone()
        assert row == ("0" * 64, 2, 11, '{"winner":true}')
    finally:
        connection.close()


def test_source_local_partition_keeps_one_parent_together() -> None:
    assert _output_shard("data/one.parquet", 128) == _output_shard(
        "data/one.parquet", 128
    )
    assert 0 <= _output_shard("data/two.parquet", 128) < 128


def _signed(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _quarantine_registry(root: Path, content_hashes: list[str]) -> Path:
    root.mkdir()
    rows = []
    for index, content_hash in enumerate(content_hashes):
        row = {
            "schema": QUARANTINE_RECORD_SCHEMA,
            "candidate_identity_sha256": f"{index + 1:064x}",
            "source_content_sha256": content_hash,
            "source_manifest_receipt_sha256": "a" * 64,
            "source_record_sha256": "b" * 64,
            "route": "quarantine",
            "dataset_materialization_allowed": False,
            "source_text_persisted": False,
        }
        row["record_sha256"] = canonical_sha256(row)
        rows.append(row)
    registry = root / "quarantine_registry.jsonl"
    registry.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    receipt = _signed(
        {
            "schema": QUARANTINE_REGISTRY_SCHEMA,
            "status": "complete_quarantine_exclusion_registry",
            "unique_quarantine_rows": len(rows),
            "registry": {
                "path": registry.name,
                "rows": len(rows),
                "bytes": registry.stat().st_size,
                "sha256": sha256_file(registry),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in rows]
                ),
            },
            "dataset_materialization_allowed": False,
            "source_text_persisted": False,
            "training_ready": False,
        }
    )
    (root / "receipt.json").write_text(json.dumps(receipt))
    return root


def test_quarantine_registry_tamper_fails_closed(tmp_path: Path) -> None:
    root = _quarantine_registry(tmp_path / "quarantine", ["4" * 64])
    registry = root / "quarantine_registry.jsonl"
    registry.write_text(registry.read_text().replace("4" * 64, "5" * 64))
    with pytest.raises(PleiasPracticalAdmissionError, match="registry differs"):
        _load_quarantine_content_hashes(root)


def test_build_admission_deduplicates_and_respects_combined_cap(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest_row = {
        "source_id": "pleias_common_corpus",
        "source_path": "data/file.parquet",
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "bytes": 123,
        "sha256": "2" * 64,
        "raw_source_is_training_ready": False,
    }
    second_manifest_row = {
        **manifest_row,
        "source_path": "data/c.parquet",
        "bytes": 456,
    }
    manifest.write_text(
        json.dumps(manifest_row) + "\n" + json.dumps(second_manifest_row) + "\n"
    )
    scan_shard = tmp_path / "scan" / "shards" / "shard_00000"
    scan_shard.mkdir(parents=True)
    rows = [
        _locator("f" * 64),
        _locator("0" * 64),
        {
            **_locator("1" * 64),
            "content_sha256": "4" * 64,
            "text_utf8_bytes": 2_000,
            "source_token_count": 500,
            "source_row_index": 5,
            "identifier": "doc-5",
        },
        {
            **_locator("2" * 64),
            "source_path": "data/c.parquet",
            "content_sha256": "5" * 64,
            "text_utf8_bytes": 2_000,
            "source_token_count": 500,
            "source_row_index": 6,
            "identifier": "doc-6",
        },
        {
            **_locator("3" * 64),
            "content_sha256": "6" * 64,
            "text_utf8_bytes": 3_000,
            "source_token_count": 750,
            "source_row_index": 7,
            "identifier": "doc-7",
        },
    ]
    locator_path = scan_shard / "locators.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema()), locator_path)
    manifest_rows = [second_manifest_row, manifest_row]
    ordered_parents, parent_scan_order = _ordered_parents(manifest_rows, True)
    scan_receipt = _signed(
        {
            "schema": SHARD_SCHEMA,
            "status": "complete_pleias_practical_locator_scan_shard",
            "logical_shards": 1,
            "shard_index": 0,
            "source": {
                "manifest_sha256": sha256_file(manifest),
                "selected_paths_sha256": canonical_sha256(
                    ["data/c.parquet", "data/file.parquet"]
                ),
                "selected_parent_count": 2,
                "scanned_parent_count": 2,
                "ordered_scanned_paths_sha256": canonical_sha256(
                    [row["source_path"] for row in ordered_parents]
                ),
            },
            "policy": {
                "parent_scan_order": parent_scan_order,
                "stop_at_byte_cap": True,
            },
            "selected": {
                "rows": 5,
                "text_utf8_bytes": 11_800,
                "source_token_count": 2_950,
            },
            "output": {
                "path": locator_path.name,
                "bytes": locator_path.stat().st_size,
                "sha256": sha256_file(locator_path),
            },
            "practical_candidate_complete": True,
            "complete_assigned_parent_scan": True,
            "deterministic_byte_cap_sampling_complete": True,
            "byte_cap_respected": True,
            "training_ready": False,
        }
    )
    (scan_shard / "receipt.json").write_text(json.dumps(scan_receipt))
    books_path = tmp_path / "books.json"
    books = _signed(
        {
            "schema": BOOKS_ADMISSION_SCHEMA,
            "counts": {"admitted_text_utf8_bytes": 1_000},
            "practical_pretraining_ready": True,
            "training_ready": True,
        }
    )
    books_path.write_text(json.dumps(books))
    quarantine_root = _quarantine_registry(tmp_path / "quarantine", ["4" * 64])

    hashes, descriptor = _load_quarantine_content_hashes(quarantine_root)
    assert hashes == {"4" * 64}
    assert descriptor["rows"] == 1

    result = build_admission(
        manifest,
        tmp_path / "scan",
        books_path,
        quarantine_root,
        tmp_path / "output",
        logical_shards=1,
        total_text_byte_ceiling=8_000,
        output_shards=2,
        scratch_root=tmp_path,
    )

    assert result["counts"]["exact_duplicate_rows_excluded"] == 1
    assert result["counts"]["known_quarantine_rows_excluded"] == 1
    assert result["counts"]["known_quarantine_text_utf8_bytes_excluded"] == 2_000
    assert result["counts"]["byte_cap_excluded_rows"] == 1
    assert result["counts"]["admitted_rows"] == 2
    assert result["counts"]["combined_books_plus_pleias_text_utf8_bytes"] == 5_400
    assert result["counts"]["admitted_collection_count"] == 1
    assert result["counts"]["collections"] == {"books": 2}
    assert [row["shard_index"] for row in result["outputs"]["descriptors"]] == [0, 1]
    admitted = []
    for descriptor in result["outputs"]["descriptors"]:
        admitted.extend(
            pq.read_table(tmp_path / "output" / descriptor["path"]).to_pylist()
        )
    assert sorted(row["content_sha256"] for row in admitted) == ["3" * 64, "5" * 64]
    assert result["policy"]["byte_cap_selection_policy"] == (
        "canonical_content_sha256_order"
    )
    assert result["global_exact_content_deduplication_complete"] is True
    assert result["known_quarantine_exclusions_applied"] is True
    assert result["source"]["quarantine_registry"]["rows"] == 1
    assert result["practical_pretraining_ready"] is True
