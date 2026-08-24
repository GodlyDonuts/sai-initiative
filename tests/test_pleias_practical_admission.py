import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data.institutional_books_practical_admission import (
    SCHEMA as BOOKS_ADMISSION_SCHEMA,
)
from sai.data.pleias_practical_admission import (
    _UPSERT,
    _open_database,
    _valid_locator,
    build_admission,
)
from sai.data.pleias_practical_locator_scan import LOCATOR_SCHEMA, SHARD_SCHEMA, _schema
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
        connection.execute(
            _UPSERT,
            ("3" * 64, "f" * 64, 10, 20, "public domain", '{"winner":false}'),
        )
        connection.execute(
            _UPSERT,
            ("3" * 64, "0" * 64, 11, 21, "cc0", '{"winner":true}'),
        )
        row = connection.execute(
            "SELECT identity_sha256, text_utf8_bytes, row_json FROM winners"
        ).fetchone()
        assert row == ("0" * 64, 11, '{"winner":true}')
    finally:
        connection.close()


def _signed(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


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
    manifest.write_text(json.dumps(manifest_row) + "\n")
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
    ]
    locator_path = scan_shard / "locators.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema()), locator_path)
    scan_receipt = _signed(
        {
            "schema": SHARD_SCHEMA,
            "status": "complete_pleias_practical_locator_scan_shard",
            "logical_shards": 1,
            "shard_index": 0,
            "source": {
                "manifest_sha256": sha256_file(manifest),
                "selected_paths_sha256": canonical_sha256(["data/file.parquet"]),
            },
            "selected": {
                "rows": 3,
                "text_utf8_bytes": 6_800,
                "source_token_count": 1_700,
            },
            "output": {
                "path": locator_path.name,
                "bytes": locator_path.stat().st_size,
                "sha256": sha256_file(locator_path),
            },
            "practical_candidate_complete": True,
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

    result = build_admission(
        manifest,
        tmp_path / "scan",
        books_path,
        tmp_path / "output",
        logical_shards=1,
        total_text_byte_ceiling=5_000,
        output_shards=2,
        scratch_root=tmp_path,
    )

    assert result["counts"]["exact_duplicate_rows_excluded"] == 1
    assert result["counts"]["byte_cap_excluded_rows"] == 1
    assert result["counts"]["admitted_rows"] == 1
    assert result["counts"]["combined_books_plus_pleias_text_utf8_bytes"] == 3_400
    assert result["global_exact_content_deduplication_complete"] is True
    assert result["practical_pretraining_ready"] is True
