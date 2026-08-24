import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.institutional_books_cross_source_subdocument_rewrite_aggregate import (
    SCHEMA as BOOK_SCHEMA,
)
from sai.data.pleias_virtual_byte_balance import (
    PleiasVirtualByteBalanceError,
    aggregate_shards,
    build_allocation,
    excluded_database,
    is_excluded,
    select_shard,
)
from sai.data.pleias_virtual_cross_source_reconstruction import (
    AGGREGATE_SCHEMA,
    AGGREGATE_STATUS,
    LOCATOR_SCHEMA,
    SHARD_SCHEMA,
    SHARD_STATUS,
    _locator_schema,
)
from sai.data.token_stream import canonical_sha256, sha256_file
from sai.data.virtual_spiral_curriculum_index import index_pleias_shard


def _signed(path: Path, payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True))
    return payload


def _locator(identity_digit: str, size: int, split: str, quality: int) -> dict:
    row = {
        "schema": LOCATOR_SCHEMA,
        "virtual_row_index": 0,
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "source_path": f"common_corpus/{identity_digit}.parquet",
        "source_parent_sha256": "b" * 64,
        "source_row_index": 0,
        "source_row_identity_sha256": identity_digit * 64,
        "pre_internal_content_sha256": "c" * 64,
        "post_internal_content_sha256": "d" * 64,
        "content_sha256": "e" * 64,
        "source_text_utf8_bytes": size,
        "output_text_utf8_bytes": size,
        "source_text_characters": size,
        "output_text_characters": size,
        "source_word_count": 20,
        "output_word_count": 20,
        "source_token_count": 30,
        "token_count_requires_recomputation": True,
        "collection": "Open Textbooks",
        "open_type": "Open Culture",
        "license": "Public Domain",
        "language": "English",
        "semantic_stratum": "science::textbook",
        "semantic_quality_floor_milli": quality,
        "semantic_quality_mean_milli": quality,
        "semantic_difficulty_mean_milli": 2_000,
        "semantic_prerequisite_burden_mean_milli": 1_000,
        "semantic_curriculum_phase": "expansion",
        "semantic_domains": ["science"],
        "semantic_recurring_concepts": ["measurement"],
        "semantic_recurring_prerequisites": ["arithmetic"],
        "code_document": False,
        "internal_subdocument_transform_sha256": "f" * 64,
        "cross_source_subdocument_transform_sha256": "1" * 64,
        "source_group_sha256": "2" * 64,
        "source_group_bucket": 1,
        "corpus_split": split,
        "source_split_policy_sha256": "3" * 64,
        "training_ready": False,
    }
    row["locator_sha256"] = canonical_sha256(row)
    return row


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    books = tmp_path / "books.json"
    _signed(
        books,
        {
            "schema": BOOK_SCHEMA,
            "totals": {"output_text_utf8_bytes": 100},
            "benchmark_decontamination_complete": True,
            "cross_source_subdocument_deduplication_complete": True,
            "private_storage_only": True,
            "huggingface_redistribution_authorized": False,
            "training_ready": False,
        },
    )
    final = tmp_path / "final"
    receipts = []
    rows_by_shard = (
        [_locator("4", 60, "train", 9_000), _locator("5", 60, "train", 7_000)],
        [
            _locator("6", 60, "development", 9_000),
            _locator("7", 60, "development", 7_000),
        ],
    )
    for shard_index, rows in enumerate(rows_by_shard):
        for row_index, row in enumerate(rows):
            row["virtual_row_index"] = row_index
            row["locator_sha256"] = canonical_sha256(
                {key: value for key, value in row.items() if key != "locator_sha256"}
            )
        root = final / "shards" / f"shard_{shard_index:05d}"
        root.mkdir(parents=True)
        locators = root / "final-locators.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=_locator_schema()), locators)
        receipt = _signed(
            root / "receipt.json",
            {
                "schema": SHARD_SCHEMA,
                "status": SHARD_STATUS,
                "logical_shards": 2,
                "shard_index": shard_index,
                "counts": {"documents": 2, "output_text_utf8_bytes": 120},
                "final_locators": {
                    "path": locators.name,
                    "rows": 2,
                    "bytes": locators.stat().st_size,
                    "sha256": sha256_file(locators),
                },
                "complete_final_pleias_document_coverage": True,
                "benchmark_decontamination_complete": True,
                "cross_source_subdocument_deduplication_complete": True,
                "source_disjoint_split_complete": True,
                "source_text_persisted": False,
                "training_ready": False,
            },
        )
        receipts.append(receipt)
    _signed(
        final / "aggregate.json",
        {
            "schema": AGGREGATE_SCHEMA,
            "status": AGGREGATE_STATUS,
            "shards": {
                "logical_shards": 2,
                "ordered_receipts_sha256": canonical_sha256(
                    [receipt["receipt_sha256"] for receipt in receipts]
                ),
            },
            "totals": {"documents": 4, "output_text_utf8_bytes": 240},
            "complete_final_pleias_document_coverage": True,
            "benchmark_decontamination_complete": True,
            "pleias_internal_subdocument_deduplication_complete": True,
            "cross_source_subdocument_deduplication_complete": True,
            "source_text_persisted": False,
            "training_ready": False,
        },
    )
    return books, final


def test_balance_uses_exact_residual_and_keeps_highest_quality(tmp_path: Path) -> None:
    books, final = _workspace(tmp_path)
    allocation_path = tmp_path / "allocation.json"
    allocation = build_allocation(
        books,
        final,
        allocation_path,
        byte_ceiling=300,
        reserved_bytes=20,
    )
    assert allocation["policy"]["maximum_pleias_text_utf8_bytes"] == 180
    assert [
        row["maximum_selected_text_utf8_bytes"] for row in allocation["allocations"]
    ] == [90, 90]

    balance = tmp_path / "balance"
    for index in range(2):
        result = select_shard(
            final,
            allocation_path,
            balance / "shards" / f"shard_{index:05d}",
            logical_shards=2,
            shard_index=index,
        )
        assert result["selected_counts"]["documents"] == 1
        assert result["selected_counts"]["output_text_utf8_bytes"] == 60
        assert result["source_text_persisted"] is False
    aggregate = aggregate_shards(allocation_path, balance, balance / "aggregate.json")
    assert aggregate["selected_counts"]["documents"] == 2
    assert aggregate["selected_counts"]["output_text_utf8_bytes"] == 120
    assert aggregate["remaining_pleias_byte_headroom"] == 60

    database, _receipt = excluded_database(balance, 0)
    try:
        assert is_excluded(database, "4" * 64) is False
        assert is_excluded(database, "5" * 64) is True
    finally:
        database.close()
    assert "source text" not in (balance / "aggregate.json").read_text().lower()

    index = index_pleias_shard(
        final,
        balance,
        tmp_path / "index",
        logical_shards=2,
        shard_index=0,
    )
    assert index["index"]["rows"] == 1
    assert index["counts"]["output_text_utf8_bytes"] == 60
    assert index["source_balance_receipt_sha256"]


def test_balance_rejects_tampered_exclusion_database(tmp_path: Path) -> None:
    books, final = _workspace(tmp_path)
    allocation_path = tmp_path / "allocation.json"
    build_allocation(
        books,
        final,
        allocation_path,
        byte_ceiling=300,
        reserved_bytes=20,
    )
    balance = tmp_path / "balance"
    for index in range(2):
        select_shard(
            final,
            allocation_path,
            balance / "shards" / f"shard_{index:05d}",
            logical_shards=2,
            shard_index=index,
        )
    database = balance / "shards" / "shard_00000" / "excluded.sqlite3"
    database.write_bytes(database.read_bytes() + b"tamper")
    with pytest.raises(PleiasVirtualByteBalanceError, match="database differs"):
        aggregate_shards(allocation_path, balance, balance / "aggregate.json")
