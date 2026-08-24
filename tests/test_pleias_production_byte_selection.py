import json
import sqlite3

import pytest

from sai.data.pleias_production_byte_selection import (
    PleiasProductionByteSelectionError,
    build_selection,
    choose_rows,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def test_keeps_every_row_when_quality_core_is_under_ceiling():
    rows = [("c", "science", 20), ("a", "books", 10), ("b", "code", 15)]
    selected, strata, size = choose_rows(rows, 100)
    assert selected == {"a", "b", "c"}
    assert strata == {"books": 10, "code": 15, "science": 20}
    assert size == 45


def test_diversity_pass_precedes_deterministic_refill():
    rows = [
        ("a", "dominant", 15),
        ("b", "dominant", 15),
        ("c", "dominant", 15),
        ("d", "rare", 15),
        ("e", "rare-two", 15),
        ("f", "rare-three", 15),
    ]
    selected, strata, size = choose_rows(rows, 80)
    assert selected == {"a", "b", "d", "e", "f"}
    assert strata["rare"] == 15
    assert strata["rare-two"] == 15
    assert size == 75


def test_rejects_duplicate_or_invalid_candidates():
    with pytest.raises(PleiasProductionByteSelectionError):
        choose_rows([("a", "books", 1), ("a", "books", 2)], 10)
    with pytest.raises(PleiasProductionByteSelectionError):
        choose_rows([("a", "books", 0)], 10)


def _signed(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def test_build_selection_replays_dedup_databases_without_text(tmp_path):
    exact_root = tmp_path / "exact"
    exact_root.mkdir()
    exact_database = exact_root / "keep.sqlite3"
    connection = sqlite3.connect(exact_database)
    connection.execute(
        "CREATE TABLE keep ("
        "normalized_content_sha256 TEXT PRIMARY KEY, "
        "source_row_identity_sha256 TEXT NOT NULL UNIQUE, "
        "content_sha256 TEXT NOT NULL, source_path TEXT NOT NULL, "
        "source_parent_sha256 TEXT NOT NULL, source_row_index INTEGER NOT NULL, "
        "stratum TEXT NOT NULL, text_utf8_bytes INTEGER NOT NULL, "
        "token_count INTEGER NOT NULL, descriptor_sha256 TEXT NOT NULL"
        ") WITHOUT ROWID"
    )
    identities = [character * 64 for character in "abc"]
    for index, identity in enumerate(identities):
        connection.execute(
            "INSERT INTO keep VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"{index + 1:064x}",
                identity,
                f"{index + 4:064x}",
                "data/parent.parquet",
                "d" * 64,
                index,
                ["books", "science", "code"][index],
                10,
                5,
                f"{index + 8:064x}",
            ),
        )
    connection.commit()
    connection.close()
    exact = _signed(
        {
            "schema": "sai-pleias-production-normalized-exact-dedup-v1",
            "status": "complete_nontraining_pleias_production_normalized_exact_dedup",
            "keep_database": {
                "path": exact_database.name,
                "bytes": exact_database.stat().st_size,
                "sha256": sha256_file(exact_database),
            },
            "decision_contains_source_text": False,
            "normalized_exact_deduplication_complete": True,
            "training_ready": False,
        }
    )
    (exact_root / "receipt.json").write_text(json.dumps(exact, sort_keys=True))
    near_root = tmp_path / "near"
    near_root.mkdir()
    near_database = near_root / "drops.sqlite3"
    connection = sqlite3.connect(near_database)
    connection.execute(
        "CREATE TABLE drops ("
        "source_row_identity_sha256 TEXT PRIMARY KEY, "
        "representative_source_row_identity_sha256 TEXT NOT NULL"
        ") WITHOUT ROWID"
    )
    connection.execute(
        "INSERT INTO drops VALUES (?, ?)", (identities[1], identities[0])
    )
    connection.commit()
    connection.close()
    near = _signed(
        {
            "schema": "sai-pleias-production-high-precision-near-dedup-v1",
            "status": "complete_nontraining_pleias_high_precision_near_dedup",
            "source": {"normalized_exact_receipt_sha256": exact["receipt_sha256"]},
            "drop_database": {
                "path": near_database.name,
                "bytes": near_database.stat().st_size,
                "sha256": sha256_file(near_database),
            },
            "decision_contains_source_text": False,
            "high_precision_near_duplicate_pass_complete": True,
            "training_ready": False,
        }
    )
    (near_root / "receipt.json").write_text(json.dumps(near, sort_keys=True))
    output = tmp_path / "selection"
    result = build_selection(exact_root, near_root, output, 15)
    assert result["counts"]["post_near_candidate_rows"] == 2
    assert result["counts"]["selected_rows"] == 1
    assert result["counts"]["selected_text_utf8_bytes"] == 10
    assert result["selection_contains_source_text"] is False
    assert result["padding_performed"] is False
    connection = sqlite3.connect(output / "selected_rows.sqlite3")
    selected = connection.execute(
        "SELECT source_row_identity_sha256 FROM selected"
    ).fetchall()
    connection.close()
    assert selected == [(identities[0],)]
