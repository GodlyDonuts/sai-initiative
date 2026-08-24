import hashlib
import json
import sqlite3

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data.pleias_production_descriptor_census import _schema, descriptor
from sai.data.pleias_production_near_dedup import (
    band_keys,
    build_decision,
    high_confidence_near_duplicate,
    union_lowest,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def test_band_keys_require_complete_sorted_sketch():
    values = list(range(32))
    keys = band_keys(values)
    assert len(keys) == 4
    assert len(set(keys)) == 4
    assert band_keys(values[:8]) == []
    assert band_keys(list(reversed(values))) == []


def test_near_duplicate_requires_high_overlap_and_matched_length():
    first = list(range(32))
    close = list(range(2, 34))
    far = list(range(100, 132))
    assert high_confidence_near_duplicate(first, close, 10_000, 10_500)
    assert not high_confidence_near_duplicate(first, far, 10_000, 10_500)
    assert not high_confidence_near_duplicate(first, close, 10_000, 20_000)


def test_union_uses_lowest_stable_identity_and_transitive_root():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE forest (identity TEXT PRIMARY KEY, parent TEXT NOT NULL)"
    )
    assert union_lowest(connection, "c", "b") == ("b", True)
    assert union_lowest(connection, "b", "a") == ("a", True)
    assert union_lowest(connection, "c", "a") == ("a", False)
    parents = dict(connection.execute("SELECT identity, parent FROM forest"))
    assert parents["a"] == "a"
    connection.close()


def _signed(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _source_row(identifier, text):
    return {
        "identifier": identifier,
        "collection": "Books",
        "open_type": "Open Culture",
        "license": "Public Domain",
        "language": "English",
        "word_count": len(text.split()),
        "token_count": len(text.split()) * 2,
        "text": text,
    }


def test_builds_source_safe_high_precision_drop_decision(tmp_path):
    parent = {"source_path": "data/p.parquet", "sha256": "a" * 64}
    words = [f"concept{index}" for index in range(200)]
    changed = list(words)
    changed[100] = "replacementconcept"
    rows = [
        descriptor(_source_row("one", " ".join(words)), parent, 0),
        descriptor(_source_row("two", " ".join(changed)), parent, 1),
        descriptor(
            _source_row("three", " ".join(f"other{index}" for index in range(200))),
            parent,
            2,
        ),
    ]
    descriptor_root = tmp_path / "descriptors"
    shard = descriptor_root / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    data = shard / "candidate_descriptors.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=_schema()), data, compression="zstd"
    )
    ordered = hashlib.sha256()
    for row in rows:
        ordered.update(bytes.fromhex(row["descriptor_sha256"]))
    shard_receipt = _signed(
        {
            "schema": "sai-pleias-production-descriptor-census-shard-v1",
            "status": "complete_nontraining_pleias_production_descriptor_census_shard",
            "logical_shards": 1,
            "shard_index": 0,
            "source": {
                "manifest_sha256": "b" * 64,
                "policy_receipt_sha256": "c" * 64,
                "semantic_decision_receipt_sha256": "d" * 64,
            },
            "counts": {"production_candidate_descriptors": 3},
            "ordered_descriptor_digests_sha256": ordered.hexdigest(),
            "output": {
                "path": data.name,
                "bytes": data.stat().st_size,
                "sha256": sha256_file(data),
            },
            "source_text_persisted": False,
            "training_ready": False,
        }
    )
    (shard / "receipt.json").write_text(json.dumps(shard_receipt, sort_keys=True))
    aggregate = _signed(
        {
            "schema": "sai-pleias-production-descriptor-census-aggregate-v1",
            "status": "complete_nontraining_pleias_production_descriptor_census",
            "source": {
                "manifest_sha256": "b" * 64,
                "policy_receipt_sha256": "c" * 64,
                "semantic_decision_receipt_sha256": "d" * 64,
            },
            "shards": {"logical_shards": 1},
            "complete_source_parent_coverage": True,
            "source_text_persisted": False,
            "training_ready": False,
        }
    )
    aggregate_path = tmp_path / "descriptor-aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, sort_keys=True))
    exact_root = tmp_path / "exact"
    exact_root.mkdir()
    exact_database = exact_root / "normalized_exact_keep.sqlite3"
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
    for row in rows:
        connection.execute(
            "INSERT INTO keep VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["normalized_content_sha256"],
                row["source_row_identity_sha256"],
                row["content_sha256"],
                row["source_path"],
                row["source_parent_sha256"],
                row["source_row_index"],
                row["stratum"],
                row["text_utf8_bytes"],
                row["token_count"],
                row["descriptor_sha256"],
            ),
        )
    connection.commit()
    connection.close()
    exact_receipt = _signed(
        {
            "schema": "sai-pleias-production-normalized-exact-dedup-v1",
            "status": "complete_nontraining_pleias_production_normalized_exact_dedup",
            "source": {
                "descriptor_aggregate_receipt_sha256": aggregate["receipt_sha256"]
            },
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
    (exact_root / "receipt.json").write_text(json.dumps(exact_receipt, sort_keys=True))
    output = tmp_path / "near"
    result = build_decision(
        descriptor_root,
        aggregate_path,
        exact_root,
        output,
        1,
    )
    assert result["counts"]["high_confidence_edges"] >= 1
    assert result["counts"]["dropped_rows"] == 1
    assert result["decision_contains_source_text"] is False
    assert result["global_near_deduplication_complete"] is False
    connection = sqlite3.connect(output / "high_precision_near_drops.sqlite3")
    drops = connection.execute("SELECT * FROM drops").fetchall()
    connection.close()
    assert len(drops) == 1
