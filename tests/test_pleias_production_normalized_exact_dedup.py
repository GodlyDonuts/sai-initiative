import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.pleias_production_descriptor_census import _schema, descriptor
from sai.data.pleias_production_normalized_exact_dedup import (
    PleiasProductionNormalizedExactDedupError,
    build_decision,
    validate_descriptor,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _row(identifier, text):
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


def _fixture(tmp_path: Path):
    parent = {"source_path": "data/p.parquet", "sha256": "a" * 64}
    base = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 10
    first = descriptor(_row("first", base.upper()), parent, 0)
    second = descriptor(_row("second", "  " + base + "\n"), parent, 1)
    distinct = descriptor(
        _row("third", "physics chemistry biology astronomy geology ecology " * 15),
        parent,
        2,
    )
    rows = [first, second, distinct]
    shard = tmp_path / "descriptors" / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    data = shard / "candidate_descriptors.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=_schema()), data, compression="zstd"
    )
    import hashlib

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
            "counts": {
                "production_candidate_descriptors": len(rows),
                "production_candidate_text_utf8_bytes": sum(
                    row["text_utf8_bytes"] for row in rows
                ),
                "production_candidate_source_tokens": sum(
                    row["token_count"] for row in rows
                ),
            },
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
            "totals": {
                "production_candidate_descriptors": len(rows),
                "production_candidate_text_utf8_bytes": sum(
                    row["text_utf8_bytes"] for row in rows
                ),
                "production_candidate_source_tokens": sum(
                    row["token_count"] for row in rows
                ),
            },
            "complete_source_parent_coverage": True,
            "source_text_persisted": False,
            "training_ready": False,
        }
    )
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, sort_keys=True))
    return rows, aggregate_path


def test_normalized_exact_decision_collapses_surface_variants(tmp_path):
    rows, aggregate = _fixture(tmp_path)
    result = build_decision(
        tmp_path / "descriptors",
        aggregate,
        tmp_path / "decision",
        1,
    )
    assert result["counts"]["source_rows"] == 3
    assert result["counts"]["unique_full_content_rows"] == 3
    assert result["counts"]["unique_normalized_content_rows"] == 2
    assert result["counts"]["additional_normalized_exact_duplicate_rows"] == 1
    assert result["decision_contains_source_text"] is False
    connection = sqlite3.connect(
        tmp_path / "decision" / "normalized_exact_keep.sqlite3"
    )
    identities = {
        row[0]
        for row in connection.execute("SELECT source_row_identity_sha256 FROM keep")
    }
    connection.close()
    assert (
        min(
            rows[0]["source_row_identity_sha256"], rows[1]["source_row_identity_sha256"]
        )
        in identities
    )
    assert result["training_ready"] is False


def test_descriptor_validation_rejects_mutated_sketch(tmp_path):
    rows, _aggregate = _fixture(tmp_path)
    changed = dict(rows[0])
    changed["near_dedup_bottom_k_u64"] = list(
        reversed(changed["near_dedup_bottom_k_u64"])
    )
    with pytest.raises(
        PleiasProductionNormalizedExactDedupError, match="descriptor row"
    ):
        validate_descriptor(changed)
