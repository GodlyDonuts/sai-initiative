import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data.institutional_books_materializer import (
    OUTPUT_SCHEMA,
    PARENT_SCHEMA,
    SHARD_SCHEMA,
)
from sai.data.institutional_books_mechanical_sample_audit import build_audit
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def test_audits_real_hash_bound_rows_without_persisting_text(tmp_path):
    materialized = tmp_path / "books"
    root = materialized / "shards" / "shard_00000"
    (root / "parents").mkdir(parents=True)
    (root / "lineage").mkdir()
    (root / "data").mkdir()
    text = "A coherent historical explanation grounded in primary evidence. " * 8
    source = root / "data" / "parent_00000.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "schema": OUTPUT_SCHEMA,
                    "barcode_src": "book-1",
                    "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "text": text,
                    "training_ready": False,
                }
            ]
        ),
        source,
    )
    lineage = root / "lineage" / "parent_00000.jsonl"
    lineage.write_text("{}\n")
    parent = _signed(
        {
            "schema": PARENT_SCHEMA,
            "ordinal": 0,
            "source": {"path": "upstream.parquet", "sha256": "a" * 64},
            "counts": {"materialized_rows": 1},
            "lineage": {
                "path": "lineage/parent_00000.jsonl",
                "bytes": lineage.stat().st_size,
                "sha256": sha256_file(lineage),
            },
            "output": {
                "path": "data/parent_00000.parquet",
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
                "rows": 1,
            },
            "training_ready": False,
        }
    )
    (root / "parents" / "parent_00000.json").write_text(json.dumps(parent))
    shard = _signed(
        {
            "schema": SHARD_SCHEMA,
            "logical_shards": 1,
            "shard_index": 0,
            "assigned_parents": 1,
            "training_ready": False,
        }
    )
    (root / "receipt.json").write_text(json.dumps(shard))
    result = build_audit(
        materialized, tmp_path / "audit.json", logical_shards=1
    )
    assert result["sampled_rows"] == 1
    assert result["counts"] == {"decision::pass_mechanical_gate": 1}
    assert result["source_text_persisted"] is False
    assert text not in json.dumps(result)
    assert result["rows"][0]["source_content_sha256"] == hashlib.sha256(
        text.encode()
    ).hexdigest()
    assert result["training_ready"] is False
