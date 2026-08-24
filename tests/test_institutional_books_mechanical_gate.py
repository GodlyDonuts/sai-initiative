import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data.institutional_books_materializer import (
    AGGREGATE_SCHEMA as MATERIALIZER_AGGREGATE_SCHEMA,
)
from sai.data.institutional_books_materializer import (
    OUTPUT_SCHEMA,
    PARENT_SCHEMA,
)
from sai.data.institutional_books_materializer import (
    SHARD_SCHEMA as MATERIALIZER_SHARD_SCHEMA,
)
from sai.data.institutional_books_mechanical_gate import aggregate, run_shard
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(payload):
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_scans_complete_private_book_custody(tmp_path):
    materialized = tmp_path / "materialized"
    shard_root = materialized / "shards" / "shard_00000"
    data_path = shard_root / "data" / "parent_00000.parquet"
    data_path.parent.mkdir(parents=True)
    good = "A coherent book paragraph with ordinary language. " * 20
    corrupt = ("Useful beginning. " * 20) + ("\ufffd" * 20)
    rows = []
    for barcode, text in (("good", good), ("corrupt", corrupt)):
        rows.append(
            {
                "schema": OUTPUT_SCHEMA,
                "barcode_src": barcode,
                "text": text,
                "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "training_ready": False,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), data_path, compression="zstd")
    parent = _signed(
        {
            "schema": PARENT_SCHEMA,
            "counts": {"materialized_rows": 2},
            "source": {"path": "train/parent.parquet"},
            "output": {
                "path": "data/parent_00000.parquet",
                "bytes": data_path.stat().st_size,
                "sha256": sha256_file(data_path),
                "rows": 2,
            },
            "training_ready": False,
        }
    )
    _write_json(shard_root / "parents" / "parent_00000.json", parent)
    source_shard = _signed(
        {
            "schema": MATERIALIZER_SHARD_SCHEMA,
            "logical_shards": 1,
            "shard_index": 0,
            "counts": {"materialized_rows": 2},
            "training_ready": False,
        }
    )
    _write_json(shard_root / "receipt.json", source_shard)
    source_aggregate = _signed(
        {
            "schema": MATERIALIZER_AGGREGATE_SCHEMA,
            "counts": {"materialized_rows": 2},
            "training_ready": False,
        }
    )
    _write_json(materialized / "aggregate.json", source_aggregate)
    output = tmp_path / "gate"
    shard = run_shard(materialized, output, 1, 0)
    assert shard["decision_counts"] == {
        "hard_reject": 1,
        "pass_mechanical_gate": 1,
    }
    result = aggregate(materialized, output, 1, output / "aggregate.json")
    assert result["all_rows_accounted"] is True
    assert result["decision_counts"] == shard["decision_counts"]
    assert result["training_ready"] is False
