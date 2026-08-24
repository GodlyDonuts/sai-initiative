import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data import pleias_final_subdocument_signature as signatures
from sai.data.pleias_final_subdocument_signature import (
    aggregate,
    final_signature_rows,
    run_shard,
)
from sai.data.pleias_subdocument_rewrite import OUTPUT_SCHEMA
from sai.data.token_stream import canonical_sha256


def _signed(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _row(text):
    return {
        "schema": OUTPUT_SCHEMA,
        "source_row_identity_sha256": "a" * 64,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "collection": "Books",
        "text": text,
        "training_ready": False,
    }


def test_final_rows_bind_post_rewrite_content_without_text():
    text = "A final astronomy chapter.\n\nA bridge from optics to geometry."
    rows = final_signature_rows(_row(text), 4, 9)
    assert rows
    assert all(row["component"] == "pleias_common_corpus" for row in rows)
    assert all(row["source_shard"] == 4 for row in rows)
    assert all(row["source_row_index"] == 9 for row in rows)
    assert all("text" not in row for row in rows)


def test_shard_and_aggregate_replay_rewritten_remote_identity(tmp_path, monkeypatch):
    text = "A final astronomy chapter.\n\nA bridge from optics to geometry."
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([_row(text)]), source)
    rewritten = tmp_path / "rewritten"
    shard = rewritten / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    receipt = _signed(
        {
            "schema": "sai-pleias-subdocument-rewritten-shard-v1",
            "status": "complete_nontraining_pleias_subdocument_rewritten_shard",
            "logical_shards": 1,
            "shard_index": 0,
            "counts": {
                "documents": 1,
                "output_text_utf8_bytes": len(text.encode()),
            },
            "remote_output": {
                "repository": "Godlydonuts/Sai",
                "path": "final/nontraining/pleias/test.parquet",
                "bytes": source.stat().st_size,
                "sha256": "b" * 64,
            },
            "local_payload_removed_after_remote_verification": True,
            "pleias_global_subdocument_rewrite_complete": True,
            "training_ready": False,
        }
    )
    (shard / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    monkeypatch.setattr(signatures, "_download", lambda *_args: source)
    output = tmp_path / "signatures" / "shard_00000"
    result = run_shard(rewritten, output, 1, 0, "token", tmp_path)
    assert result["counts"]["documents"] == 1
    assert result["pleias_internal_subdocument_deduplication_complete"] is True
    combined = aggregate(
        rewritten,
        tmp_path / "signatures",
        1,
        tmp_path / "signatures" / "aggregate.json",
    )
    assert combined["totals"]["documents"] == 1
    assert combined["complete_final_pleias_document_coverage"] is True
    assert combined["training_ready"] is False
