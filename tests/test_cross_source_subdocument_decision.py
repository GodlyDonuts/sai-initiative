import json

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data.cross_source_subdocument_decision import build_decision
from sai.data.cross_source_subdocument_decision_aggregate import build_aggregate
from sai.data.pleias_subdocument_signature import (
    HASH_BUCKETS,
    _schema,
    signature_rows_for_text,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _root(tmp_path, *, component, identity, shard_schema, aggregate_schema, key):
    text = " ".join(["authoritative knowledge bridge"] * 80)
    import hashlib

    rows = signature_rows_for_text(
        component=component,
        text=text,
        identity=identity,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        source_shard=0,
        source_row_index=0,
        code_document=False,
    )
    assert len(rows) == 1
    bucket = int(rows[0]["normalized_sha256"][0], 16)
    root = tmp_path / component
    shard = root / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    outputs = []
    counts = {"documents": 1, "signatures": 1}
    for index in range(HASH_BUCKETS):
        path = shard / f"bucket-{index:02x}.parquet"
        bucket_rows = rows if index == bucket else []
        pq.write_table(pa.Table.from_pylist(bucket_rows, schema=_schema()), path)
        import hashlib as _hashlib

        ordered = _hashlib.sha256()
        for row in bucket_rows:
            ordered.update(bytes.fromhex(row["signature_sha256"]))
        counts[f"bucket_{index:02x}_signatures"] = len(bucket_rows)
        outputs.append(
            {
                "bucket": index,
                "path": path.name,
                "rows": len(bucket_rows),
                "ordered_signature_digests_sha256": ordered.hexdigest(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    receipt = _signed(
        {
            "schema": shard_schema,
            "status": "complete_nontraining_signatures",
            "logical_shards": 1,
            "shard_index": 0,
            "counts": counts,
            "outputs": outputs,
            "training_ready": False,
        }
    )
    (shard / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    aggregate = _signed(
        {
            "schema": aggregate_schema,
            "status": "complete_nontraining_signatures",
            "shards": {"logical_shards": 1},
            "totals": counts,
            key: True,
            "source_text_persisted": False,
            "training_ready": False,
        }
    )
    (root / "aggregate.json").write_text(json.dumps(aggregate, sort_keys=True))
    return root, bucket


def test_cross_source_decision_prefers_authoritative_book(tmp_path):
    books, book_bucket = _root(
        tmp_path,
        component="institutional_books",
        identity="a" * 64,
        shard_schema="sai-institutional-books-subdocument-signature-shard-v1",
        aggregate_schema=(
            "sai-institutional-books-subdocument-signature-aggregate-v1"
        ),
        key="complete_benchmark_disjoint_book_coverage",
    )
    pleias, pleias_bucket = _root(
        tmp_path,
        component="pleias_common_corpus",
        identity="0" * 64,
        shard_schema="sai-pleias-final-subdocument-signature-shard-v1",
        aggregate_schema="sai-pleias-final-subdocument-signature-aggregate-v1",
        key="complete_final_pleias_document_coverage",
    )
    assert book_bucket == pleias_bucket
    decision_root = tmp_path / "decision"
    result = build_decision(
        books,
        pleias,
        decision_root / "buckets" / f"bucket_{book_bucket:02x}",
        book_bucket,
        book_logical_shards=1,
        pleias_logical_shards=1,
        chunk_records=1,
        maximum_open_runs=2,
        maximum_open_deletion_files=1,
        temporary_root=tmp_path,
    )
    assert result["counts"]["duplicate_groups"] == 1
    descriptors = {
        row["component"]: row for row in result["deletions"]
    }
    assert descriptors["institutional_books"]["rows"] == 0
    assert descriptors["pleias_common_corpus"]["rows"] == 1
    assert result["policy"]["source_priority"][0] == "institutional_books"
    assert result["decision_contains_source_text"] is False
    assert result["cross_source_subdocument_decision_complete"] is True
    assert result["training_ready"] is False
    combined = build_aggregate(
        decision_root,
        decision_root / "aggregate.json",
        bucket_indexes=[book_bucket],
    )
    assert combined["totals"]["deletion_occurrences"] == 1
    assert combined["hash_partition"]["complete"] is False
    assert combined["cross_source_subdocument_decision_complete"] is False
