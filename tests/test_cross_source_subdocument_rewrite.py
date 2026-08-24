import hashlib
import json

import pytest

from sai.data.cross_source_subdocument_decision import (
    DELETE_RECORD,
)
from sai.data.cross_source_subdocument_decision import (
    SCHEMA as DECISION_SCHEMA,
)
from sai.data.cross_source_subdocument_decision_aggregate import (
    SCHEMA as AGGREGATE_SCHEMA,
)
from sai.data.cross_source_subdocument_rewrite import (
    CrossSourceSubdocumentRewriteError,
    decision_database,
    rewrite_text,
)
from sai.data.frequency_length_subdocument_deduplication import (
    _normalized_chunk,
    segment_subdocuments,
)
from sai.data.pleias_subdocument_signature import HASH_BUCKETS
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def test_rewrite_text_removes_only_verified_coherent_chunk():
    unique = "A unique discussion of optics and laboratory evidence. " * 8
    repeated = "A repetitive archive footer with boilerplate wording. " * 8
    text = f"{unique}\n\n{repeated}"
    identity = "a" * 64
    chunks = segment_subdocuments(text)
    decisions = []
    for chunk_index in (len(chunks) - 2, len(chunks) - 1):
        chunk = chunks[chunk_index]
        normalized = _normalized_chunk(chunk["text"], code=chunk["code"])
        decisions.append(
            (
                identity,
                chunk_index,
                chunk["character_start"],
                chunk["character_end"],
                hashlib.sha256(normalized.encode()).hexdigest(),
                12,
                1,
            )
        )
    rewritten, counts, transform = rewrite_text(
        text=text,
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        identity=identity,
        source_row_index=7,
        decisions=decisions,
        code_document=False,
    )
    assert len(rewritten) < len(text)
    assert counts["deleted_chunks"] == 2
    assert len(transform) == 64

    mutated = list(decisions[0])
    mutated[4] = "f" * 64
    with pytest.raises(CrossSourceSubdocumentRewriteError, match="chunk replay"):
        rewrite_text(
            text=text,
            content_sha256=hashlib.sha256(text.encode()).hexdigest(),
            identity=identity,
            source_row_index=7,
            decisions=[tuple(mutated)],
            code_document=False,
        )


def test_decision_database_replays_all_buckets_and_exact_partition(tmp_path):
    root = tmp_path / "decision"
    identity = bytes.fromhex("a" * 64)
    normalized = bytes.fromhex("b" * 64)
    expected = ("a" * 64, 4, 10, 350, "b" * 64, 9, 1)
    receipt_hashes = []
    for bucket in range(HASH_BUCKETS):
        bucket_root = root / "buckets" / f"bucket_{bucket:02x}"
        deletion_root = bucket_root / "deletions" / "pleias_common_corpus"
        deletion_root.mkdir(parents=True)
        path = deletion_root / "shard_00000.deletions.bin"
        if bucket == 11:
            path.write_bytes(
                DELETE_RECORD.pack(1, 0, 3, identity, 4, 10, 350, normalized, 9, 1)
            )
        else:
            path.write_bytes(b"")
        descriptor = {
            "component": "pleias_common_corpus",
            "component_priority": 1,
            "source_shard": 0,
            "path": "pleias_common_corpus/shard_00000.deletions.bin",
            "rows": 1 if bucket == 11 else 0,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        receipt = _signed(
            {
                "schema": DECISION_SCHEMA,
                "hash_bucket": {"index": bucket, "buckets": HASH_BUCKETS},
                "deletions": [descriptor],
                "cross_source_subdocument_decision_complete": True,
                "decision_contains_source_text": False,
                "training_ready": False,
            }
        )
        (bucket_root / "receipt.json").write_text(json.dumps(receipt))
        receipt_hashes.append(receipt["receipt_sha256"])
    aggregate = _signed(
        {
            "schema": AGGREGATE_SCHEMA,
            "hash_partition": {
                "complete": True,
                "required_buckets": HASH_BUCKETS,
            },
            "components": [
                {
                    "component": "pleias_common_corpus",
                    "priority": 1,
                    "logical_shards": 1,
                }
            ],
            "cross_source_subdocument_decision_complete": True,
            "decision_contains_source_text": False,
            "training_ready": False,
        }
    )
    root.mkdir(exist_ok=True)
    (root / "aggregate.json").write_text(json.dumps(aggregate))
    connection, receipts, total = decision_database(
        root, "pleias_common_corpus", 1, 0, 1, tmp_path / "decisions.sqlite3"
    )
    try:
        rows = connection.execute(
            "SELECT document_identity_sha256, chunk_index, character_start, "
            "character_end, normalized_sha256, frequency, budget "
            "FROM deletions WHERE source_row_index=3"
        ).fetchall()
    finally:
        connection.close()
    assert rows == [expected]
    assert receipts == receipt_hashes
    assert total == 1
