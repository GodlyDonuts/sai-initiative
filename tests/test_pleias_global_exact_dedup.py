import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data.pleias_full_candidate_decontamination import (
    AGGREGATE_SCHEMA as DECONTAMINATION_AGGREGATE_SCHEMA,
)
from sai.data.pleias_full_candidate_decontamination import (
    SHARD_SCHEMA as DECONTAMINATION_SHARD_SCHEMA,
)
from sai.data.pleias_global_exact_dedup import (
    aggregate_filters,
    build_decision,
    run_filter_shard,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(path, payload):
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def _source_shard(root, shard_index, rows, bounded_receipt):
    shard = root / "shards" / f"shard_{shard_index:05d}"
    shard.mkdir(parents=True)
    parquet_path = shard / "benchmark_disjoint_candidates.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path, compression="zstd")
    index_rows = [
        {
            "content_sha256": row["content_sha256"],
            "source_row_identity_sha256": row["source_row_identity_sha256"],
            "shard_index": shard_index,
            "source_row_index": row["source_row_index"],
            "stratum": "science::Open Science::512to4095",
        }
        for row in rows
    ]
    index_path = shard / "global_exact_dedup_index.jsonl"
    index_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows)
    )
    digest = hashlib.sha256()
    for row in index_rows:
        digest.update(bytes.fromhex(canonical_sha256(row)))
    return _signed(
        shard / "receipt.json",
        {
            "schema": DECONTAMINATION_SHARD_SCHEMA,
            "logical_shards": 2,
            "shard_index": shard_index,
            "source": {
                "bounded_aggregate_receipt_sha256": bounded_receipt,
            },
            "counts": {"retained_candidates": len(rows)},
            "output": {
                "path": parquet_path.name,
                "rows": len(rows),
                "bytes": parquet_path.stat().st_size,
                "sha256": sha256_file(parquet_path),
            },
            "global_exact_dedup_index": {
                "path": index_path.name,
                "rows": len(rows),
                "bytes": index_path.stat().st_size,
                "sha256": sha256_file(index_path),
                "ordered_row_digests_sha256": digest.hexdigest(),
            },
            "training_ready": False,
        },
    )


def _row(identity, text, index):
    return {
        "source_row_identity_sha256": identity,
        "source_row_index": index,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "training_ready": False,
    }


def test_builds_and_applies_one_global_exact_representative(tmp_path):
    source = tmp_path / "source"
    bounded_receipt = "a" * 64
    duplicate_text = "A duplicate scientific document. " * 30
    shard_receipts = [
        _source_shard(
            source,
            0,
            [_row("1" * 64, duplicate_text, 0), _row("3" * 64, "Unique. " * 80, 1)],
            bounded_receipt,
        ),
        _source_shard(
            source,
            1,
            [_row("2" * 64, duplicate_text, 0)],
            bounded_receipt,
        ),
    ]
    aggregate_path = source / "aggregate.json"
    source_aggregate = _signed(
        aggregate_path,
        {
            "schema": DECONTAMINATION_AGGREGATE_SCHEMA,
            "status": "complete_nontraining_pleias_full_candidate_decontamination",
            "source": {
                "bounded_aggregate_receipt_sha256": bounded_receipt,
            },
            "shards": {
                "logical_shards": 2,
                "ordered_receipts_sha256": canonical_sha256(
                    [row["receipt_sha256"] for row in shard_receipts]
                ),
            },
            "totals": {"retained_candidates": 3},
            "full_candidate_benchmark_decontamination_complete": True,
            "global_exact_deduplication_complete": False,
            "training_ready": False,
        },
    )
    decision_root = tmp_path / "decision"
    decision = build_decision(source, aggregate_path, decision_root, 2)
    assert decision["counts"] == {
        "source_rows": 3,
        "unique_content_rows": 2,
        "global_exact_duplicate_rows": 1,
    }
    filtered = tmp_path / "filtered"
    results = [
        run_filter_shard(
            source,
            aggregate_path,
            decision_root,
            filtered / f"shard_{index:05d}",
            2,
            index,
        )
        for index in range(2)
    ]
    assert results[0]["counts"]["retained_rows"] == 2
    assert results[1]["counts"]["global_exact_duplicate_rows"] == 1
    final = aggregate_filters(
        aggregate_path,
        decision_root,
        filtered,
        2,
        filtered / "aggregate.json",
    )
    assert final["totals"]["retained_rows"] == 2
    assert final["global_exact_deduplication_complete"] is True
    assert final["global_near_deduplication_complete"] is False
    assert final["training_ready"] is False
    assert (
        final["source"]["decontamination_aggregate_receipt_sha256"]
        == (source_aggregate["receipt_sha256"])
    )
