import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data.pleias_bounded_mechanical_candidates import CANDIDATE_SCHEMA
from sai.data.pleias_subdocument_decision import build_decision
from sai.data.pleias_subdocument_signature import _schema, signature_rows
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _candidate(identity, unique, repeated):
    import hashlib

    text = f"{unique}\n\n{repeated}"
    return {
        "schema": CANDIDATE_SCHEMA,
        "source_row_identity_sha256": identity,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "collection": "Books",
        "text": text,
        "training_ready": False,
    }


def _signature_shard(root: Path, shard_index: int, rows):
    shard = root / "shards" / f"shard_{shard_index:05d}"
    shard.mkdir(parents=True)
    import hashlib

    ordered = hashlib.sha256()
    for row in rows:
        ordered.update(bytes.fromhex(row["signature_sha256"]))
    counts = {"signatures": len(rows)}
    outputs = []
    for bucket in range(16):
        selected = [
            row for row in rows if int(row["normalized_sha256"][0], 16) == bucket
        ]
        path = shard / f"bucket-{bucket:02x}.parquet"
        pq.write_table(
            pa.Table.from_pylist(selected, schema=_schema()),
            path,
            compression="zstd",
        )
        digest = hashlib.sha256()
        for row in selected:
            digest.update(bytes.fromhex(row["signature_sha256"]))
        counts[f"bucket_{bucket:02x}_signatures"] = len(selected)
        outputs.append(
            {
                "bucket": bucket,
                "path": path.name,
                "rows": len(selected),
                "ordered_signature_digests_sha256": digest.hexdigest(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    receipt = _signed(
        {
            "schema": "sai-pleias-subdocument-signature-shard-v1",
            "logical_shards": 2,
            "shard_index": shard_index,
            "counts": counts,
            "ordered_signature_digests_sha256": ordered.hexdigest(),
            "outputs": outputs,
            "training_ready": False,
        }
    )
    (shard / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    return receipt


def test_global_decision_emits_text_free_shard_deletions(tmp_path):
    root = tmp_path / "signatures"
    repeated = (
        "A recurring copyright and navigation statement long enough to deduplicate."
    )
    first = signature_rows(
        _candidate("a" * 64, "A unique opening about astronomy.", repeated), 0, 0
    )
    second = signature_rows(
        _candidate("b" * 64, "A unique opening about biology.", repeated), 1, 0
    )
    _signature_shard(root, 0, first)
    _signature_shard(root, 1, second)
    bucket_totals = {
        f"bucket_{bucket:02x}_signatures": sum(
            int(row["normalized_sha256"][0], 16) == bucket for row in [*first, *second]
        )
        for bucket in range(16)
    }
    aggregate = _signed(
        {
            "schema": "sai-pleias-subdocument-signature-aggregate-v1",
            "shards": {"logical_shards": 2},
            "totals": {"signatures": len(first) + len(second), **bucket_totals},
            "complete_materialized_document_coverage": True,
            "source_text_persisted": False,
            "training_ready": False,
        }
    )
    (root / "aggregate.json").write_text(json.dumps(aggregate, sort_keys=True))
    output = tmp_path / "decision"
    common = {row["normalized_sha256"] for row in first}.intersection(
        row["normalized_sha256"] for row in second
    )
    bucket_index = int(next(iter(common))[0], 16)
    result = build_decision(
        root,
        output,
        2,
        bucket_index,
        chunk_records=1,
        maximum_open_runs=2,
        reference_characters=16,
        effective_shards_numerator=2,
        effective_shards_denominator=1,
        temporary_root=tmp_path,
    )
    assert result["counts"]["duplicate_groups"] >= 1
    assert result["counts"]["deletion_occurrences"] >= 1
    assert result["decision_contains_source_text"] is False
    assert repeated not in (output / "receipt.json").read_text()
    assert result["cross_source_subdocument_deduplication_complete"] is False
