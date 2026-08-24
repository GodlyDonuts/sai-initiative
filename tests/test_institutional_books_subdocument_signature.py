import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.institutional_books_full_decontamination import CLEAN_SCHEMA
from sai.data.institutional_books_materializer import OUTPUT_SCHEMA as BOOK_SCHEMA
from sai.data.institutional_books_subdocument_signature import (
    InstitutionalBooksSubdocumentSignatureError,
    aggregate,
    run_shard,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _inputs(tmp_path):
    filtered = tmp_path / "filtered"
    shard = filtered / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    first_text = (
        "A grounded history of astronomy and measurement.\n\n"
        "A separate chapter connects optics to geometry."
    )
    second_text = "A mechanically clean but semantically held book. " * 5
    rows = [
        {
            "schema": BOOK_SCHEMA,
            "barcode_src": "book-a",
            "text": first_text,
            "source_content_sha256": hashlib.sha256(first_text.encode()).hexdigest(),
            "training_ready": False,
        },
        {
            "schema": BOOK_SCHEMA,
            "barcode_src": "book-b",
            "text": second_text,
            "source_content_sha256": hashlib.sha256(second_text.encode()).hexdigest(),
            "training_ready": False,
        },
    ]
    data = shard / "data.parquet"
    pq.write_table(pa.Table.from_pylist(rows), data)
    shard_receipt = _signed(
        {
            "schema": "sai-institutional-books-mechanical-filter-shard-v1",
            "status": "complete_nontraining_private_book_mechanical_filter_shard",
            "logical_shards": 1,
            "shard_index": 0,
            "retained_rows": 2,
            "output": {
                "path": data.name,
                "rows": 2,
                "bytes": data.stat().st_size,
                "sha256": sha256_file(data),
            },
            "training_ready": False,
        }
    )
    (shard / "receipt.json").write_text(json.dumps(shard_receipt, sort_keys=True))
    filtered_aggregate = _signed(
        {
            "schema": "sai-institutional-books-mechanical-filter-aggregate-v1",
            "status": "complete_nontraining_private_book_mechanical_filter",
            "shards": {"logical_shards": 1},
            "training_ready": False,
        }
    )
    (filtered / "aggregate.json").write_text(
        json.dumps(filtered_aggregate, sort_keys=True)
    )

    decontamination = tmp_path / "decontamination"
    decontamination.mkdir()
    clean = {
        "schema": CLEAN_SCHEMA,
        "candidate_identity_sha256": "a" * 64,
        "source_book_id": "book-a",
        "full_source_content_sha256": rows[0]["source_content_sha256"],
        "benchmark_decontamination_complete": True,
        "global_semantic_deduplication_complete": False,
        "training_ready": False,
    }
    clean["record_sha256"] = canonical_sha256(clean)
    clean_path = decontamination / "benchmark_disjoint_books.jsonl"
    clean_path.write_text(json.dumps(clean, sort_keys=True) + "\n")
    decontamination_receipt = _signed(
        {
            "schema": (
                "sai-institutional-books-full-benchmark-decontamination-receipt-v1"
            ),
            "status": "complete_full_consensus_book_benchmark_decontamination",
            "clean_rows": 1,
            "benchmark_disjoint_books": {
                "path": clean_path.name,
                "rows": 1,
                "bytes": clean_path.stat().st_size,
                "sha256": sha256_file(clean_path),
                "ordered_records_sha256": canonical_sha256(
                    [clean["record_sha256"]]
                ),
            },
            "full_selected_source_population_decontaminated": True,
            "training_ready": False,
        }
    )
    (decontamination / "receipt.json").write_text(
        json.dumps(decontamination_receipt, sort_keys=True)
    )
    return filtered, decontamination, data


def test_signs_only_benchmark_disjoint_books_without_persisting_text(tmp_path):
    filtered, decontamination, _data = _inputs(tmp_path)
    output = tmp_path / "signatures" / "shard_00000"
    receipt = run_shard(filtered, decontamination, output, 1, 0)
    assert receipt["counts"]["filtered_source_rows"] == 2
    assert receipt["counts"]["documents"] == 1
    assert receipt["benchmark_decontamination_complete"] is True
    rows = []
    for path in output.glob("bucket-*.parquet"):
        rows.extend(pq.read_table(path).to_pylist())
    assert rows
    assert all(row["component"] == "institutional_books" for row in rows)
    assert all("text" not in row for row in rows)
    combined = aggregate(
        filtered,
        decontamination,
        tmp_path / "signatures",
        1,
        tmp_path / "signatures" / "aggregate.json",
    )
    assert combined["totals"]["documents"] == 1
    assert combined["complete_benchmark_disjoint_book_coverage"] is True
    assert combined["training_ready"] is False


def test_rejects_clean_book_content_mutation(tmp_path):
    filtered, decontamination, data = _inputs(tmp_path)
    rows = pq.read_table(data).to_pylist()
    rows[0]["text"] += " mutation"
    pq.write_table(pa.Table.from_pylist(rows), data)
    shard_receipt_path = filtered / "shards" / "shard_00000" / "receipt.json"
    receipt = json.loads(shard_receipt_path.read_text())
    receipt.pop("receipt_sha256")
    receipt["output"]["bytes"] = data.stat().st_size
    receipt["output"]["sha256"] = sha256_file(data)
    shard_receipt_path.write_text(json.dumps(_signed(receipt), sort_keys=True))
    with pytest.raises(
        InstitutionalBooksSubdocumentSignatureError, match="full book identity"
    ):
        run_shard(filtered, decontamination, tmp_path / "output", 1, 0)
