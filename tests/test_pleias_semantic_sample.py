import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data.agent_labeling import CANDIDATE_SCHEMA
from sai.data.pleias_bounded_mechanical_candidates import (
    AGGREGATE_SCHEMA as BOUNDED_AGGREGATE_SCHEMA,
)
from sai.data.pleias_bounded_mechanical_candidates import (
    CANDIDATE_SCHEMA as BOUNDED_CANDIDATE_SCHEMA,
)
from sai.data.pleias_bounded_mechanical_candidates import (
    SHARD_SCHEMA as BOUNDED_SHARD_SCHEMA,
)
from sai.data.pleias_semantic_sample import build_population
from sai.data.reservoir_audit_aggregate import load_population
from sai.data.token_stream import canonical_sha256, sha256_file


def _write_signed(path, payload):
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def _bounded_row(parent, index, identifier, collection, text, tokens):
    content_sha256 = hashlib.sha256(text.encode()).hexdigest()
    identity = canonical_sha256(
        {
            "source_path": parent["source_path"],
            "row_index": index,
            "identifier": identifier,
            "content_sha256": content_sha256,
        }
    )
    return {
        "schema": BOUNDED_CANDIDATE_SCHEMA,
        "source_id": "pleias_common_corpus",
        "source_repository": parent["source_repository"],
        "source_revision": parent["source_revision"],
        "source_path": parent["source_path"],
        "source_parent_sha256": parent["sha256"],
        "source_row_index": index,
        "source_row_identity_sha256": identity,
        "identifier": identifier,
        "collection": collection,
        "open_type": "Open Science",
        "license": "CC-BY-4.0",
        "language": "English",
        "word_count": len(text.split()),
        "token_count": tokens,
        "content_sha256": content_sha256,
        "text": text,
        "training_ready": False,
    }


def test_builds_diverse_replayable_compiler_population(tmp_path):
    parent = {
        "source_id": "pleias_common_corpus",
        "source_path": "data/parent.parquet",
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "bytes": 12345,
        "sha256": "b" * 64,
        "raw_source_is_training_ready": False,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(parent, sort_keys=True) + "\n")
    bounded = tmp_path / "bounded"
    shard = bounded / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    prose = "A careful scientific explanation with grounded details. " * 80
    rows = [
        _bounded_row(parent, 0, "one", "Science A", prose, 400),
        _bounded_row(parent, 1, "two", "Science A", prose + "Two", 2_000),
        _bounded_row(parent, 2, "three", "Science B", prose + "Three", 40_000),
    ]
    candidate_path = shard / "candidates.parquet"
    pq.write_table(pa.Table.from_pylist(rows), candidate_path, compression="zstd")
    shard_receipt = _write_signed(
        shard / "receipt.json",
        {
            "schema": BOUNDED_SHARD_SCHEMA,
            "logical_shards": 1,
            "shard_index": 0,
            "source": {
                "selected_paths_sha256": canonical_sha256([parent["source_path"]])
            },
            "output": {
                "path": candidate_path.name,
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
            },
            "training_ready": False,
        },
    )
    aggregate_path = bounded / "aggregate.json"
    _write_signed(
        aggregate_path,
        {
            "schema": BOUNDED_AGGREGATE_SCHEMA,
            "status": "complete_nontraining_bounded_pleias_mechanical_candidates",
            "shards": {
                "logical_shards": 1,
                "ordered_receipts_sha256": canonical_sha256(
                    [shard_receipt["receipt_sha256"]]
                ),
            },
            "totals": {
                "selected_rows": 3,
                "output_bytes": candidate_path.stat().st_size,
            },
            "complete_source_parent_coverage": True,
            "semantic_admission_complete": False,
            "training_ready": False,
        },
    )
    output = tmp_path / "semantic"
    receipt = build_population(
        manifest,
        bounded,
        aggregate_path,
        output,
        logical_shards=1,
        maximum_rows=3,
        maximum_rows_per_stratum=2,
    )
    assert receipt["population"]["rows"] == 3
    assert receipt["available_strata"] == 3
    assert receipt["hermes_judgments_complete"] is False
    assert receipt["training_ready"] is False
    candidates, lineage, replay = load_population(output)
    assert len(candidates) == len(lineage) == 3
    assert all(row["schema"] == CANDIDATE_SCHEMA for row in candidates)
    assert replay["receipt_sha256"] == receipt["receipt_sha256"]
    assert {row["stratum"].rsplit("::", 1)[-1] for row in lineage} == {
        "lt512",
        "512to4095",
        "ge32768",
    }
