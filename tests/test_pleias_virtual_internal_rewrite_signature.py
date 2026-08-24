import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.pleias_bounded_mechanical_candidates import CANDIDATE_SCHEMA
from sai.data.pleias_subdocument_rewrite import rewrite_candidate
from sai.data.pleias_subdocument_signature import SHARD_SCHEMA
from sai.data.pleias_virtual_internal_rewrite_signature import (
    TRANSFORMED_LOCATOR_SCHEMA,
    PleiasVirtualInternalRewriteSignatureError,
    _locator_database,
    transformed_locator_row,
)
from sai.data.pleias_virtual_subdocument_signature import (
    VIRTUAL_SHARD_STATUS,
    _locator_schema,
    locator_row,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _candidate(text: str) -> dict:
    return {
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "source_path": "common_corpus_0/source.parquet",
        "source_parent_sha256": "b" * 64,
        "source_row_identity_sha256": "c" * 64,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_count": len(text.split()) * 2,
        "word_count": len(text.split()),
        "collection": "Open Textbooks",
        "open_type": "Open Culture",
        "license": "Public Domain",
        "language": "English",
        "semantic_stratum": "Open Textbooks::Open Culture::medium",
        "semantic_quality_floor_milli": 8_000,
        "semantic_quality_mean_milli": 8_500,
        "semantic_difficulty_mean_milli": 3_000,
        "semantic_prerequisite_burden_mean_milli": 2_000,
        "semantic_curriculum_phase": "expansion",
        "semantic_domains": ["physics_astronomy"],
        "semantic_recurring_concepts": ["measurement"],
        "semantic_recurring_prerequisites": ["algebra"],
        "text": text,
    }


def test_transformed_locator_binds_rewrite_without_persisting_text() -> None:
    text = "A source-grounded explanation of reproducible measurement. " * 8
    candidate = _candidate(text)
    source_locator = locator_row(candidate, 0, 4)
    rewrite_input = {**candidate, "schema": CANDIDATE_SCHEMA, "training_ready": False}
    rewritten, _counts = rewrite_candidate(rewrite_input, 0, [])
    result = transformed_locator_row(source_locator, rewritten)
    assert result["schema"] == TRANSFORMED_LOCATOR_SCHEMA
    assert "text" not in result
    assert result["pre_internal_content_sha256"] == candidate["content_sha256"]
    assert result["content_sha256"] == candidate["content_sha256"]
    assert result["output_text_characters"] == len(text)
    assert result["locator_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "locator_sha256"}
    )


def _write_locator_shard(root: Path, locator: dict) -> Path:
    shard = root / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    path = shard / "retained-locators.parquet"
    pq.write_table(pa.Table.from_pylist([locator], schema=_locator_schema()), path)
    receipt = {
        "schema": SHARD_SCHEMA,
        "status": VIRTUAL_SHARD_STATUS,
        "logical_shards": 1,
        "shard_index": 0,
        "counts": {"retained_rows": 1},
        "retained_locators": {
            "path": path.name,
            "rows": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "ordered_locator_digests_sha256": hashlib.sha256(
                bytes.fromhex(locator["locator_sha256"])
            ).hexdigest(),
        },
        "complete_virtual_document_coverage": True,
        "source_text_persisted": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (shard / "receipt.json").write_text(json.dumps(receipt))
    return path


def test_locator_database_verifies_exact_ordered_coverage(tmp_path: Path) -> None:
    locator = locator_row(_candidate("Verified source text. " * 20), 0, 4)
    root = tmp_path / "virtual"
    _write_locator_shard(root, locator)
    connection, receipt, rows = _locator_database(
        root, 0, 1, tmp_path / "locators.sqlite3"
    )
    try:
        stored = connection.execute(
            "SELECT virtual_row_index, source_path, source_row_index, content_sha256 "
            "FROM locators"
        ).fetchone()
    finally:
        connection.close()
    assert rows == 1
    assert receipt["receipt_sha256"]
    assert stored == (
        0,
        locator["source_path"],
        4,
        locator["content_sha256"],
    )


def test_locator_database_rejects_mutated_locator(tmp_path: Path) -> None:
    locator = locator_row(_candidate("Verified source text. " * 20), 0, 4)
    root = tmp_path / "virtual"
    path = _write_locator_shard(root, locator)
    mutated = dict(locator)
    mutated["content_sha256"] = "f" * 64
    pq.write_table(pa.Table.from_pylist([mutated], schema=_locator_schema()), path)
    receipt_path = path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["retained_locators"]["bytes"] = path.stat().st_size
    receipt["retained_locators"]["sha256"] = sha256_file(path)
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(
        PleiasVirtualInternalRewriteSignatureError,
        match="virtual locator row differs",
    ):
        _locator_database(root, 0, 1, tmp_path / "mutated.sqlite3")
