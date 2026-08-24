import hashlib
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.pleias_semantic_sample import _token_band
from sai.data.pleias_subdocument_signature import SHARD_SCHEMA
from sai.data.pleias_virtual_subdocument_signature import (
    LOCATOR_SCHEMA,
    VIRTUAL_SHARD_STATUS,
    PleiasVirtualSubdocumentSignatureError,
    aggregate_virtual,
    locator_row,
    run_shard,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _candidate(text: str = "Verified technical prose.") -> dict:
    return {
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "source_path": "common_corpus_0/subset.parquet",
        "source_parent_sha256": "a" * 64,
        "source_row_identity_sha256": "b" * 64,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "token_count": 12,
        "word_count": 6,
        "collection": "Github Open Source",
        "open_type": "Open Culture",
        "license": "Public Domain",
        "language": "English",
        "semantic_stratum": "Github Open Source::Open::short",
        "semantic_quality_floor_milli": 8_000,
        "semantic_quality_mean_milli": 8_500,
        "semantic_difficulty_mean_milli": 2_500,
        "semantic_prerequisite_burden_mean_milli": 2_000,
        "semantic_curriculum_phase": "expansion",
        "semantic_domains": ["computing"],
        "semantic_recurring_concepts": ["testing"],
        "semantic_recurring_prerequisites": ["programming"],
        "text": text,
    }


def test_locator_strips_text_and_binds_reconstruction_identity() -> None:
    row = locator_row(_candidate(), 7, 11)
    assert row["schema"] == LOCATOR_SCHEMA
    assert "text" not in row
    assert row["virtual_row_index"] == 7
    assert row["source_row_index"] == 11
    assert row["text_characters"] == len(_candidate()["text"])
    assert row["code_document"] is True
    assert row["locator_sha256"] == canonical_sha256(
        {key: value for key, value in row.items() if key != "locator_sha256"}
    )


def _write_virtual_shard(
    root: Path,
    shard_index: int,
    selection_receipt_sha256: str,
    selected_rows: int,
    selected_bytes: int,
    retained_rows: int,
) -> None:
    shard = root / f"shard_{shard_index:05d}"
    shard.mkdir(parents=True)
    outputs = []
    counts = {
        "selected_rows_replayed": selected_rows,
        "selected_text_utf8_bytes_replayed": selected_bytes,
        "retained_rows": retained_rows,
        "benchmark_contaminated_rows": selected_rows - retained_rows,
    }
    for bucket in range(16):
        path = shard / f"bucket-{bucket:02x}.parquet"
        path.write_bytes(f"bucket {bucket}".encode())
        counts[f"bucket_{bucket:02x}_signatures"] = 0
        outputs.append(
            {
                "bucket": bucket,
                "path": path.name,
                "rows": 0,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    locator = shard / "retained-locators.parquet"
    locator.write_bytes(b"source-safe locators")
    receipt = {
        "schema": SHARD_SCHEMA,
        "status": VIRTUAL_SHARD_STATUS,
        "logical_shards": 2,
        "shard_index": shard_index,
        "source": {"selection_receipt_sha256": selection_receipt_sha256},
        "counts": counts,
        "outputs": outputs,
        "retained_locators": {
            "path": locator.name,
            "rows": retained_rows,
            "bytes": locator.stat().st_size,
            "sha256": sha256_file(locator),
        },
        "complete_virtual_document_coverage": True,
        "source_text_persisted": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (shard / "receipt.json").write_text(json.dumps(receipt))


def test_virtual_aggregate_replays_exact_selection_coverage(tmp_path: Path) -> None:
    database = tmp_path / "selection.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE selected(text_utf8_bytes INTEGER NOT NULL)")
    connection.executemany("INSERT INTO selected VALUES (?)", [(100,), (200,), (300,)])
    connection.commit()
    connection.close()
    selection = {
        "receipt_sha256": "c" * 64,
        "counts": {"selected_rows": 3, "selected_text_utf8_bytes": 600},
    }
    shards = tmp_path / "shards"
    _write_virtual_shard(shards, 0, selection["receipt_sha256"], 2, 300, 1)
    _write_virtual_shard(shards, 1, selection["receipt_sha256"], 1, 300, 1)
    with patch(
        "sai.data.pleias_virtual_subdocument_signature._selection_database",
        return_value=(selection, database),
    ):
        result = aggregate_virtual(
            tmp_path / "selection", shards, 2, tmp_path / "aggregate.json"
        )
    assert result["totals"]["selected_rows_replayed"] == 3
    assert result["totals"]["retained_rows"] == 2
    assert result["totals"]["benchmark_contaminated_rows"] == 1
    assert result["complete_virtual_document_coverage"] is True
    assert result["source_text_persisted"] is False


def test_virtual_aggregate_rejects_partial_selection_coverage(tmp_path: Path) -> None:
    database = tmp_path / "selection.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE selected(text_utf8_bytes INTEGER NOT NULL)")
    connection.execute("INSERT INTO selected VALUES (100)")
    connection.commit()
    connection.close()
    selection = {"receipt_sha256": "d" * 64}
    shards = tmp_path / "shards"
    _write_virtual_shard(shards, 0, selection["receipt_sha256"], 0, 0, 0)
    _write_virtual_shard(shards, 1, selection["receipt_sha256"], 0, 0, 0)
    with (
        patch(
            "sai.data.pleias_virtual_subdocument_signature._selection_database",
            return_value=(selection, database),
        ),
        pytest.raises(
            PleiasVirtualSubdocumentSignatureError,
            match="selection coverage",
        ),
    ):
        aggregate_virtual(
            tmp_path / "selection", shards, 2, tmp_path / "aggregate.json"
        )


def test_virtual_shard_streams_text_but_persists_only_safe_signatures(
    tmp_path: Path,
) -> None:
    text = (
        "A verified discussion of orbital measurement, telescope calibration, "
        "and reproducible astronomical evidence. "
    ) * 20
    source_row = {
        "identifier": "astronomy-one",
        "collection": "Open Textbooks",
        "open_type": "Open Culture",
        "license": "Public Domain",
        "language": "English",
        "word_count": len(text.split()),
        "token_count": len(text.split()) * 2,
        "text": text,
    }
    source = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pylist([source_row]), source)
    parent = {
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "source_path": "common_corpus_0/source.parquet",
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }
    content_sha256 = hashlib.sha256(text.encode()).hexdigest()
    identity = canonical_sha256(
        {
            "source_path": parent["source_path"],
            "row_index": 0,
            "identifier": source_row["identifier"],
            "content_sha256": content_sha256,
        }
    )
    stratum = "::".join(
        (
            source_row["collection"],
            source_row["open_type"],
            _token_band(source_row["token_count"]),
        )
    )
    database = tmp_path / "selection.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE selected(source_path TEXT, source_row_index INTEGER, "
        "source_row_identity_sha256 TEXT, source_parent_sha256 TEXT, "
        "content_sha256 TEXT, stratum TEXT, text_utf8_bytes INTEGER, "
        "token_count INTEGER, stratum_quality_floor_milli INTEGER, "
        "stratum_quality_mean_milli INTEGER)"
    )
    connection.execute(
        "INSERT INTO selected VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            parent["source_path"],
            0,
            identity,
            parent["sha256"],
            content_sha256,
            stratum,
            len(text.encode()),
            source_row["token_count"],
            8_000,
            8_500,
        ),
    )
    connection.commit()
    connection.close()
    semantic = {
        stratum: {
            "difficulty_mean_milli": 2_500,
            "prerequisite_burden_mean_milli": 2_000,
            "dominant_curriculum_phase": "expansion",
            "domains": ["physics_astronomy"],
            "recurring_concepts": ["orbital measurement"],
            "recurring_prerequisites": ["classical mechanics"],
        }
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("pinned manifest\n")

    class Boundary(set):
        def close(self) -> None:
            return None

    output = tmp_path / "virtual" / "shard_00000"
    output.parent.mkdir()
    with (
        patch(
            "sai.data.pleias_virtual_subdocument_signature.load_manifest",
            return_value=[parent],
        ),
        patch(
            "sai.data.pleias_virtual_subdocument_signature.select_shard",
            return_value=[parent],
        ),
        patch(
            "sai.data.pleias_virtual_subdocument_signature._selection_database",
            return_value=({"receipt_sha256": "c" * 64}, database),
        ),
        patch(
            "sai.data.pleias_virtual_subdocument_signature._semantic_metadata",
            return_value=(semantic, {"receipt_sha256": "d" * 64}),
        ),
        patch(
            "sai.data.pleias_virtual_subdocument_signature.binary_boundary_index",
            return_value=(
                [Boundary()],
                [Boundary()],
                [{"receipt_sha256": "e" * 64}],
            ),
        ),
        patch(
            "sai.data.pleias_virtual_subdocument_signature._download",
            return_value=source,
        ),
    ):
        result = run_shard(
            manifest,
            tmp_path / "selection",
            tmp_path / "semantic.json",
            [tmp_path / "boundary"],
            output,
            1,
            0,
            "token",
            tmp_path,
        )
    assert result["counts"]["selected_rows_replayed"] == 1
    assert result["counts"]["retained_rows"] == 1
    assert result["counts"].get("benchmark_contaminated_rows", 0) == 0
    assert result["source_text_persisted"] is False
    locators = pq.read_table(output / "retained-locators.parquet").to_pylist()
    assert len(locators) == 1
    assert "text" not in locators[0]
    assert locators[0]["source_row_identity_sha256"] == identity
    for path in output.glob("*.parquet"):
        assert text.encode() not in path.read_bytes()
