import hashlib
import io
import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.cross_source_subdocument_decision_aggregate import (
    SCHEMA as CROSS_DECISION_SCHEMA,
)
from sai.data.foundation_source_split import POLICY_SHA256 as SPLIT_POLICY_SHA256
from sai.data.pleias_bounded_mechanical_candidates import CANDIDATE_SCHEMA
from sai.data.pleias_cross_source_subdocument_rewrite import rewrite_row
from sai.data.pleias_final_subdocument_signature import (
    AGGREGATE_SCHEMA as INTERNAL_AGGREGATE_SCHEMA,
)
from sai.data.pleias_final_subdocument_signature import (
    SHARD_SCHEMA as INTERNAL_SHARD_SCHEMA,
)
from sai.data.pleias_semantic_sample import _token_band
from sai.data.pleias_subdocument_rewrite import rewrite_candidate
from sai.data.pleias_virtual_byte_balance import (
    AGGREGATE_SCHEMA as BALANCE_AGGREGATE_SCHEMA,
)
from sai.data.pleias_virtual_byte_balance import SHARD_SCHEMA as BALANCE_SHARD_SCHEMA
from sai.data.pleias_virtual_cross_source_reconstruction import (
    AGGREGATE_STATUS,
    LOCATOR_SCHEMA,
    SHARD_SCHEMA,
    SHARD_STATUS,
    PleiasVirtualCrossSourceReconstructionError,
    _locator_schema,
    aggregate,
    final_locator_row,
    run_shard,
)
from sai.data.pleias_virtual_internal_rewrite_signature import (
    AGGREGATE_STATUS as INTERNAL_AGGREGATE_STATUS,
)
from sai.data.pleias_virtual_internal_rewrite_signature import (
    SHARD_STATUS as INTERNAL_SHARD_STATUS,
)
from sai.data.pleias_virtual_internal_rewrite_signature import (
    _transformed_locator_schema,
    transformed_locator_row,
)
from sai.data.pleias_virtual_subdocument_signature import locator_row
from sai.data.pleias_virtual_transient_stream import (
    ENVELOPE_SCHEMA,
    PleiasVirtualTransientStreamError,
    _internal_locator,
    _locator_database,
    stream_shard,
    training_envelope,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _candidate(text: str) -> dict:
    return {
        "schema": CANDIDATE_SCHEMA,
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
        "training_ready": False,
    }


def _final_locator() -> dict:
    candidate = _candidate("A verified explanation of orbital measurement. " * 20)
    source_locator = locator_row(candidate, 0, 7)
    internal, _internal_counts = rewrite_candidate(candidate, 0, [])
    internal_locator = transformed_locator_row(source_locator, internal)
    final, _cross_counts = rewrite_row(internal, 0, [])
    return final_locator_row(internal_locator, final)


def test_final_locator_binds_both_transforms_without_text() -> None:
    result = _final_locator()
    assert result["schema"] == LOCATOR_SCHEMA
    assert "text" not in result
    assert result["pre_internal_content_sha256"]
    assert result["post_internal_content_sha256"]
    assert result["content_sha256"]
    assert result["corpus_split"] in {"train", "development"}
    assert result["token_count_requires_recomputation"] is True
    assert result["locator_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "locator_sha256"}
    )


def test_final_locator_rejects_mutated_internal_binding() -> None:
    candidate = _candidate("A verified explanation of orbital measurement. " * 20)
    source_locator = locator_row(candidate, 0, 7)
    internal, _counts = rewrite_candidate(candidate, 0, [])
    internal_locator = transformed_locator_row(source_locator, internal)
    internal_locator["content_sha256"] = "f" * 64
    final, _cross_counts = rewrite_row(internal, 0, [])
    with pytest.raises(
        PleiasVirtualCrossSourceReconstructionError,
        match="final locator source differs",
    ):
        final_locator_row(internal_locator, final)


def test_transient_envelope_replays_final_locator_without_persisting_text() -> None:
    text = "A verified explanation of orbital measurement. " * 20
    candidate = _candidate(text)
    source_locator = locator_row(candidate, 0, 7)
    internal, _counts = rewrite_candidate(candidate, 0, [])
    final, _cross_counts = rewrite_row(internal, 0, [])
    locator = final_locator_row(
        transformed_locator_row(source_locator, internal), final
    )
    rebuilt = _internal_locator(locator, candidate, internal)
    assert final_locator_row(rebuilt, final) == locator
    envelope = training_envelope(locator, final["text"], "d" * 64, "e" * 64)
    assert envelope["schema"] == ENVELOPE_SCHEMA
    assert envelope["document"]["text"] == final["text"]
    assert envelope["document"]["verification"]["benchmark_disjoint"] is True
    assert envelope["document"]["source"]["domain"] == "science"
    assert envelope["tokenization_ready"] is True
    with pytest.raises(
        PleiasVirtualTransientStreamError, match="training envelope source differs"
    ):
        training_envelope(locator, final["text"] + "tamper", "d" * 64, "e" * 64)


def test_transient_stream_seals_only_source_text_free_accounting(
    tmp_path: Path,
) -> None:
    text = "A verified explanation of orbital measurement. " * 20
    locator = _final_locator()
    envelope = training_envelope(locator, text, "d" * 64, "e" * 64)
    output = io.StringIO()
    receipt = tmp_path / "receipt.json"
    with patch(
        "sai.data.pleias_virtual_transient_stream.iter_reconstructed_shard",
        return_value=iter([envelope]),
    ):
        result = stream_shard(
            output,
            receipt,
            logical_shards=1,
            shard_index=0,
        )
    assert json.loads(output.getvalue())["document"]["text"] == text
    assert result["counts"]["documents"] == 1
    assert result["source_text_persisted_by_compiler"] is False
    assert text not in receipt.read_text()


def test_transient_locator_database_requires_aggregate_bound_shard(
    tmp_path: Path,
) -> None:
    locator = _final_locator()
    final_root = tmp_path / "final"
    shard_root = final_root / "shards" / "shard_00000"
    shard_root.mkdir(parents=True)
    locators = shard_root / "final-locators.parquet"
    pq.write_table(pa.Table.from_pylist([locator], schema=_locator_schema()), locators)
    shard = {
        "schema": SHARD_SCHEMA,
        "status": SHARD_STATUS,
        "logical_shards": 1,
        "shard_index": 0,
        "counts": {"documents": 1},
        "final_locators": {
            "path": locators.name,
            "rows": 1,
            "bytes": locators.stat().st_size,
            "sha256": sha256_file(locators),
            "ordered_locator_digests_sha256": hashlib.sha256(
                bytes.fromhex(locator["locator_sha256"])
            ).hexdigest(),
        },
        "complete_final_pleias_document_coverage": True,
        "benchmark_decontamination_complete": True,
        "cross_source_subdocument_deduplication_complete": True,
        "source_disjoint_split_complete": True,
        "source_text_persisted": False,
        "training_ready": False,
    }
    _signed(shard_root / "receipt.json", shard)
    aggregate_receipts = canonical_sha256([shard["receipt_sha256"]])
    _signed(
        final_root / "aggregate.json",
        {
            "schema": "sai-pleias-virtual-final-reconstruction-aggregate-v1",
            "status": AGGREGATE_STATUS,
            "shards": {
                "logical_shards": 1,
                "ordered_receipts_sha256": aggregate_receipts,
            },
            "complete_final_pleias_document_coverage": True,
            "benchmark_decontamination_complete": True,
            "cross_source_subdocument_deduplication_complete": True,
            "source_disjoint_split_complete": True,
            "source_text_persisted": False,
            "training_ready": False,
        },
    )
    balance_root = tmp_path / "balance"
    balance_shard_root = balance_root / "shards" / "shard_00000"
    balance_shard_root.mkdir(parents=True)
    exclusions = balance_shard_root / "excluded.sqlite3"
    database = sqlite3.connect(exclusions)
    database.execute(
        "CREATE TABLE excluded (source_row_identity_sha256 TEXT PRIMARY KEY) "
        "WITHOUT ROWID"
    )
    database.commit()
    database.close()
    balance_shard = {
        "schema": BALANCE_SHARD_SCHEMA,
        "status": "complete_nontraining_pleias_virtual_byte_balance_shard",
        "logical_shards": 1,
        "shard_index": 0,
        "selected_counts": {"documents": 1},
        "excluded_rows": 0,
        "exclusion_database": {
            "path": exclusions.name,
            "rows": 0,
            "bytes": exclusions.stat().st_size,
            "sha256": sha256_file(exclusions),
        },
        "source_text_persisted": False,
        "training_ready": False,
    }
    _signed(balance_shard_root / "receipt.json", balance_shard)
    _signed(
        balance_root / "aggregate.json",
        {
            "schema": BALANCE_AGGREGATE_SCHEMA,
            "status": "complete_nontraining_pleias_virtual_byte_balance",
            "shards": {
                "logical_shards": 1,
                "ordered_receipts_sha256": canonical_sha256(
                    [balance_shard["receipt_sha256"]]
                ),
            },
            "source_text_persisted": False,
            "training_ready": False,
        },
    )
    connection, _aggregate, loaded_shard, _balance_shard, rows = _locator_database(
        final_root, balance_root, 1, 0, tmp_path / "locators.sqlite3"
    )
    connection.close()
    assert rows == 1
    assert loaded_shard["receipt_sha256"] == shard["receipt_sha256"]

    shard["counts"]["documents"] = 2
    shard.pop("receipt_sha256")
    _signed(shard_root / "receipt.json", shard)
    with pytest.raises(
        PleiasVirtualTransientStreamError, match="aggregate shard custody differs"
    ):
        _locator_database(final_root, balance_root, 1, 0, tmp_path / "tampered.sqlite3")


def _empty_decision_database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE deletions ("
        "source_row_index INTEGER, document_identity_sha256 TEXT, "
        "chunk_index INTEGER, character_start INTEGER, character_end INTEGER, "
        "normalized_sha256 TEXT, frequency INTEGER, budget INTEGER)"
    )
    return connection


def test_final_reconstruction_streams_text_but_persists_only_locators(
    tmp_path: Path,
) -> None:
    text = "A verified discussion of telescope calibration and evidence. " * 20
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
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "source_repository": parent["source_repository"],
        "source_revision": parent["source_revision"],
        "source_path": parent["source_path"],
        "source_parent_sha256": parent["sha256"],
        "source_row_identity_sha256": identity,
        "content_sha256": content_sha256,
        "token_count": source_row["token_count"],
        "word_count": source_row["word_count"],
        "collection": source_row["collection"],
        "open_type": source_row["open_type"],
        "license": source_row["license"],
        "language": source_row["language"],
        "semantic_stratum": stratum,
        "semantic_quality_floor_milli": 8_000,
        "semantic_quality_mean_milli": 8_500,
        "semantic_difficulty_mean_milli": 2_500,
        "semantic_prerequisite_burden_mean_milli": 2_000,
        "semantic_curriculum_phase": "expansion",
        "semantic_domains": ["physics_astronomy"],
        "semantic_recurring_concepts": ["orbital measurement"],
        "semantic_recurring_prerequisites": ["classical mechanics"],
        "text": text,
        "training_ready": False,
    }
    source_locator = locator_row(candidate, 0, 0)
    internal, _counts = rewrite_candidate(candidate, 0, [])
    internal_locator = transformed_locator_row(source_locator, internal)
    internal_root = tmp_path / "internal"
    internal_shard = internal_root / "shards" / "shard_00000"
    internal_shard.mkdir(parents=True)
    internal_path = internal_shard / "transformed-locators.parquet"
    pq.write_table(
        pa.Table.from_pylist([internal_locator], schema=_transformed_locator_schema()),
        internal_path,
    )
    internal_receipt = {
        "schema": INTERNAL_SHARD_SCHEMA,
        "status": INTERNAL_SHARD_STATUS,
        "logical_shards": 1,
        "shard_index": 0,
        "transformed_locators": {
            "path": internal_path.name,
            "rows": 1,
            "bytes": internal_path.stat().st_size,
            "sha256": sha256_file(internal_path),
            "ordered_locator_digests_sha256": hashlib.sha256(
                bytes.fromhex(internal_locator["locator_sha256"])
            ).hexdigest(),
        },
        "complete_final_pleias_document_coverage": True,
        "source_text_persisted": False,
        "training_ready": False,
    }
    _signed(internal_shard / "receipt.json", internal_receipt)
    selection_database = tmp_path / "selection.sqlite3"
    connection = sqlite3.connect(selection_database)
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
    output = tmp_path / "final" / "shards" / "shard_00000"
    with (
        patch(
            "sai.data.pleias_virtual_cross_source_reconstruction.load_manifest",
            return_value=[parent],
        ),
        patch(
            "sai.data.pleias_virtual_cross_source_reconstruction.select_shard",
            return_value=[parent],
        ),
        patch(
            "sai.data.pleias_virtual_cross_source_reconstruction._selection_database",
            return_value=({"receipt_sha256": "d" * 64}, selection_database),
        ),
        patch(
            "sai.data.pleias_virtual_cross_source_reconstruction._semantic_metadata",
            return_value=(semantic, {"receipt_sha256": "e" * 64}),
        ),
        patch(
            "sai.data.pleias_virtual_cross_source_reconstruction._download",
            return_value=source,
        ),
        patch(
            "sai.data.pleias_virtual_cross_source_reconstruction._decision_database",
            return_value=(_empty_decision_database(), ["f" * 64], 0),
        ),
        patch(
            "sai.data.pleias_virtual_cross_source_reconstruction.decision_database",
            return_value=(_empty_decision_database(), ["1" * 64], 0),
        ),
    ):
        result = run_shard(
            manifest,
            tmp_path / "selection",
            tmp_path / "semantic.json",
            internal_root,
            tmp_path / "internal-decisions",
            tmp_path / "cross-decisions",
            output,
            1,
            0,
            "token",
            tmp_path,
        )
    assert result["counts"]["documents"] == 1
    assert result["cross_source_subdocument_deduplication_complete"] is True
    assert result["source_text_persisted"] is False
    assert result["source_disjoint_split_policy_sha256"] == SPLIT_POLICY_SHA256
    rows = pq.read_table(output / "final-locators.parquet").to_pylist()
    assert len(rows) == 1
    assert "text" not in rows[0]
    assert rows[0]["content_sha256"] == content_sha256
    for path in output.glob("*.parquet"):
        assert text.encode() not in path.read_bytes()


def _signed(path: Path, payload: dict) -> None:
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_final_aggregate_closes_exact_split_and_deletion_coverage(
    tmp_path: Path,
) -> None:
    internal_root = tmp_path / "internal"
    cross_root = tmp_path / "cross"
    shards = tmp_path / "final" / "shards"
    output = tmp_path / "final" / "aggregate.json"
    _signed(
        internal_root / "aggregate.json",
        {
            "schema": INTERNAL_AGGREGATE_SCHEMA,
            "status": INTERNAL_AGGREGATE_STATUS,
            "totals": {"documents": 1},
            "complete_final_pleias_document_coverage": True,
            "source_text_persisted": False,
            "training_ready": False,
        },
    )
    _signed(
        cross_root / "aggregate.json",
        {
            "schema": CROSS_DECISION_SCHEMA,
            "totals": {"component::pleias_common_corpus::deletion_occurrences": 2},
            "cross_source_subdocument_decision_complete": True,
            "decision_contains_source_text": False,
            "training_ready": False,
        },
    )
    shard = shards / "shard_00000"
    shard.mkdir(parents=True)
    locators = shard / "final-locators.parquet"
    locators.write_bytes(b"source-safe final locators")
    _signed(
        shard / "receipt.json",
        {
            "schema": SHARD_SCHEMA,
            "status": SHARD_STATUS,
            "logical_shards": 1,
            "shard_index": 0,
            "counts": {
                "documents": 1,
                "cross::candidate_deletion_chunks": 2,
                "output_text_utf8_bytes": 500,
                "split::train::documents": 1,
                "split::train::text_utf8_bytes": 500,
                "semantic_stratum::reference::documents": 1,
                "quality_floor_milli::8000::documents": 1,
                "difficulty_mean_milli::2500::documents": 1,
                "curriculum_phase::expansion::documents": 1,
                "semantic_domain::science::documents": 1,
            },
            "final_locators": {
                "path": locators.name,
                "rows": 1,
                "bytes": locators.stat().st_size,
                "sha256": sha256_file(locators),
            },
            "complete_final_pleias_document_coverage": True,
            "cross_source_subdocument_deduplication_complete": True,
            "source_disjoint_split_policy_sha256": SPLIT_POLICY_SHA256,
            "physical_train_development_partition_complete": True,
            "semantic_quality_metadata_complete": True,
            "curriculum_metadata_complete": True,
            "source_text_persisted": False,
            "training_ready": False,
        },
    )
    result = aggregate(internal_root, cross_root, shards, 1, output)
    assert result["status"] == AGGREGATE_STATUS
    assert result["totals"]["documents"] == 1
    assert result["cross_source_subdocument_deduplication_complete"] is True
    assert result["source_text_persisted"] is False
