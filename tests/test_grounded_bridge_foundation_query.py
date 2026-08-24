import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from sai.data.grounded_bridge_curriculum_candidates import (
    RECEIPT_SCHEMA as CANDIDATE_RECEIPT_SCHEMA,
)
from sai.data.grounded_bridge_curriculum_candidates import (
    ROW_SCHEMA as CANDIDATE_ROW_SCHEMA,
)
from sai.data.grounded_bridge_curriculum_candidates import (
    STATUS as CANDIDATE_STATUS,
)
from sai.data.grounded_bridge_foundation_query import (
    DATABASE_SCHEMA,
    SCHEMA,
    build_query,
    source_key,
)
from sai.data.grounded_bridge_foundation_scan import (
    ANCHOR_MATCH_SCHEMA,
    FoundationDocument,
    GroundedBridgeFoundationScanError,
    QueryBoundary,
    scan_documents,
    source_key_aliases,
)
from sai.data.grounded_bridge_foundation_scan_aggregate import (
    DOCUMENT_DECISION_SCHEMA,
    PAIR_DECISION_SCHEMA,
    aggregate_scans,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _candidate(text: str, document_index: int) -> dict:
    pair = "1" * 64
    row = {
        "schema": CANDIDATE_ROW_SCHEMA,
        "pair_identity_sha256": pair,
        "document_identity_sha256": f"{document_index + 10:064x}",
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "normalized_content_sha256": hashlib.sha256(
            " ".join(text.casefold().split()).encode()
        ).hexdigest(),
        "text": text,
        "corpus_split": "train",
        "anchor_candidate_identity_sha256s": ["2" * 64, "3" * 64],
        "anchor_source_content_sha256s": ["4" * 64, "5" * 64],
        "anchor_sources": [
            {"dataset": "source/a", "revision": "r1", "row_id": "a"},
            {"dataset": "source/b", "revision": "r2", "row_id": "b"},
        ],
        "bridge_pair_disjoint_split_complete": True,
        "source_disjoint_against_foundation_complete": False,
        "global_deduplication_against_foundation_complete": False,
        "bridge_verified": False,
        "training_ready": False,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def _candidate_root(root: Path) -> None:
    root.mkdir()
    rows = [
        _candidate(
            "Ratios connect musical intervals and fractions through shared "
            "relational structure while frequency multiplication preserves "
            "ordered harmonic relationships across octave transformations.",
            0,
        ),
        _candidate(
            "The analogy stops where perception differs from exact arithmetic "
            "equality.",
            1,
        ),
    ]
    path = root / "curriculum_candidates.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    receipt = {
        "schema": CANDIDATE_RECEIPT_SCHEMA,
        "status": CANDIDATE_STATUS,
        "source_disjoint_against_foundation_complete": False,
        "global_deduplication_against_foundation_complete": False,
        "training_ready": False,
        "curriculum_candidates": {
            "path": path.name,
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "ordered_records_sha256": canonical_sha256(
                [row["record_sha256"] for row in rows]
            ),
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (root / "receipt.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")


def test_source_key_uses_only_stable_source_coordinates() -> None:
    first = source_key(
        {
            "dataset": "source/a",
            "revision": "r1",
            "row_id": "row",
            "license": "CC0",
        }
    )
    second = source_key({"dataset": "source/a", "revision": "r1", "row_id": "row"})
    assert first == second


def test_query_is_source_text_free_and_binds_every_document(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    _candidate_root(candidates)
    output = tmp_path / "query"
    result = build_query(candidates, output)
    assert result["schema"] == SCHEMA
    assert result["counts"]["documents"] == 2
    assert result["counts"]["pairs"] == 1
    assert result["counts"]["anchors"] == 2
    assert result["source_text_persisted"] is False
    assert result["foundation_scan_complete"] is False
    assert result["training_ready"] is False
    database_path = output / "queries.sqlite3"
    assert b"Ratios connect" not in database_path.read_bytes()
    database = sqlite3.connect(database_path)
    try:
        assert database.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 2
        assert database.execute("SELECT COUNT(*) FROM anchors").fetchone()[0] == 2
        assert (
            json.loads(
                database.execute(
                    "SELECT value FROM metadata WHERE key='schema'"
                ).fetchone()[0]
            )
            == DATABASE_SCHEMA
        )
    finally:
        database.close()


def test_query_is_reproducible_and_rejects_tampered_database(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates"
    _candidate_root(candidates)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = build_query(candidates, first)
    second_result = build_query(candidates, second)
    assert first_result == second_result
    assert (first / "queries.sqlite3").read_bytes() == (
        second / "queries.sqlite3"
    ).read_bytes()
    path = first / "queries.sqlite3"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(GroundedBridgeFoundationScanError):
        QueryBoundary(first)


def test_source_aliases_reconcile_separate_and_joined_revisions() -> None:
    separate = source_key_aliases(
        {"dataset": "source/a", "revision": "r1", "row_id": "row"}
    )
    joined = source_key_aliases(
        {"dataset": "source/a@r1", "revision": "", "row_id": "row"}
    )
    assert separate == joined


def test_scan_finds_exact_overlap_and_anchor_without_persisting_text(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates"
    _candidate_root(candidates)
    query = tmp_path / "query"
    build_query(candidates, query)
    shared = (
        "Ratios connect musical intervals and fractions through shared "
        "relational structure while frequency multiplication preserves "
        "ordered harmonic relationships across octave transformations."
    )
    documents = [
        FoundationDocument(
            component="pleias_common_corpus",
            document_identity_sha256="6" * 64,
            text=f"A source preface. {shared} A source conclusion.",
            corpus_split="development",
            source_group_sha256="7" * 64,
            source={
                "dataset": "source/a@r1",
                "revision": "",
                "row_id": "a",
            },
            source_content_sha256s=("4" * 64, "8" * 64),
            source_custody_sha256="9" * 64,
        ),
        FoundationDocument(
            component="pleias_common_corpus",
            document_identity_sha256="a" * 64,
            text="An unrelated foundation document with enough independent material.",
            corpus_split="train",
            source_group_sha256="b" * 64,
            source={"dataset": "other", "revision": "r2", "row_id": "b"},
            source_content_sha256s=("c" * 64,),
            source_custody_sha256="d" * 64,
        ),
    ]
    output = tmp_path / "scan"
    result = scan_documents(
        query,
        documents,
        output,
        component="pleias_common_corpus",
        logical_shards=2,
        shard_index=0,
        source_custody={"final_shard_receipt_sha256": "e" * 64},
    )
    assert result["counts"]["documents"] == 2
    assert result["counts"]["documents_with_word_overlap"] == 1
    assert result["counts"]["unique_matched_word_signatures"] > 0
    assert result["counts"]["anchor_match_records"] == 1
    assert result["source_text_persisted"] is False
    match = json.loads((output / "anchor_matches.jsonl").read_text())
    assert match["schema"] == ANCHOR_MATCH_SCHEMA
    assert match["foundation_split"] == "development"
    assert match["match_types"] == [
        "source_content_sha256",
        "source_key_sha256",
    ]
    persisted = b"".join(path.read_bytes() for path in output.iterdir())
    assert b"Ratios connect musical intervals" not in persisted


def test_aggregate_reconciles_anchor_split_and_holds_only_overlap(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates"
    _candidate_root(candidates)
    query = tmp_path / "query"
    build_query(candidates, query)
    pleias = tmp_path / "pleias" / "shards" / "shard_00000"
    books = tmp_path / "books" / "shards" / "shard_00000"
    scan_documents(
        query,
        [
            FoundationDocument(
                component="pleias_common_corpus",
                document_identity_sha256="6" * 64,
                text=(
                    "A source preface. Ratios connect musical intervals and "
                    "fractions through shared relational structure while "
                    "frequency multiplication preserves ordered harmonic "
                    "relationships across octave transformations."
                ),
                corpus_split="development",
                source_group_sha256="7" * 64,
                source={
                    "dataset": "source/a@r1",
                    "revision": "",
                    "row_id": "a",
                },
                source_content_sha256s=("4" * 64,),
                source_custody_sha256="8" * 64,
            )
        ],
        pleias,
        component="pleias_common_corpus",
        logical_shards=1,
        shard_index=0,
        source_custody={"final_shard_receipt_sha256": "9" * 64},
    )
    scan_documents(
        query,
        [
            FoundationDocument(
                component="institutional_books",
                document_identity_sha256="a" * 64,
                text="A distinct book passage covering unrelated archival details.",
                corpus_split="train",
                source_group_sha256="b" * 64,
                source={"dataset": "books", "revision": "r1", "row_id": "book"},
                source_content_sha256s=("c" * 64,),
                source_custody_sha256="d" * 64,
            )
        ],
        books,
        component="institutional_books",
        logical_shards=1,
        shard_index=0,
        source_custody={"final_shard_receipt_sha256": "e" * 64},
    )
    output = tmp_path / "aggregate"
    result = aggregate_scans(
        query,
        tmp_path / "pleias",
        tmp_path / "books",
        output,
        pleias_logical_shards=1,
        book_logical_shards=1,
    )
    assert result["global_foundation_scan_complete"] is True
    assert result["global_deduplication_against_foundation_complete"] is True
    assert result["source_disjoint_against_foundation_complete"] is True
    assert result["positive_transfer_ablation_complete"] is False
    document_rows = [
        json.loads(line)
        for line in (output / "document_decisions.jsonl").read_text().splitlines()
    ]
    assert all(row["schema"] == DOCUMENT_DECISION_SCHEMA for row in document_rows)
    assert {row["decision"] for row in document_rows} == {
        "hold_exact_foundation_overlap",
        "retain_pending_positive_transfer_ablation",
    }
    assert {row["resolved_split"] for row in document_rows} == {"development"}
    pair = json.loads((output / "pair_decisions.jsonl").read_text())
    assert pair["schema"] == PAIR_DECISION_SCHEMA
    assert pair["resolved_split"] == "development"
    assert pair["representations"] == {"exact_overlap": 1, "retained": 1, "total": 2}
    assert pair["decision"] == "retain_pending_positive_transfer_ablation"
    persisted = b"".join(path.read_bytes() for path in output.iterdir())
    assert b"Ratios connect musical intervals" not in persisted


def test_aggregate_excludes_pair_on_foundation_anchor_split_conflict(
    tmp_path: Path,
) -> None:
    candidates = tmp_path / "candidates"
    _candidate_root(candidates)
    query = tmp_path / "query"
    build_query(candidates, query)
    for component, split, root, identity, group, custody in (
        (
            "pleias_common_corpus",
            "development",
            tmp_path / "pleias" / "shards" / "shard_00000",
            "6" * 64,
            "7" * 64,
            "8" * 64,
        ),
        (
            "institutional_books",
            "train",
            tmp_path / "books" / "shards" / "shard_00000",
            "9" * 64,
            "a" * 64,
            "b" * 64,
        ),
    ):
        scan_documents(
            query,
            [
                FoundationDocument(
                    component=component,
                    document_identity_sha256=identity,
                    text=(
                        "Distinct source wording that does not duplicate a bridge "
                        "lesson."
                    ),
                    corpus_split=split,
                    source_group_sha256=group,
                    source={"dataset": "other", "revision": "r1", "row_id": identity},
                    source_content_sha256s=("4" * 64,),
                    source_custody_sha256=custody,
                )
            ],
            root,
            component=component,
            logical_shards=1,
            shard_index=0,
            source_custody={"final_shard_receipt_sha256": custody},
        )
    output = tmp_path / "aggregate"
    aggregate_scans(
        query,
        tmp_path / "pleias",
        tmp_path / "books",
        output,
        pleias_logical_shards=1,
        book_logical_shards=1,
    )
    pair = json.loads((output / "pair_decisions.jsonl").read_text())
    assert pair["resolved_split"] is None
    assert pair["decision"] == "exclude_anchor_foundation_split_conflict"
    document_rows = [
        json.loads(line)
        for line in (output / "document_decisions.jsonl").read_text().splitlines()
    ]
    assert {row["decision"] for row in document_rows} == {
        "exclude_pair_anchor_foundation_split_conflict"
    }
