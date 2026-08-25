from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.grounded_bridge_curriculum_candidates import (
    RECEIPT_SCHEMA as CANDIDATE_SCHEMA,
)
from sai.data.grounded_bridge_curriculum_candidates import ROW_SCHEMA
from sai.data.institutional_books_practical_admission import SCHEMA as BOOKS_SCHEMA
from sai.data.pleias_practical_admission import SCHEMA as PLEIAS_SCHEMA
from sai.data.practical_bridge_reconcile import (
    PracticalBridgeReconcileError,
    reconcile,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(payload: dict) -> dict:
    payload = dict(payload)
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _bridge(pair: str, split: str, content: str, anchors: list[str]) -> dict:
    row = {
        "schema": ROW_SCHEMA,
        "pair_identity_sha256": pair,
        "corpus_split": split,
        "content_sha256": content,
        "anchor_source_content_sha256s": anchors,
        "text": f"lesson-{pair[:2]}",
        "text_utf8_bytes": 9,
        "training_ready": False,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict]]:
    book_hash = "b" * 64
    pleias_hash = "c" * 64
    rows = [
        _bridge("1" * 64, "development", "d" * 64, [book_hash, "2" * 64]),
        _bridge("3" * 64, "development", "e" * 64, ["4" * 64, "5" * 64]),
        _bridge("6" * 64, "train", pleias_hash, ["7" * 64, "8" * 64]),
    ]
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    stream = candidates / "curriculum_candidates.jsonl"
    stream.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    candidate_receipt = _signed(
        {
            "schema": CANDIDATE_SCHEMA,
            "status": "complete_nontraining_grounded_bridge_curriculum_candidates",
            "independent_model_family_verification_complete": True,
            "benchmark_decontamination_complete": True,
            "bridge_pair_disjoint_split_complete": True,
            "global_deduplication_against_foundation_complete": False,
            "transfer_ablation_complete": False,
            "training_ready": False,
            "curriculum_candidates": {
                "path": stream.name,
                "rows": len(rows),
                "bytes": stream.stat().st_size,
                "sha256": sha256_file(stream),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in rows]
                ),
            },
        }
    )
    _write_json(candidates / "receipt.json", candidate_receipt)

    books = tmp_path / "books"
    books.mkdir()
    book_manifest = books / "manifest.jsonl"
    book_row = {
        "source_content_sha256": book_hash,
        "record_sha256": "9" * 64,
    }
    book_manifest.write_text(json.dumps(book_row, sort_keys=True) + "\n")
    _write_json(
        books / "receipt.json",
        _signed(
            {
                "schema": BOOKS_SCHEMA,
                "training_ready": True,
                "practical_pretraining_ready": True,
                "manifest": {
                    "path": book_manifest.name,
                    "rows": 1,
                    "bytes": book_manifest.stat().st_size,
                    "sha256": sha256_file(book_manifest),
                    "ordered_records_sha256": canonical_sha256(["9" * 64]),
                },
            }
        ),
    )

    pleias = tmp_path / "pleias"
    shard = pleias / "shards" / "shard_00000"
    shard.mkdir(parents=True)
    locator = shard / "locators.parquet"
    pq.write_table(pa.table({"content_sha256": [pleias_hash, "a" * 64]}), locator)
    descriptor = {
        "shard_index": 0,
        "path": str(locator.relative_to(pleias)),
        "rows": 2,
        "bytes": locator.stat().st_size,
        "sha256": sha256_file(locator),
    }
    _write_json(
        pleias / "receipt.json",
        _signed(
            {
                "schema": PLEIAS_SCHEMA,
                "training_ready": True,
                "global_exact_content_deduplication_complete": True,
                "counts": {"admitted_rows": 2},
                "outputs": {"descriptors": [descriptor]},
            }
        ),
    )
    return candidates, books / "receipt.json", pleias / "receipt.json", rows


def test_reconciles_exact_foundation_overlap_and_preserves_dev(tmp_path: Path) -> None:
    candidates, books, pleias, _rows = _fixture(tmp_path)
    result = reconcile(
        candidates,
        books,
        pleias,
        tmp_path / "output",
        tmp_path / "evidence" / "receipt.json",
    )
    assert result["counts"]["development_pairs_promoted_to_train"] == 1
    assert result["counts"]["train_documents"] == 1
    assert result["counts"]["development_documents"] == 1
    assert result["counts"]["exact_duplicate_documents_held"] == 1
    assert result["development_source_disjoint_against_foundation_complete"] is True
    train = json.loads((tmp_path / "output" / "train.jsonl").read_text())
    development = json.loads((tmp_path / "output" / "development.jsonl").read_text())
    assert train["provisional_corpus_split"] == "development"
    assert train["corpus_split"] == "train"
    assert development["corpus_split"] == "development"
    assert result["training_ready"] is False


def test_rejects_candidate_tampering(tmp_path: Path) -> None:
    candidates, books, pleias, rows = _fixture(tmp_path)
    rows[0]["text"] = "tampered"
    stream = candidates / "curriculum_candidates.jsonl"
    stream.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    receipt = json.loads((candidates / "receipt.json").read_text())
    receipt["curriculum_candidates"]["bytes"] = stream.stat().st_size
    receipt["curriculum_candidates"]["sha256"] = sha256_file(stream)
    receipt.pop("receipt_sha256")
    _write_json(candidates / "receipt.json", _signed(receipt))
    with pytest.raises(PracticalBridgeReconcileError, match="candidate row"):
        reconcile(
            candidates,
            books,
            pleias,
            tmp_path / "output",
            tmp_path / "evidence" / "receipt.json",
        )
