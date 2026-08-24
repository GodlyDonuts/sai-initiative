from __future__ import annotations

import json

import pytest

from sai.data.institutional_books_independent_population import (
    InstitutionalBooksIndependentPopulationError,
    _decisions,
)
from sai.data.institutional_books_semantic_decision import (
    RECORD_SCHEMA,
    SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _root(tmp_path):
    root = tmp_path / "decisions"
    root.mkdir()
    row = {
        "schema": RECORD_SCHEMA,
        "candidate_identity_sha256": "a" * 64,
        "training_ready": False,
        "disposition": "independent_verification",
    }
    row["record_sha256"] = canonical_sha256(row)
    manifest = root / "decisions.jsonl"
    manifest.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    receipt = {
        "schema": SCHEMA,
        "status": "complete_nontraining_conservative_book_semantic_decision",
        "manifest": {
            "path": manifest.name,
            "rows": 1,
            "bytes": manifest.stat().st_size,
            "sha256": sha256_file(manifest),
            "ordered_records_sha256": canonical_sha256([row["record_sha256"]]),
        },
        "independent_verification_complete": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (root / "receipt.json").write_text(json.dumps(receipt))
    return root


def test_decision_population_replays_exact_record_hashes(tmp_path) -> None:
    rows, receipt = _decisions(_root(tmp_path))
    assert len(rows) == 1
    assert receipt["manifest"]["rows"] == 1


def test_decision_population_rejects_tampered_manifest(tmp_path) -> None:
    root = _root(tmp_path)
    path = root / "decisions.jsonl"
    path.write_text(
        path.read_text().replace("independent_verification", "quality_hold")
    )
    with pytest.raises(
        InstitutionalBooksIndependentPopulationError,
        match="manifest differs",
    ):
        _decisions(root)
