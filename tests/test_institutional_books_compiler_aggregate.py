from __future__ import annotations

import json

import pytest

from sai.data.institutional_books_compiler_aggregate import (
    INDEPENDENT_POPULATION_SCHEMA,
    InstitutionalBooksAggregateError,
    _validate_population,
    build_aggregate,
    triage_route,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _judgment() -> dict:
    return {
        "verdict": "retain",
        "current_language": "english",
        "quality": {
            "overall_quality": 4,
            "ocr_quality": 4,
            "knowledge_density": 4,
            "literary_value": 1,
            "historical_value": 3,
        },
        "risks": {
            "rights_evidence_incomplete": False,
            "duplicate_or_near_duplicate_edition": False,
            "ocr_damage": False,
            "factual_unreliability": False,
            "outdated_or_harmful_claims": False,
        },
    }


def test_high_quality_book_routes_to_representation_verification() -> None:
    assert triage_route(_judgment()) == "representation_verification"


def test_historical_risk_routes_before_quality_score() -> None:
    judgment = _judgment()
    judgment["risks"]["outdated_or_harmful_claims"] = True
    assert triage_route(judgment) == "historical_context_transformation"


def test_non_english_routes_to_translation_after_ocr_and_rights() -> None:
    judgment = _judgment()
    judgment["current_language"] = "russian"
    assert triage_route(judgment) == "translation_review"
    judgment["risks"]["rights_evidence_incomplete"] = True
    assert triage_route(judgment) == "rights_hold"


def test_missing_quality_geometry_fails_closed() -> None:
    with pytest.raises(InstitutionalBooksAggregateError, match="route"):
        triage_route({"risks": {}})


def test_aggregate_rejects_invalid_logical_shard_geometry(tmp_path) -> None:
    with pytest.raises(InstitutionalBooksAggregateError, match="logical shards differ"):
        build_aggregate(
            tmp_path / "population",
            tmp_path / "judgments",
            tmp_path / "aggregate.json",
            logical_shards=0,
        )


def test_zero_survivor_independent_population_is_valid_negative_result(
    tmp_path,
) -> None:
    root = tmp_path / "population"
    root.mkdir()
    candidates = root / "candidates.jsonl"
    candidates.write_bytes(b"")
    receipt = {
        "schema": INDEPENDENT_POPULATION_SCHEMA,
        "status": (
            "complete_nontraining_private_independent_book_candidate_population"
        ),
        "output": {
            "path": candidates.name,
            "rows": 0,
            "bytes": 0,
            "sha256": sha256_file(candidates),
        },
        "source_text_private": True,
        "source_text_publishable": False,
        "independent_verification_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (root / "receipt.json").write_text(json.dumps(receipt))
    rows, replay = _validate_population(root)
    assert rows == []
    assert replay["output"]["rows"] == 0
