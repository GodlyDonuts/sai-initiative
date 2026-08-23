import json
from pathlib import Path

import pytest

from sai.data.hermes_teacher_census import (
    HermesTeacherCensusError,
    build_census,
)
from sai.data.institutional_books_compiler_aggregate import (
    SCHEMA as BOOK_AGGREGATE_SCHEMA,
)
from sai.data.reservoir_audit_aggregate import SCHEMA as RESERVOIR_AGGREGATE_SCHEMA
from sai.data.source_quality_gate_publication import SCHEMA as QUALITY_GATE_SCHEMA
from sai.data.token_stream import canonical_sha256


def _write(path: Path, payload: dict) -> Path:
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def _quality(path: Path) -> Path:
    populations = [
        {"order": 0, "source_sha256": "a" * 64, "source_rows": 2},
        {"order": 1, "source_sha256": "b" * 64, "source_rows": 1},
    ]
    return _write(
        path,
        {
            "schema": QUALITY_GATE_SCHEMA,
            "status": "complete_source_safe_mechanical_quality_gate_publication",
            "populations": populations,
            "population_assignment_rows": 3,
            "unique_candidate_rows": 3,
            "unique_source_content_rows": 3,
            "cross_population_duplicate_identity_rows": 0,
            "cross_population_duplicate_assignments": 0,
            "cross_population_duplicate_content_rows": 0,
            "cross_population_duplicate_content_assignments": 0,
            "training_ready": False,
        },
    )


def _reservoir(path: Path) -> Path:
    counts = {
        "verdict": {"retain": 1, "review": 1},
        "domains": {"mathematics": 1, "computer_science": 1},
        "source_language": {"english": 2},
        "style": {"exposition": 1, "code": 1},
        "risks": {"duplicated_boilerplate": 1},
        "recommended_representations": {"source_anchor": 2},
        "epistemic_functions": {"reality_anchor": 2},
        "likely_origin": {"organic_human": 2},
        "grounding_type": {"primary_source": 2},
        "preservation_policy": {"preserve_source_anchor_only": 2},
        "translation_disposition": {"not_needed_english": 2},
        "curriculum_phase": {"grounding": 1, "breadth": 1},
        "difficulty": {"2": 1, "3": 1},
        "prerequisite_burden": {"1": 2},
    }
    return _write(
        path,
        {
            "schema": RESERVOIR_AGGREGATE_SCHEMA,
            "status": "complete",
            "population_file_sha256": "a" * 64,
            "summary": {
                "rows": 2,
                "counts": counts,
                "conservative_triage_routes": {
                    "representation_verification": 1,
                    "cleanup_review": 1,
                },
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "model_judgments_are_verified_admissions": False,
                "representation_verification_is_training_admission": False,
            },
            "independent_factual_verification_complete": False,
            "cross_source_deduplication_complete": False,
            "benchmark_decontamination_complete": False,
            "training_ready": False,
        },
    )


def _books(path: Path) -> Path:
    return _write(
        path,
        {
            "schema": BOOK_AGGREGATE_SCHEMA,
            "status": "complete_nontraining_book_compiler_aggregate",
            "population": {"candidate_file_sha256": "b" * 64, "rows": 1},
            "counts": {
                "verdict": {"retain": 1},
                "domain": {"literature": 1},
                "current_language": {"english": 1},
                "style": {"narrative": 1},
                "risk": {"rights_evidence_incomplete": 1},
                "triage_route": {"rights_hold": 1},
                "recommended_representation": {"preserved_original": 1},
                "curriculum_band": {"intermediate": 1},
                "genre": {"novel": 1},
                "translation_type": {"human_translation": 1},
                "rights_status": {"public_domain_review": 1},
            },
            "usage": {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10},
            "source_text_persisted": False,
            "evidence_quotes_persisted": False,
            "model_judgments_are_verified_admissions": False,
            "training_ready": False,
        },
    )


def test_census_joins_exact_quality_gate_population(tmp_path) -> None:
    result = build_census(
        _quality(tmp_path / "quality.json"),
        [_books(tmp_path / "books.json"), _reservoir(tmp_path / "reservoir.json")],
        tmp_path / "census.json",
    )
    assert result["population_rows"] == 3
    assert result["population_count"] == 2
    assert [row["source_sha256"] for row in result["populations"]] == [
        "a" * 64,
        "b" * 64,
    ]
    assert result["counts"]["verdict"] == {"retain": 2, "review": 1}
    assert result["counts"]["triage_routes"] == {
        "cleanup_review": 1,
        "representation_verification": 1,
        "rights_hold": 1,
    }
    assert result["usage"] == {
        "completion_tokens": 9,
        "prompt_tokens": 16,
        "total_tokens": 25,
    }
    assert result["publication_contains_source_text"] is False
    assert result["training_ready"] is False


def test_census_rejects_missing_population_aggregate(tmp_path) -> None:
    with pytest.raises(HermesTeacherCensusError, match="coverage differs"):
        build_census(
            _quality(tmp_path / "quality.json"),
            [_reservoir(tmp_path / "reservoir.json")],
            tmp_path / "census.json",
        )


def test_census_rejects_tampered_aggregate(tmp_path) -> None:
    reservoir = _reservoir(tmp_path / "reservoir.json")
    payload = json.loads(reservoir.read_text())
    payload["summary"]["rows"] = 3
    reservoir.write_text(json.dumps(payload))
    with pytest.raises(HermesTeacherCensusError, match="receipt differs"):
        build_census(
            _quality(tmp_path / "quality.json"),
            [reservoir, _books(tmp_path / "books.json")],
            tmp_path / "census.json",
        )
