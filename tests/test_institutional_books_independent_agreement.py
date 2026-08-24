from __future__ import annotations

from sai.data.institutional_books_independent_agreement import (
    agreement_disposition,
    consensus_curriculum_metadata,
)
from tests.test_institutional_books_semantic_decision import _judgment


def test_two_strict_matching_model_families_form_consensus() -> None:
    assert agreement_disposition(_judgment(), _judgment()) == (
        "consensus_candidate",
        [],
    )


def test_independent_quality_regression_holds_candidate() -> None:
    original = _judgment()
    independent = _judgment()
    independent["quality"]["overall_quality"] = 3
    disposition, reasons = agreement_disposition(original, independent)
    assert disposition == "agreement_hold"
    assert "independent:overall_quality_below_floor" in reasons
    assert "independent_does_not_satisfy_policy" in reasons


def test_taxonomy_disagreement_holds_candidate() -> None:
    original = _judgment()
    original["domains"] = ["natural_sciences"]
    independent = _judgment()
    independent["domains"] = ["engineering"]
    independent["genre"] = "engineering"
    disposition, reasons = agreement_disposition(original, independent)
    assert disposition == "agreement_hold"
    assert reasons == ["domain_disagreement", "genre_disagreement"]


def test_consensus_curriculum_keeps_ranges_and_shared_graph_without_quotes() -> None:
    original = {
        "quality": {"overall_quality": 4, "knowledge_density": 4},
        "complexity": {
            "linguistic_complexity": 2,
            "conceptual_complexity": 3,
            "reasoning_complexity": 2,
        },
        "concept_edges": [
            {
                "prerequisite": "algebra",
                "dependent": "calculus",
                "relation": "strict_prerequisite",
                "confidence_ppm": 950_000,
                "evidence_quote": "first exact source quote",
            }
        ],
        "work_id_candidate": "work-a",
        "edition_id_candidate": "edition-a",
        "subdomains": ["analysis", "history"],
        "style": "instructional",
        "curriculum_band": "intermediate",
        "prerequisites": ["algebra"],
        "concepts": ["calculus"],
        "period": ["nineteenth century"],
        "culture_geography": ["india"],
        "recommended_representations": ["worked_examples", "clean_ocr_english"],
        "translation_type": "none_english",
        "confidence_ppm": 950_000,
    }
    independent = {
        **original,
        "quality": {"overall_quality": 4, "knowledge_density": 3},
        "complexity": {
            "linguistic_complexity": 3,
            "conceptual_complexity": 3,
            "reasoning_complexity": 4,
        },
        "concept_edges": [
            {
                **original["concept_edges"][0],
                "confidence_ppm": 900_000,
                "evidence_quote": "second exact source quote",
            }
        ],
        "subdomains": ["analysis", "physics"],
    }
    result = consensus_curriculum_metadata(original, independent)
    assert result["quality_floor"]["knowledge_density"] == 3
    assert result["complexity_range"]["reasoning_complexity"] == {
        "minimum": 2,
        "maximum": 4,
    }
    assert result["shared_subdomains"] == ["analysis"]
    assert result["shared_concept_edges"][0]["confidence_floor_ppm"] == 900_000
    assert "exact source quote" not in str(result)
    assert len(result["metadata_sha256"]) == 64
