from __future__ import annotations

from sai.data.institutional_books_independent_agreement import (
    agreement_disposition,
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
