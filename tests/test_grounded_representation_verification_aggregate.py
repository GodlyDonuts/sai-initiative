from __future__ import annotations

from sai.data.grounded_representation_verification_aggregate import (
    REJECTION_SCHEMA,
    RETAINED_SCHEMA,
    REVISION_SCHEMA,
    route_candidate,
)


def _candidate() -> dict:
    return {
        "candidate_identity_sha256": "1" * 64,
        "source_candidate_identity_sha256": "2" * 64,
        "generated_record_sha256": "3" * 64,
        "clean_record_sha256": "4" * 64,
        "generator_receipt_sha256": "5" * 64,
        "representation_index": 0,
        "representation_type": "conceptual_summary",
        "source": {
            "dataset": "common-pile/public_domain_review_filtered",
            "row_id": "historical-design",
            "source_url": "https://publicdomainreview.org/collection/design/",
            "source_type": "collection",
            "license": "CC-BY-SA-4.0",
            "attribution_required": True,
            "share_alike_required": True,
        },
        "title": "Historical design",
        "generated_text": "A faithful, source-grounded account of historical design.",
        "generated_text_sha256": "6" * 64,
        "concepts": ["historical design"],
        "difficulty": 1,
    }


def _receipt(verdict: str) -> dict:
    defects = [] if verdict == "retain" else ["not_entailed"]
    return {
        "receipt_sha256": "7" * 64,
        "judgment": {
            "judgment_sha256": "8" * 64,
            "verdict": verdict,
            "scores": {
                "source_entailment": 4 if verdict == "retain" else 2,
                "factual_fidelity": 4 if verdict == "retain" else 2,
                "pedagogical_value": 4,
                "linguistic_quality": 4,
                "cultural_fidelity": 3,
                "uncertainty_fidelity": 3,
            },
            "defects": defects,
            "revision_brief": (
                "Remove the unsupported claim." if verdict != "retain" else ""
            ),
            "source_evidence_quotes": ["exact source evidence"],
            "representation_evidence_quotes": ["source-grounded account"],
            "rationale": "The decision follows the source comparison.",
        },
    }


def test_retain_route_remains_nontraining_and_same_family_only() -> None:
    route, row = route_candidate(_candidate(), _receipt("retain"))
    assert route == "retain"
    assert row["schema"] == RETAINED_SCHEMA
    assert row["same_family_retention_passed"] is True
    assert row["independent_model_family_verification_complete"] is False
    assert row["representation_verified"] is False
    assert row["training_ready"] is False


def test_revision_route_keeps_work_text_and_hashes_evidence() -> None:
    route, row = route_candidate(_candidate(), _receipt("revise"))
    assert route == "revise"
    assert row["schema"] == REVISION_SCHEMA
    assert row["revision_complete"] is False
    assert row["source_evidence_quote_sha256s"]
    assert row["text"] == _candidate()["generated_text"]


def test_rejection_route_does_not_persist_generated_text() -> None:
    route, row = route_candidate(_candidate(), _receipt("reject"))
    assert route == "reject"
    assert row["schema"] == REJECTION_SCHEMA
    assert row["generated_text_persisted"] is False
    assert "text" not in row
