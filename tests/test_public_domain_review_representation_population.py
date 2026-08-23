from __future__ import annotations

import hashlib

import pytest

from sai.data.public_domain_review_representation_population import (
    PublicDomainReviewRepresentationPopulationError,
    build_candidate,
    select_derivative_representations,
)
from sai.data.token_stream import canonical_sha256


def _source() -> dict:
    text = "A museum preserves a documented cultural history. " * 8
    row = {
        "schema": "sai-public-domain-review-scoped-candidate-v1",
        "text": text,
        "scoped_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "record_sha256": "",
        "original_candidate_identity_sha256": "1" * 64,
        "source": {
            "dataset": "common-pile/public_domain_review_filtered",
            "row_id": "museum-history",
            "license": "CC-BY-SA-4.0",
        },
        "source_url": "https://publicdomainreview.org/collection/example/",
        "source_type": "collection",
        "attribution_required": True,
        "share_alike_required": True,
    }
    return row


def _compiler() -> dict:
    judgment = {
        "judgment_sha256": "3" * 64,
        "recommended_representations": [
            "original_english",
            "faq",
            "conceptual_summary",
            "concise_reference",
            "cleaned_source",
        ],
        "verdict": "retain",
        "preservation_policy": "preserve_plus_derivatives",
        "domains": ["history"],
        "subdomains": ["museum history"],
        "concepts_taught": ["cultural preservation"],
        "prerequisites_assumed": ["chronology"],
        "cross_domain_bridges": ["history::visual_arts"],
        "difficulty": 2,
        "curriculum_phase": "breadth",
    }
    return {
        "candidate_identity_sha256": "2" * 64,
        "receipt_sha256": "4" * 64,
        "judgment": judgment,
    }


def _lane() -> dict:
    return {
        "original_candidate_identity_sha256": "1" * 64,
        "representation_priority_candidate": True,
        "compiler_candidate_identity_sha256": "2" * 64,
        "compiler_receipt_sha256": "4" * 64,
        "compiler_judgment_sha256": "3" * 64,
        "work_record_sha256": "5" * 64,
        "content_route": "representation_verification",
        "rights_route": "editorial_scope_review",
    }


def test_selects_derivatives_in_frozen_priority_order() -> None:
    assert select_derivative_representations(
        [
            "faq",
            "original_english",
            "conceptual_summary",
            "cleaned_source",
            "concise_reference",
        ]
    ) == ["conceptual_summary", "concise_reference", "faq"]


def test_build_candidate_binds_clean_text_and_compiler_plan() -> None:
    row = build_candidate(_source(), _lane(), _compiler())
    assert row is not None
    assert row["compiler"]["requested_representations"] == [
        "conceptual_summary",
        "concise_reference",
        "faq",
    ]
    assert row["source"]["share_alike_required"] is True
    assert row["training_ready"] is False
    identity = row.pop("candidate_identity_sha256")
    assert identity == canonical_sha256(row)


def test_build_candidate_skips_preservation_only_plan() -> None:
    compiler = _compiler()
    compiler["judgment"]["recommended_representations"] = ["original_english"]
    assert build_candidate(_source(), _lane(), compiler) is None


def test_build_candidate_rejects_rights_drift() -> None:
    source = _source()
    source["source"]["license"] = "CC-BY-4.0"
    with pytest.raises(PublicDomainReviewRepresentationPopulationError, match="rights"):
        build_candidate(source, _lane(), _compiler())
