from __future__ import annotations

import hashlib

import pytest

from sai.data.grounded_representation_verification_population import (
    GroundedRepresentationVerificationPopulationError,
    build_candidate,
)
from sai.data.token_stream import canonical_sha256


def _source() -> dict:
    quote = "A source documents the historical relationship between art and geometry."
    text = quote + (" It preserves context and uncertainty." * 6)
    return {
        "text": text,
        "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_record_sha256": "1" * 64,
        "candidate_identity_sha256": "2" * 64,
        "source": {
            "dataset": "common-pile/public_domain_review_filtered",
            "row_id": "art-geometry",
            "source_url": "https://publicdomainreview.org/collection/art-geometry/",
            "source_type": "collection",
            "license": "CC-BY-SA-4.0",
            "attribution_required": True,
            "share_alike_required": True,
        },
    }


def _generated() -> dict:
    text = (
        "The document presents art and geometry as historically related while "
        "retaining the source's uncertainty."
    )
    row = {
        "schema": "sai-generated-grounded-representation-candidate-v1",
        "source_candidate_identity_sha256": "2" * 64,
        "generator_receipt_sha256": "3" * 64,
        "representation_index": 0,
        "representation_type": "conceptual_summary",
        "title": "Art and geometry",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "concepts": ["art and geometry"],
        "difficulty": 1,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def _clean(generated: dict) -> dict:
    row = {
        **generated,
        "schema": "sai-benchmark-disjoint-grounded-representation-candidate-v1",
        "pre_decontamination_record_sha256": generated["record_sha256"],
        "benchmark_decontamination_complete": True,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def _receipt() -> dict:
    quote = "A source documents the historical relationship between art and geometry."
    return {
        "receipt_sha256": "3" * 64,
        "judgment": {
            "judgment_sha256": "4" * 64,
            "representations": [
                {
                    "type": "conceptual_summary",
                    "title": "Art and geometry",
                    "text": _generated()["text"],
                    "evidence_quotes": [quote],
                    "concepts": ["art and geometry"],
                    "difficulty": 1,
                }
            ],
        },
    }


def test_build_candidate_binds_clean_text_to_literal_source_citations() -> None:
    generated = _generated()
    generated["evidence_quote_sha256s"] = [
        hashlib.sha256(
            b"A source documents the historical relationship between art and geometry."
        ).hexdigest()
    ]
    clean = _clean(generated)
    row = build_candidate(clean, generated, _source(), _receipt())
    assert row["source_evidence_quotes"][0] in row["source_text"]
    assert row["generated_text"] == clean["text"]
    assert row["benchmark_decontamination_complete"] is True
    assert row["independent_request_verification_complete"] is False
    identity = row.pop("candidate_identity_sha256")
    assert identity == canonical_sha256(row)


def test_build_candidate_rejects_generated_text_drift() -> None:
    generated = _generated()
    quote = "A source documents the historical relationship between art and geometry."
    generated["evidence_quote_sha256s"] = [hashlib.sha256(quote.encode()).hexdigest()]
    clean = _clean(generated)
    clean["text"] += " Drift."
    with pytest.raises(
        GroundedRepresentationVerificationPopulationError, match="binding"
    ):
        build_candidate(clean, generated, _source(), _receipt())
