from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.authored_review_human_attestation import (
    AuthoredReviewHumanAttestationError,
    run,
    validate,
)

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts" / "authored-curriculum-sources-r1"


def _inputs(tmp_path: Path) -> dict:
    packet = ARTIFACT / "authored-curriculum-blind-review.jsonl"
    packet_receipt = ARTIFACT / "authored-curriculum-review-receipt.json"
    packet_rows = [json.loads(line) for line in packet.read_text().splitlines()]
    reviews = []
    for source in packet_rows:
        quote = next(
            source["text"][start : start + 16]
            for start in range(len(source["text"]) - 15)
            if source["text"].count(source["text"][start : start + 16]) == 1
        )
        start = source["text"].index(quote)
        reviews.append(
            {
                "schema": "sai-authored-curriculum-completed-review-row-v1",
                "review_identity_sha256": source["review_identity_sha256"],
                "instructional_quality_ppm": 900_000,
                "assumed_prior_concepts": [],
                "taught_concepts": [
                    {
                        "concept_id": "code.literal",
                        "confidence_ppm": 900_000,
                        "evidence_spans": [
                            {
                                "start": start,
                                "end": start + len(quote),
                                "text_sha256": hashlib.sha256(
                                    quote.encode()
                                ).hexdigest(),
                            }
                        ],
                    }
                ],
                "defects": [],
                "admission_recommendation": "admit",
            }
        )
    completed = tmp_path / "reviews.jsonl"
    completed.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reviews)
    )
    return {
        "role": "annotator",
        "identity_attestation_sha256": "1" * 64,
        "review_packet": packet,
        "review_packet_receipt": packet_receipt,
        "expected_review_packet_sha256": hashlib.sha256(
            packet.read_bytes()
        ).hexdigest(),
        "expected_review_packet_receipt_sha256": hashlib.sha256(
            packet_receipt.read_bytes()
        ).hexdigest(),
        "concept_list": ROOT
        / "docs"
        / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json",
        "annotation_policy": ROOT / "docs" / "SAI_SEMANTIC_ANNOTATION_POLICY.json",
        "completed_reviews": completed,
        "output": tmp_path / "attestation.json",
    }


def test_human_attestation_binds_exact_completed_labels_and_replays(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    payload = run(**inputs)
    assert payload["human_review_completed"] is True
    assert payload["model_generated_labels"] is False
    assert payload["reviewed_documents"] == 127
    assert validate(**inputs) == payload
    inputs["completed_reviews"].write_text(
        inputs["completed_reviews"].read_text().replace("900000", "900001", 1)
    )
    with pytest.raises(
        AuthoredReviewHumanAttestationError, match="attestation differs"
    ):
        validate(**inputs)


def test_human_attestation_rejects_placeholder_or_model_style_identity(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    inputs["identity_attestation_sha256"] = "0" * 64
    with pytest.raises(AuthoredReviewHumanAttestationError, match="identity differs"):
        run(**inputs)
