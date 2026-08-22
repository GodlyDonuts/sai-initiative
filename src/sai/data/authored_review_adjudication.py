"""Adjudicate two independent semantic reviews of every authored chapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sai.data.annotation_policy import AnnotationPolicyError, validate_policy
from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.authored_review_packet import (
    AuthoredReviewPacketError,
    validate_packet,
)
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-authored-curriculum-semantic-review-receipt-v1"
ROW_SCHEMA = "sai-authored-curriculum-completed-review-row-v1"
CONCEPT_LIST_SCHEMA = "sai-semantic-prerequisite-concept-list-v1"
MAXIMUM_DISAGREEMENT_PPM = 50_000
MAXIMUM_QUALITY_DELTA_PPM = 100_000
RECOMMENDATIONS = {"admit", "exclude", "revise"}
DEFECT_CATEGORIES = {"none", "extraction", "factual", "pedagogical", "licensing"}
_ROW_KEYS = {
    "schema",
    "review_identity_sha256",
    "instructional_quality_ppm",
    "assumed_prior_concepts",
    "taught_concepts",
    "defects",
    "admission_recommendation",
}
_CONCEPT_KEYS = {"concept_id", "confidence_ppm", "evidence_spans"}
_SPAN_KEYS = {"start", "end", "text_sha256"}
_DEFECT_KEYS = {"category", "start", "end", "text_sha256"}


class AuthoredReviewAdjudicationError(RuntimeError):
    """The review evidence, independence, or measured agreement differs."""


def _ppm(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 1_000_000
    ):
        raise AuthoredReviewAdjudicationError(f"{label} differs")
    return value


def _load_json(
    path: Path, label: str, maximum_bytes: int = 1 << 20
) -> tuple[Any, bytes]:
    try:
        encoded = _read_regular_bytes(path, maximum_bytes=maximum_bytes)
        return json.loads(encoded), encoded
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise AuthoredReviewAdjudicationError(f"{label} differs") from error


def _load_jsonl(path: Path, label: str) -> tuple[list[Any], bytes]:
    try:
        encoded = _read_regular_bytes(path, maximum_bytes=1 << 30)
        rows = [json.loads(line) for line in encoded.decode().splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise AuthoredReviewAdjudicationError(f"{label} differs") from error
    if not rows:
        raise AuthoredReviewAdjudicationError(f"{label} differs")
    return rows, encoded


def _concepts(path: Path) -> tuple[set[str], bytes]:
    payload, encoded = _load_json(path, "concept list", 8 << 20)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "status", "concepts"}
        or payload["schema"] != CONCEPT_LIST_SCHEMA
        or payload["status"] != "candidate"
        or not isinstance(payload["concepts"], list)
        or not payload["concepts"]
    ):
        raise AuthoredReviewAdjudicationError("concept list differs")
    identities = [concept.get("concept_id") for concept in payload["concepts"]]
    if any(not isinstance(value, str) or not value for value in identities) or len(
        identities
    ) != len(set(identities)):
        raise AuthoredReviewAdjudicationError("concept list differs")
    return set(identities), encoded


def _validate_spans(spans: Any, text: str, label: str) -> None:
    if not isinstance(spans, list) or not spans:
        raise AuthoredReviewAdjudicationError(f"{label} spans differ")
    previous_end = -1
    for span in spans:
        if not isinstance(span, dict) or set(span) != _SPAN_KEYS:
            raise AuthoredReviewAdjudicationError(f"{label} span differs")
        start, end = span["start"], span["end"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or start >= end
            or end - start < 16
            or end > len(text)
            or start < previous_end
            or span["text_sha256"]
            != hashlib.sha256(text[start:end].encode()).hexdigest()
        ):
            raise AuthoredReviewAdjudicationError(f"{label} span differs")
        previous_end = end


def _validate_reviews(
    rows: list[Any], packet: list[Any], concepts: set[str], label: str
) -> list[dict[str, Any]]:
    if len(rows) != len(packet):
        raise AuthoredReviewAdjudicationError(f"{label} population differs")
    validated = []
    for row, source in zip(rows, packet, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != _ROW_KEYS
            or row["schema"] != ROW_SCHEMA
            or row["review_identity_sha256"] != source["review_identity_sha256"]
            or row["admission_recommendation"] not in RECOMMENDATIONS
        ):
            raise AuthoredReviewAdjudicationError(f"{label} row differs")
        _ppm(row["instructional_quality_ppm"], f"{label} quality")
        assumed = row["assumed_prior_concepts"]
        if (
            not isinstance(assumed, list)
            or assumed != sorted(assumed)
            or len(assumed) != len(set(assumed))
            or any(concept not in concepts for concept in assumed)
        ):
            raise AuthoredReviewAdjudicationError(f"{label} assumptions differ")
        taught = row["taught_concepts"]
        if not isinstance(taught, list):
            raise AuthoredReviewAdjudicationError(f"{label} concepts differ")
        taught_ids = []
        for evidence in taught:
            if not isinstance(evidence, dict) or set(evidence) != _CONCEPT_KEYS:
                raise AuthoredReviewAdjudicationError(f"{label} evidence differs")
            concept_id = evidence["concept_id"]
            if concept_id not in concepts or concept_id in taught_ids:
                raise AuthoredReviewAdjudicationError(f"{label} concepts differ")
            if _ppm(evidence["confidence_ppm"], f"{label} confidence") < 800_000:
                raise AuthoredReviewAdjudicationError(f"{label} confidence differs")
            _validate_spans(evidence["evidence_spans"], source["text"], label)
            taught_ids.append(concept_id)
        if taught_ids != sorted(taught_ids):
            raise AuthoredReviewAdjudicationError(f"{label} concepts differ")
        if set(taught_ids) & set(assumed):
            raise AuthoredReviewAdjudicationError(f"{label} concept roles differ")
        if row["admission_recommendation"] == "admit" and not taught_ids:
            raise AuthoredReviewAdjudicationError(
                f"{label} admitted row contains no taught concept"
            )
        defects = row["defects"]
        if not isinstance(defects, list):
            raise AuthoredReviewAdjudicationError(f"{label} defects differ")
        for defect in defects:
            if (
                not isinstance(defect, dict)
                or set(defect) != _DEFECT_KEYS
                or defect["category"] not in DEFECT_CATEGORIES - {"none"}
            ):
                raise AuthoredReviewAdjudicationError(f"{label} defect differs")
            _validate_spans(
                [
                    {
                        "start": defect["start"],
                        "end": defect["end"],
                        "text_sha256": defect["text_sha256"],
                    }
                ],
                source["text"],
                label,
            )
        validated.append(
            {
                "taught": tuple(taught_ids),
                "assumed": tuple(assumed),
                "quality": row["instructional_quality_ppm"],
                "recommendation": row["admission_recommendation"],
                "defect_categories": tuple(
                    sorted(defect["category"] for defect in defects)
                ),
            }
        )
    return validated


def _ceil_ppm(numerator: int, denominator: int) -> int:
    return (numerator * 1_000_000 + denominator - 1) // denominator


def adjudicate(
    *,
    candidate: Path,
    candidate_receipt: Path,
    review_packet: Path,
    review_key: Path,
    review_packet_receipt: Path,
    concept_list: Path,
    annotation_policy: Path,
    annotator_identity: Path,
    reviewer_identity: Path,
    annotator_reviews: Path,
    reviewer_reviews: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate two complete reviews and publish their unmodified agreement."""

    if output.exists():
        raise AuthoredReviewAdjudicationError("adjudication output exists")
    try:
        packet_receipt = validate_packet(
            candidate=candidate,
            candidate_receipt=candidate_receipt,
            review_output=review_packet,
            key_output=review_key,
            receipt_output=review_packet_receipt,
        )
        packet_rows, packet_encoded = _load_jsonl(review_packet, "review packet")
        concept_ids, concept_encoded = _concepts(concept_list)
        policy_encoded = _read_regular_bytes(annotation_policy, maximum_bytes=1 << 20)
        validate_policy(
            annotation_policy,
            expected_concept_list_sha256=hashlib.sha256(concept_encoded).hexdigest(),
        )
    except (AuthoredReviewPacketError, AnnotationPolicyError, OSError) as error:
        raise AuthoredReviewAdjudicationError(
            "review parent evidence differs"
        ) from error
    identity_a = _read_regular_bytes(annotator_identity, maximum_bytes=1 << 20)
    identity_b = _read_regular_bytes(reviewer_identity, maximum_bytes=1 << 20)
    identity_hashes = [
        hashlib.sha256(value).hexdigest() for value in (identity_a, identity_b)
    ]
    if identity_hashes[0] == identity_hashes[1]:
        raise AuthoredReviewAdjudicationError("review identities are not independent")
    annotator_rows, annotator_encoded = _load_jsonl(
        annotator_reviews, "annotator reviews"
    )
    reviewer_rows, reviewer_encoded = _load_jsonl(reviewer_reviews, "reviewer reviews")
    annotator = _validate_reviews(annotator_rows, packet_rows, concept_ids, "annotator")
    reviewer = _validate_reviews(reviewer_rows, packet_rows, concept_ids, "reviewer")
    concept_disagreements = []
    assumption_disagreements = []
    quality_disagreements = []
    recommendation_disagreements = []
    defect_disagreements = []
    for source, left, right in zip(packet_rows, annotator, reviewer, strict=True):
        identity = source["review_identity_sha256"]
        if left["taught"] != right["taught"]:
            concept_disagreements.append(identity)
        if left["assumed"] != right["assumed"]:
            assumption_disagreements.append(identity)
        if abs(left["quality"] - right["quality"]) > MAXIMUM_QUALITY_DELTA_PPM:
            quality_disagreements.append(identity)
        if left["recommendation"] != right["recommendation"]:
            recommendation_disagreements.append(identity)
        if left["defect_categories"] != right["defect_categories"]:
            defect_disagreements.append(identity)
    denominator = len(packet_rows)
    observed = {
        "taught_concepts": _ceil_ppm(len(concept_disagreements), denominator),
        "assumed_prerequisites": _ceil_ppm(len(assumption_disagreements), denominator),
        "instructional_quality": _ceil_ppm(len(quality_disagreements), denominator),
        "admission_recommendation": _ceil_ppm(
            len(recommendation_disagreements), denominator
        ),
        "defect_categories": _ceil_ppm(len(defect_disagreements), denominator),
    }
    passed = all(value <= MAXIMUM_DISAGREEMENT_PPM for value in observed.values())
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "audit_qualified": passed,
        "training_authorized": False,
        "four_b_training_authorized": False,
        "review_packet_receipt_sha256": packet_receipt["receipt_sha256"],
        "review_packet_file_sha256": hashlib.sha256(packet_encoded).hexdigest(),
        "concept_list_sha256": hashlib.sha256(concept_encoded).hexdigest(),
        "annotation_policy_sha256": hashlib.sha256(policy_encoded).hexdigest(),
        "annotator_identity_sha256": identity_hashes[0],
        "reviewer_identity_sha256": identity_hashes[1],
        "annotator_reviews_sha256": hashlib.sha256(annotator_encoded).hexdigest(),
        "reviewer_reviews_sha256": hashlib.sha256(reviewer_encoded).hexdigest(),
        "reviewed_documents": denominator,
        "thresholds": {
            "maximum_disagreement_ppm": MAXIMUM_DISAGREEMENT_PPM,
            "maximum_instructional_quality_delta_ppm": MAXIMUM_QUALITY_DELTA_PPM,
            "minimum_concept_confidence_ppm": 800_000,
        },
        "observed_disagreement_ppm": observed,
        "disagreement_identities": {
            "taught_concepts": concept_disagreements,
            "assumed_prerequisites": assumption_disagreements,
            "instructional_quality": quality_disagreements,
            "admission_recommendation": recommendation_disagreements,
            "defect_categories": defect_disagreements,
        },
        "limitations": [
            "agreement_does_not_prove_taxonomy_completeness",
            "agreement_does_not_replace_license_deduplication_or_decontamination",
            "pass_authorizes_only_taxonomy_and_progression_analysis",
            "no_training_or_architecture_promotion_is_authorized",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _write_create_only(
        output, json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate",
        "candidate-receipt",
        "review-packet",
        "review-key",
        "review-packet-receipt",
        "concept-list",
        "annotation-policy",
        "annotator-identity",
        "reviewer-identity",
        "annotator-reviews",
        "reviewer-reviews",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = adjudicate(
        **{key.replace("-", "_"): value for key, value in vars(args).items()}
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "audit_qualified": payload["audit_qualified"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
