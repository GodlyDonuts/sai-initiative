"""Compare two complete independent Sai syllabus subject reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.curriculum import PHASES
from sai.data.foundational_syllabus import _prepare as _prepare_syllabus
from sai.data.foundational_syllabus_audit import _prepare as _prepare_audit
from sai.data.foundational_syllabus_review_workspace import (
    REVIEW_ROW_SCHEMA,
    _review_rows,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-foundational-syllabus-review-comparison-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TOP_KEYS = {
    "schema",
    "reviewer_id",
    "concept_id",
    "concept_verdict",
    "proposed_name",
    "proposed_earliest_phase",
    "granularity",
    "edge_reviews",
    "missing_prerequisites",
    "rationale",
}
_EDGE_KEYS = {"prerequisite_id", "classification", "rationale"}


class FoundationalSyllabusReviewCompareError(RuntimeError):
    """The review identity, population, decisions, or comparison differs."""


def _read_review(path: Path, *, expected_sha256: str) -> tuple[list[Any], bytes]:
    if not _HEX64.fullmatch(expected_sha256) or sha256_file(path) != expected_sha256:
        raise FoundationalSyllabusReviewCompareError("review file hash differs")
    encoded = _read_regular_bytes(path, maximum_bytes=16 << 20)
    try:
        rows = [json.loads(line) for line in encoded.decode().splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FoundationalSyllabusReviewCompareError("review JSONL differs") from error
    if not rows:
        raise FoundationalSyllabusReviewCompareError("review JSONL differs")
    return rows, encoded


def _validate_review(
    path: Path,
    *,
    expected_sha256: str,
    source_rows: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], bytes]:
    raw_rows, encoded = _read_review(path, expected_sha256=expected_sha256)
    if len(raw_rows) != len(source_rows):
        raise FoundationalSyllabusReviewCompareError("review population differs")
    concept_ids = {row["concept_id"] for row in source_rows}
    reviewer_ids: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    for raw, source in zip(raw_rows, source_rows, strict=True):
        if (
            not isinstance(raw, dict)
            or set(raw) != _TOP_KEYS
            or raw["schema"] != REVIEW_ROW_SCHEMA
            or not isinstance(raw["reviewer_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", raw["reviewer_id"])
            or raw["concept_id"] != source["concept_id"]
            or raw["concept_verdict"] not in {"accept", "revise", "reject"}
            or raw["granularity"] not in {"appropriate", "too_broad", "too_narrow"}
            or raw["proposed_earliest_phase"] not in PHASES
            or (
                raw["proposed_name"] is not None
                and (
                    not isinstance(raw["proposed_name"], str)
                    or len(raw["proposed_name"].strip()) < 3
                )
            )
            or not isinstance(raw["rationale"], str)
            or len(raw["rationale"].strip()) < 40
            or not isinstance(raw["edge_reviews"], list)
            or not isinstance(raw["missing_prerequisites"], list)
        ):
            raise FoundationalSyllabusReviewCompareError("review row differs")
        reviewer_ids.add(raw["reviewer_id"])
        expected_edges = sorted(row["concept_id"] for row in source["prerequisites"])
        observed_edges: list[str] = []
        edge_reviews = []
        for edge in raw["edge_reviews"]:
            if (
                not isinstance(edge, dict)
                or set(edge) != _EDGE_KEYS
                or edge["classification"] not in {"hard", "supporting", "remove"}
                or not isinstance(edge["rationale"], str)
                or len(edge["rationale"].strip()) < 20
            ):
                raise FoundationalSyllabusReviewCompareError("edge review differs")
            observed_edges.append(edge["prerequisite_id"])
            edge_reviews.append(dict(edge))
        if sorted(observed_edges) != expected_edges or len(set(observed_edges)) != len(
            observed_edges
        ):
            raise FoundationalSyllabusReviewCompareError(
                "edge review population differs"
            )
        edge_reviews.sort(key=lambda row: row["prerequisite_id"])
        missing = []
        seen_missing: set[str] = set()
        for edge in raw["missing_prerequisites"]:
            if (
                not isinstance(edge, dict)
                or set(edge) != _EDGE_KEYS
                or edge["classification"] not in {"hard", "supporting"}
                or edge["prerequisite_id"] not in concept_ids
                or edge["prerequisite_id"] == source["concept_id"]
                or edge["prerequisite_id"] in expected_edges
                or edge["prerequisite_id"] in seen_missing
                or not isinstance(edge["rationale"], str)
                or len(edge["rationale"].strip()) < 20
            ):
                raise FoundationalSyllabusReviewCompareError(
                    "missing prerequisite review differs"
                )
            seen_missing.add(edge["prerequisite_id"])
            missing.append(dict(edge))
        missing.sort(key=lambda row: row["prerequisite_id"])
        if raw["concept_verdict"] == "accept" and (
            raw["proposed_name"] is not None
            or raw["proposed_earliest_phase"] != source["earliest_phase"]
            or raw["granularity"] != "appropriate"
        ):
            raise FoundationalSyllabusReviewCompareError("accepted concept row differs")
        if raw["concept_verdict"] == "revise" and (
            raw["proposed_name"] is None
            and raw["proposed_earliest_phase"] == source["earliest_phase"]
            and raw["granularity"] == "appropriate"
            and all(edge["classification"] == "hard" for edge in edge_reviews)
            and not missing
        ):
            raise FoundationalSyllabusReviewCompareError("revised concept row differs")
        normalized_rows.append(
            {
                **raw,
                "edge_reviews": edge_reviews,
                "missing_prerequisites": missing,
            }
        )
    if len(reviewer_ids) != 1:
        raise FoundationalSyllabusReviewCompareError("reviewer identity differs")
    return reviewer_ids.pop(), normalized_rows, encoded


def _structured(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "concept_verdict": row["concept_verdict"],
        "proposed_name": row["proposed_name"],
        "proposed_earliest_phase": row["proposed_earliest_phase"],
        "granularity": row["granularity"],
        "edge_classifications": {
            edge["prerequisite_id"]: edge["classification"]
            for edge in row["edge_reviews"]
        },
        "missing_prerequisites": {
            edge["prerequisite_id"]: edge["classification"]
            for edge in row["missing_prerequisites"]
        },
    }


def _prepare_comparison(
    *,
    base_concepts: Path,
    additions: Path,
    review_a: Path,
    expected_review_a_sha256: str,
    review_b: Path,
    expected_review_b_sha256: str,
) -> dict[str, Any]:
    try:
        composition_receipt, concept_encoded, _ = _prepare_syllabus(
            base_concepts=base_concepts, additions=additions
        )
        audit_payload, _ = _prepare_audit(
            base_concepts=base_concepts, additions=additions
        )
        source_rows = _review_rows(json.loads(concept_encoded), audit_payload)
    except Exception as error:
        raise FoundationalSyllabusReviewCompareError(
            "syllabus comparison inputs differ"
        ) from error
    reviewer_a, rows_a, encoded_a = _validate_review(
        review_a,
        expected_sha256=expected_review_a_sha256,
        source_rows=source_rows,
    )
    reviewer_b, rows_b, encoded_b = _validate_review(
        review_b,
        expected_sha256=expected_review_b_sha256,
        source_rows=source_rows,
    )
    if reviewer_a == reviewer_b:
        raise FoundationalSyllabusReviewCompareError(
            "independent reviewer identities differ"
        )
    comparisons = []
    structured_agreements = 0
    concept_verdict_agreements = 0
    edge_agreements = 0
    edge_total = 0
    for source, left, right in zip(source_rows, rows_a, rows_b, strict=True):
        left_structured = _structured(left)
        right_structured = _structured(right)
        if left["concept_verdict"] == right["concept_verdict"]:
            concept_verdict_agreements += 1
        left_edges = left_structured["edge_classifications"]
        right_edges = right_structured["edge_classifications"]
        edge_total += len(left_edges)
        edge_agreements += sum(
            left_edges[identity] == right_edges[identity] for identity in left_edges
        )
        agreed = left_structured == right_structured
        structured_agreements += int(agreed)
        comparisons.append(
            {
                "concept_id": source["concept_id"],
                "structured_agreement": agreed,
                "consensus": left_structured if agreed else None,
                "review_a": {
                    "structured": left_structured,
                    "rationale": left["rationale"],
                    "edge_rationales": {
                        edge["prerequisite_id"]: edge["rationale"]
                        for edge in left["edge_reviews"]
                    },
                    "missing_prerequisite_rationales": {
                        edge["prerequisite_id"]: edge["rationale"]
                        for edge in left["missing_prerequisites"]
                    },
                },
                "review_b": {
                    "structured": right_structured,
                    "rationale": right["rationale"],
                    "edge_rationales": {
                        edge["prerequisite_id"]: edge["rationale"]
                        for edge in right["edge_reviews"]
                    },
                    "missing_prerequisite_rationales": {
                        edge["prerequisite_id"]: edge["rationale"]
                        for edge in right["missing_prerequisites"]
                    },
                },
            }
        )
    concepts = len(source_rows)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "complete_consensus_requires_final_syllabus_application"
            if structured_agreements == concepts
            else "disagreements_require_independent_adjudication"
        ),
        "composition_receipt_sha256": composition_receipt["receipt_sha256"],
        "concept_sha256": hashlib.sha256(concept_encoded).hexdigest(),
        "reviews": [
            {
                "reviewer_id": reviewer_a,
                "file_sha256": hashlib.sha256(encoded_a).hexdigest(),
                "rows": len(rows_a),
            },
            {
                "reviewer_id": reviewer_b,
                "file_sha256": hashlib.sha256(encoded_b).hexdigest(),
                "rows": len(rows_b),
            },
        ],
        "summary": {
            "concepts": concepts,
            "structured_concept_agreements": structured_agreements,
            "structured_concept_agreement_ppm": structured_agreements
            * 1_000_000
            // concepts,
            "concept_verdict_agreements": concept_verdict_agreements,
            "concept_verdict_agreement_ppm": concept_verdict_agreements
            * 1_000_000
            // concepts,
            "existing_edges": edge_total,
            "edge_classification_agreements": edge_agreements,
            "edge_classification_agreement_ppm": (
                1_000_000
                if edge_total == 0
                else edge_agreements * 1_000_000 // edge_total
            ),
            "unresolved_concepts": concepts - structured_agreements,
        },
        "concept_comparisons": comparisons,
        "comparison_completed": True,
        "subject_review_qualified": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def compare(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    payload = _prepare_comparison(**kwargs)
    try:
        _write_create_only(
            output, json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
        )
    except Exception as error:
        raise FoundationalSyllabusReviewCompareError(
            "review comparison output differs"
        ) from error
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-concepts", type=Path, required=True)
    parser.add_argument("--additions", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--expected-review-a-sha256", required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--expected-review-b-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    payload = compare(**vars(parser.parse_args(argv)))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "unresolved_concepts": payload["summary"]["unresolved_concepts"],
                "edge_agreement_ppm": payload["summary"][
                    "edge_classification_agreement_ppm"
                ],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
