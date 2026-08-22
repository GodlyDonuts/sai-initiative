"""Compile exact quoted review evidence into source-bound semantic labels.

Reviewers provide verbatim quotes rather than numeric offsets.  This compiler
resolves only unique occurrences in the immutable blind packet, then emits the
strict completed-review rows consumed by the independent adjudicator.  A
compiled review remains candidate evidence; compilation is not human review,
agreement, data admission, or training authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sai.data.annotation_policy import AnnotationPolicyError, validate_policy
from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.authored_review_adjudication import (
    CONCEPT_LIST_SCHEMA,
    DEFECT_CATEGORIES,
    RECOMMENDATIONS,
    ROW_SCHEMA,
)
from sai.data.authored_review_packet import (
    AuthoredReviewPacketError,
    validate_packet,
)
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-authored-curriculum-compiled-review-receipt-v1"
DRAFT_SCHEMA = "sai-authored-curriculum-quoted-review-draft-row-v1"
_DRAFT_KEYS = {
    "schema",
    "review_identity_sha256",
    "instructional_quality_ppm",
    "assumed_prior_concepts",
    "taught_concepts",
    "defects",
    "admission_recommendation",
}
_TAUGHT_KEYS = {"concept_id", "confidence_ppm", "evidence_quotes"}
_DEFECT_KEYS = {"category", "evidence_quote"}
_CONCEPT_LIST_KEYS = {"schema", "status", "concepts"}


class AuthoredReviewCompileError(RuntimeError):
    """Quoted review evidence cannot be bound exactly to the blind packet."""


def _read_jsonl(path: Path, label: str) -> tuple[list[Any], bytes]:
    try:
        encoded = _read_regular_bytes(path, maximum_bytes=1 << 30)
        rows = [json.loads(line) for line in encoded.decode().splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise AuthoredReviewCompileError(f"{label} differs") from error
    if not rows:
        raise AuthoredReviewCompileError(f"{label} differs")
    return rows, encoded


def _read_json(
    path: Path, label: str, maximum_bytes: int = 8 << 20
) -> tuple[Any, bytes]:
    try:
        encoded = _read_regular_bytes(path, maximum_bytes=maximum_bytes)
        return json.loads(encoded), encoded
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise AuthoredReviewCompileError(f"{label} differs") from error


def _concept_ids(path: Path) -> tuple[set[str], bytes]:
    payload, encoded = _read_json(path, "concept list")
    if (
        not isinstance(payload, dict)
        or set(payload) != _CONCEPT_LIST_KEYS
        or payload["schema"] != CONCEPT_LIST_SCHEMA
        or payload["status"] != "candidate"
        or not isinstance(payload["concepts"], list)
        or not payload["concepts"]
    ):
        raise AuthoredReviewCompileError("concept list differs")
    identities = [item.get("concept_id") for item in payload["concepts"]]
    if any(not isinstance(value, str) or not value for value in identities) or len(
        identities
    ) != len(set(identities)):
        raise AuthoredReviewCompileError("concept list differs")
    return set(identities), encoded


def _ppm(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 1_000_000
    ):
        raise AuthoredReviewCompileError(f"{label} differs")
    return value


def _resolve_quote(text: str, quote: Any, label: str, minimum: int) -> dict[str, Any]:
    if not isinstance(quote, str) or len(quote) < minimum:
        raise AuthoredReviewCompileError(f"{label} quote differs")
    start = text.find(quote)
    if start < 0 or text.find(quote, start + 1) >= 0:
        raise AuthoredReviewCompileError(f"{label} quote is missing or ambiguous")
    end = start + len(quote)
    return {
        "start": start,
        "end": end,
        "text_sha256": hashlib.sha256(quote.encode()).hexdigest(),
    }


def _compile_rows(
    drafts: list[Any],
    packet: list[Any],
    concepts: set[str],
    *,
    minimum_span_codepoints: int,
    minimum_confidence_ppm: int,
) -> list[dict[str, Any]]:
    if len(drafts) != len(packet):
        raise AuthoredReviewCompileError("draft population differs")
    output = []
    for draft, source in zip(drafts, packet, strict=True):
        if (
            not isinstance(draft, dict)
            or set(draft) != _DRAFT_KEYS
            or draft["schema"] != DRAFT_SCHEMA
            or draft["review_identity_sha256"] != source["review_identity_sha256"]
            or draft["admission_recommendation"] not in RECOMMENDATIONS
        ):
            raise AuthoredReviewCompileError("draft row differs")
        quality = _ppm(draft["instructional_quality_ppm"], "quality")
        assumed = draft["assumed_prior_concepts"]
        if (
            not isinstance(assumed, list)
            or assumed != sorted(assumed)
            or len(assumed) != len(set(assumed))
            or any(value not in concepts for value in assumed)
        ):
            raise AuthoredReviewCompileError("assumed concepts differ")
        taught = draft["taught_concepts"]
        if not isinstance(taught, list):
            raise AuthoredReviewCompileError("taught concepts differ")
        compiled_taught = []
        taught_ids = []
        for evidence in taught:
            if not isinstance(evidence, dict) or set(evidence) != _TAUGHT_KEYS:
                raise AuthoredReviewCompileError("taught evidence differs")
            concept_id = evidence["concept_id"]
            confidence = _ppm(evidence["confidence_ppm"], "confidence")
            quotes = evidence["evidence_quotes"]
            if (
                concept_id not in concepts
                or concept_id in taught_ids
                or confidence < minimum_confidence_ppm
                or not isinstance(quotes, list)
                or not quotes
                or len(quotes) != len(set(quotes))
            ):
                raise AuthoredReviewCompileError("taught evidence differs")
            spans = [
                _resolve_quote(
                    source["text"],
                    quote,
                    f"{concept_id} evidence",
                    minimum_span_codepoints,
                )
                for quote in quotes
            ]
            spans.sort(key=lambda value: (value["start"], value["end"]))
            if any(
                right["start"] < left["end"]
                for left, right in zip(spans, spans[1:], strict=False)
            ):
                raise AuthoredReviewCompileError("taught evidence overlaps")
            taught_ids.append(concept_id)
            compiled_taught.append(
                {
                    "concept_id": concept_id,
                    "confidence_ppm": confidence,
                    "evidence_spans": spans,
                }
            )
        if taught_ids != sorted(taught_ids) or set(taught_ids) & set(assumed):
            raise AuthoredReviewCompileError("concept roles differ")
        if draft["admission_recommendation"] == "admit" and not taught_ids:
            raise AuthoredReviewCompileError("admitted row contains no taught concept")
        defects = draft["defects"]
        if not isinstance(defects, list):
            raise AuthoredReviewCompileError("defects differ")
        compiled_defects = []
        for defect in defects:
            if (
                not isinstance(defect, dict)
                or set(defect) != _DEFECT_KEYS
                or defect["category"] not in DEFECT_CATEGORIES - {"none"}
            ):
                raise AuthoredReviewCompileError("defect differs")
            span = _resolve_quote(
                source["text"],
                defect["evidence_quote"],
                "defect",
                minimum_span_codepoints,
            )
            compiled_defects.append({"category": defect["category"], **span})
        compiled_defects.sort(
            key=lambda value: (value["start"], value["end"], value["category"])
        )
        output.append(
            {
                "schema": ROW_SCHEMA,
                "review_identity_sha256": source["review_identity_sha256"],
                "instructional_quality_ppm": quality,
                "assumed_prior_concepts": assumed,
                "taught_concepts": compiled_taught,
                "defects": compiled_defects,
                "admission_recommendation": draft["admission_recommendation"],
            }
        )
    return output


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
        for row in rows
    )


def _prepare(
    *,
    candidate: Path,
    candidate_receipt: Path,
    review_packet: Path,
    review_key: Path,
    review_packet_receipt: Path,
    concept_list: Path,
    annotation_policy: Path,
    reviewer_identity: Path,
    draft: Path,
) -> tuple[dict[str, Any], bytes, bytes, bytes, bytes, bytes, bytes]:
    try:
        packet_receipt = validate_packet(
            candidate=candidate,
            candidate_receipt=candidate_receipt,
            review_output=review_packet,
            key_output=review_key,
            receipt_output=review_packet_receipt,
        )
        packet, packet_encoded = _read_jsonl(review_packet, "review packet")
        concepts, concept_encoded = _concept_ids(concept_list)
        policy_encoded = _read_regular_bytes(annotation_policy, maximum_bytes=1 << 20)
        policy = validate_policy(
            annotation_policy,
            expected_concept_list_sha256=hashlib.sha256(concept_encoded).hexdigest(),
        )
        identity_encoded = _read_regular_bytes(reviewer_identity, maximum_bytes=1 << 20)
        drafts, draft_encoded = _read_jsonl(draft, "quoted review draft")
    except (AuthoredReviewPacketError, AnnotationPolicyError, OSError) as error:
        raise AuthoredReviewCompileError("compiled review parent differs") from error
    rows = _compile_rows(
        drafts,
        packet,
        concepts,
        minimum_span_codepoints=policy["evidence_span_contract"][
            "minimum_codepoints_per_positive_label"
        ],
        minimum_confidence_ppm=policy["confidence_contract"]["minimum_confidence_ppm"],
    )
    output_encoded = _jsonl_bytes(rows)
    return (
        packet_receipt,
        packet_encoded,
        concept_encoded,
        policy_encoded,
        identity_encoded,
        draft_encoded,
        output_encoded,
    )


def _receipt_payload(
    *,
    output: Path,
    packet_receipt: dict[str, Any],
    packet_encoded: bytes,
    concept_encoded: bytes,
    policy_encoded: bytes,
    identity_encoded: bytes,
    draft_encoded: bytes,
    output_encoded: bytes,
) -> dict[str, Any]:
    policy = json.loads(policy_encoded)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "compiled_candidate_review",
        "audit_qualified": False,
        "human_review_completed": False,
        "independent_review_completed": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
        "review_packet_receipt_sha256": packet_receipt["receipt_sha256"],
        "review_packet_file_sha256": hashlib.sha256(packet_encoded).hexdigest(),
        "concept_list_sha256": hashlib.sha256(concept_encoded).hexdigest(),
        "annotation_policy_sha256": hashlib.sha256(policy_encoded).hexdigest(),
        "reviewer_identity_sha256": hashlib.sha256(identity_encoded).hexdigest(),
        "draft_sha256": hashlib.sha256(draft_encoded).hexdigest(),
        "compiled_reviews": {
            "path": str(output.resolve()),
            "rows": len(output_encoded.splitlines()),
            "bytes": len(output_encoded),
            "sha256": hashlib.sha256(output_encoded).hexdigest(),
        },
        "evidence_contract": {
            "input": "unique_exact_verbatim_quotes",
            "coordinate_system": "unicode_codepoint_half_open",
            "minimum_codepoints": policy["evidence_span_contract"][
                "minimum_codepoints_per_positive_label"
            ],
            "ambiguous_quotes_rejected": True,
            "missing_quotes_rejected": True,
            "overlapping_quotes_within_one_concept_rejected": True,
        },
        "limitations": [
            "compiled_labels_are_candidate_evidence_only",
            "compilation_does_not_establish_reviewer_independence",
            "compilation_does_not_replace_blind_independent_review",
            "no_data_admission_training_or_architecture_promotion_is_authorized",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def compile_review(
    *,
    candidate: Path,
    candidate_receipt: Path,
    review_packet: Path,
    review_key: Path,
    review_packet_receipt: Path,
    concept_list: Path,
    annotation_policy: Path,
    reviewer_identity: Path,
    draft: Path,
    output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Compile one complete quoted review population without qualifying it."""

    if (
        output.resolve() == receipt_output.resolve()
        or output.exists()
        or receipt_output.exists()
    ):
        raise AuthoredReviewCompileError("compiled output boundary differs")
    prepared = _prepare(
        candidate=candidate,
        candidate_receipt=candidate_receipt,
        review_packet=review_packet,
        review_key=review_key,
        review_packet_receipt=review_packet_receipt,
        concept_list=concept_list,
        annotation_policy=annotation_policy,
        reviewer_identity=reviewer_identity,
        draft=draft,
    )
    output_encoded = prepared[-1]
    payload = _receipt_payload(
        output=output,
        packet_receipt=prepared[0],
        packet_encoded=prepared[1],
        concept_encoded=prepared[2],
        policy_encoded=prepared[3],
        identity_encoded=prepared[4],
        draft_encoded=prepared[5],
        output_encoded=output_encoded,
    )
    receipt_encoded = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    created: list[Path] = []
    try:
        _write_create_only(output, output_encoded)
        created.append(output)
        _write_create_only(receipt_output, receipt_encoded)
        created.append(receipt_output)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return payload


def validate_compiled_review(
    *,
    candidate: Path,
    candidate_receipt: Path,
    review_packet: Path,
    review_key: Path,
    review_packet_receipt: Path,
    concept_list: Path,
    annotation_policy: Path,
    reviewer_identity: Path,
    draft: Path,
    output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Replay every quote, span, label, and receipt field from immutable inputs."""

    try:
        observed_output = _read_regular_bytes(output, maximum_bytes=1 << 30)
        observed_receipt = _read_regular_bytes(receipt_output, maximum_bytes=1 << 20)
        payload = json.loads(observed_receipt)
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise AuthoredReviewCompileError("compiled review is unreadable") from error
    prepared = _prepare(
        candidate=candidate,
        candidate_receipt=candidate_receipt,
        review_packet=review_packet,
        review_key=review_key,
        review_packet_receipt=review_packet_receipt,
        concept_list=concept_list,
        annotation_policy=annotation_policy,
        reviewer_identity=reviewer_identity,
        draft=draft,
    )
    expected_output = prepared[-1]
    expected = _receipt_payload(
        output=output,
        packet_receipt=prepared[0],
        packet_encoded=prepared[1],
        concept_encoded=prepared[2],
        policy_encoded=prepared[3],
        identity_encoded=prepared[4],
        draft_encoded=prepared[5],
        output_encoded=expected_output,
    )
    if observed_output != expected_output or payload != expected:
        raise AuthoredReviewCompileError("compiled review replay differs")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compile", "validate"))
    for name in (
        "candidate",
        "candidate-receipt",
        "review-packet",
        "review-key",
        "review-packet-receipt",
        "concept-list",
        "annotation-policy",
        "reviewer-identity",
        "draft",
        "output",
        "receipt-output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    function = compile_review if command == "compile" else validate_compiled_review
    payload = function(**args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["compiled_reviews"]["rows"],
                "receipt_sha256": payload["receipt_sha256"],
                "audit_qualified": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
