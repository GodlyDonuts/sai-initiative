"""Compare two sealed candidate model reviews for human-review triage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.authored_review_compile import DRAFT_SCHEMA
from sai.data.authored_review_model import validate_result
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-authored-curriculum-cross-family-model-review-comparison-v1"
_DRAFT_KEYS = {
    "schema",
    "review_identity_sha256",
    "instructional_quality_ppm",
    "assumed_prior_concepts",
    "taught_concepts",
    "defects",
    "admission_recommendation",
}
_DIMENSIONS = (
    "taught_concepts",
    "assumed_prior_concepts",
    "instructional_quality",
    "admission_recommendation",
    "defect_categories",
)


class AuthoredReviewModelCompareError(RuntimeError):
    """The candidate model-review comparison or its evidence differs."""


def _rows(root: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        encoded = _read_regular_bytes(root / "draft.jsonl", maximum_bytes=1 << 30)
        rows = [json.loads(line) for line in encoded.decode().splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthoredReviewModelCompareError("model review draft differs") from error
    if len(rows) != 127:
        raise AuthoredReviewModelCompareError("model review population differs")
    identities = []
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != _DRAFT_KEYS
            or row["schema"] != DRAFT_SCHEMA
            or not isinstance(row["review_identity_sha256"], str)
            or len(row["review_identity_sha256"]) != 64
        ):
            raise AuthoredReviewModelCompareError("model review row differs")
        identities.append(row["review_identity_sha256"])
    if len(identities) != len(set(identities)):
        raise AuthoredReviewModelCompareError("model review identities differ")
    return rows, encoded


def _concepts(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(item["concept_id"] for item in row["taught_concepts"])


def _defects(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(item["category"] for item in row["defects"]))


def _ppm(numerator: int, denominator: int) -> int:
    return (numerator * 1_000_000 + denominator - 1) // denominator


def _payload(
    *,
    qwen_model_root: Path,
    qwen_manifest: Path,
    qwen_restoration_receipt: Path,
    qwen_review_root: Path,
    smol_model_root: Path,
    smol_manifest: Path,
    smol_restoration_receipt: Path,
    smol_review_root: Path,
    review_packet: Path,
    review_packet_receipt: Path,
    expected_review_packet_sha256: str,
    expected_review_packet_receipt_sha256: str,
    concept_list: Path,
    annotation_policy: Path,
) -> dict[str, Any]:
    common = {
        "review_packet": review_packet,
        "review_packet_receipt": review_packet_receipt,
        "expected_review_packet_sha256": expected_review_packet_sha256,
        "expected_review_packet_receipt_sha256": (
            expected_review_packet_receipt_sha256
        ),
        "concept_list": concept_list,
        "annotation_policy": annotation_policy,
    }
    try:
        qwen_receipt = validate_result(
            reviewer="qwen35_9b",
            model_root=qwen_model_root,
            manifest=qwen_manifest,
            restoration_receipt=qwen_restoration_receipt,
            output_root=qwen_review_root,
            **common,
        )
        smol_receipt = validate_result(
            reviewer="smollm3_3b",
            model_root=smol_model_root,
            manifest=smol_manifest,
            restoration_receipt=smol_restoration_receipt,
            output_root=smol_review_root,
            **common,
        )
        qwen_rows, qwen_encoded = _rows(qwen_review_root)
        smol_rows, smol_encoded = _rows(smol_review_root)
        packet_encoded = _read_regular_bytes(review_packet, maximum_bytes=1 << 30)
        packet_receipt_encoded = _read_regular_bytes(
            review_packet_receipt, maximum_bytes=1 << 20
        )
        concept_encoded = _read_regular_bytes(concept_list, maximum_bytes=8 << 20)
        policy_encoded = _read_regular_bytes(annotation_policy, maximum_bytes=1 << 20)
    except Exception as error:
        raise AuthoredReviewModelCompareError(
            "model review evidence differs"
        ) from error
    if [row["review_identity_sha256"] for row in qwen_rows] != [
        row["review_identity_sha256"] for row in smol_rows
    ]:
        raise AuthoredReviewModelCompareError("model review order differs")
    counts = {dimension: 0 for dimension in _DIMENSIONS}
    rows = []
    for index, (qwen, smol) in enumerate(zip(qwen_rows, smol_rows, strict=True)):
        qwen_taught = _concepts(qwen)
        smol_taught = _concepts(smol)
        qwen_assumed = tuple(qwen["assumed_prior_concepts"])
        smol_assumed = tuple(smol["assumed_prior_concepts"])
        quality_delta = abs(
            qwen["instructional_quality_ppm"] - smol["instructional_quality_ppm"]
        )
        differences = {
            "taught_concepts": qwen_taught != smol_taught,
            "assumed_prior_concepts": qwen_assumed != smol_assumed,
            "instructional_quality": quality_delta > 100_000,
            "admission_recommendation": qwen["admission_recommendation"]
            != smol["admission_recommendation"],
            "defect_categories": _defects(qwen) != _defects(smol),
        }
        for dimension, differs in differences.items():
            counts[dimension] += int(differs)
        rows.append(
            {
                "index": index,
                "review_identity_sha256": qwen["review_identity_sha256"],
                "differing_dimensions": [
                    dimension for dimension in _DIMENSIONS if differences[dimension]
                ],
                "priority_score": sum(differences.values()) * 1_000_001 + quality_delta,
                "quality_delta_ppm": quality_delta,
                "qwen_taught_concepts": list(qwen_taught),
                "smol_taught_concepts": list(smol_taught),
                "qwen_assumed_prior_concepts": list(qwen_assumed),
                "smol_assumed_prior_concepts": list(smol_assumed),
                "qwen_admission_recommendation": qwen["admission_recommendation"],
                "smol_admission_recommendation": smol["admission_recommendation"],
                "qwen_defect_categories": list(_defects(qwen)),
                "smol_defect_categories": list(_defects(smol)),
            }
        )
    review_priority = sorted(
        [row for row in rows if row["differing_dimensions"]],
        key=lambda row: (-row["priority_score"], row["review_identity_sha256"]),
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "candidate_model_triage_complete",
        "reviewers": ["qwen35_9b", "smollm3_3b"],
        "reviewed_documents": len(rows),
        "disagreement_documents": len(review_priority),
        "observed_disagreement_ppm": {
            dimension: _ppm(counts[dimension], len(rows)) for dimension in _DIMENSIONS
        },
        "quality_delta_threshold_ppm": 100_000,
        "review_priority": review_priority,
        "inputs": {
            "qwen_result_receipt_sha256": qwen_receipt["receipt_sha256"],
            "qwen_draft_file_sha256": hashlib.sha256(qwen_encoded).hexdigest(),
            "smol_result_receipt_sha256": smol_receipt["receipt_sha256"],
            "smol_draft_file_sha256": hashlib.sha256(smol_encoded).hexdigest(),
            "blind_review_packet_sha256": hashlib.sha256(packet_encoded).hexdigest(),
            "blind_review_packet_receipt_sha256": hashlib.sha256(
                packet_receipt_encoded
            ).hexdigest(),
            "concept_list_sha256": hashlib.sha256(concept_encoded).hexdigest(),
            "annotation_policy_sha256": hashlib.sha256(policy_encoded).hexdigest(),
        },
        "human_review_completed": False,
        "audit_qualified": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
        "limitations": [
            "model_model_agreement_is_candidate_triage_evidence_only",
            "independent_bound_human_reviews_remain_mandatory",
            "no_data_admission_training_or_architecture_promotion_is_authorized",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def run(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    payload = _payload(**kwargs)
    _write_create_only(
        output, json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    )
    return payload


def validate(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    try:
        actual = json.loads(_read_regular_bytes(output, maximum_bytes=8 << 20))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthoredReviewModelCompareError("comparison receipt differs") from error
    expected = _payload(**kwargs)
    if actual != expected:
        raise AuthoredReviewModelCompareError("comparison receipt differs")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compare", "validate"))
    for name in (
        "qwen-model-root",
        "qwen-manifest",
        "qwen-restoration-receipt",
        "qwen-review-root",
        "smol-model-root",
        "smol-manifest",
        "smol-restoration-receipt",
        "smol-review-root",
        "review-packet",
        "review-packet-receipt",
        "concept-list",
        "annotation-policy",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-review-packet-sha256", required=True)
    parser.add_argument("--expected-review-packet-receipt-sha256", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    payload = (run if command == "compare" else validate)(**args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "reviewed_documents": payload["reviewed_documents"],
                "disagreement_documents": payload["disagreement_documents"],
                "receipt_sha256": payload["receipt_sha256"],
                "audit_qualified": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
