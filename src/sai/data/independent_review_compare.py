"""Compare primary compiler judgments with independent provider reviews."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.data_compiler_labeling import (
    RUBRIC_SHA256,
    validate_normalized_judgment,
)
from sai.data.independent_compiler_review_worker import (
    ALLOWED_PROVIDER_MODELS,
    RECEIPT_SCHEMA,
)
from sai.data.nous_label_worker import _load_jsonl
from sai.data.reservoir_audit_aggregate import (
    _triage_route,
    _validate_compiler_receipt,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-independent-review-comparison-v1"
POPULATION_SCHEMA = "sai-independent-review-population-receipt-v1"


class IndependentReviewCompareError(RuntimeError):
    """The primary or independent review evidence differs."""


@dataclass(frozen=True)
class ReviewLane:
    name: str
    model: str
    endpoint: str
    root: Path


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise IndependentReviewCompareError("review evidence is missing or unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise IndependentReviewCompareError("review evidence is invalid") from error
    if not isinstance(value, dict):
        raise IndependentReviewCompareError("review evidence is invalid")
    return value


def _active_risks(judgment: dict[str, Any]) -> list[str]:
    return sorted(key for key, value in judgment["risks"].items() if value)


def summarize_row(
    primary: dict[str, Any], reviews: dict[str, dict[str, Any] | None]
) -> dict[str, Any]:
    """Summarize agreement without promoting any judgment."""

    primary_summary = {
        "verdict": primary["verdict"],
        "route": _triage_route(primary),
        "active_risks": _active_risks(primary),
    }
    review_summaries = {}
    available = [primary_summary]
    for name, judgment in sorted(reviews.items()):
        if judgment is None:
            review_summaries[name] = None
            continue
        summary = {
            "verdict": judgment["verdict"],
            "route": _triage_route(judgment),
            "active_risks": _active_risks(judgment),
        }
        review_summaries[name] = summary
        available.append(summary)
    complete = all(value is not None for value in reviews.values())
    verdict_agree = len({value["verdict"] for value in available}) == 1
    route_agree = len({value["route"] for value in available}) == 1
    risks_agree = len(
        {tuple(value["active_risks"]) for value in available}
    ) == 1
    return {
        "primary": primary_summary,
        "reviews": review_summaries,
        "complete_review_coverage": complete,
        "all_available_verdict_agree": verdict_agree,
        "all_available_route_agree": route_agree,
        "all_available_active_risks_agree": risks_agree,
        "manual_adjudication_required": not (
            complete and verdict_agree and route_agree and risks_agree
        ),
        "automatic_training_admission": False,
    }


def _validate_review_receipt(
    receipt: dict[str, Any],
    candidate: dict[str, Any],
    lane: ReviewLane,
) -> dict[str, Any]:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        lane.model not in ALLOWED_PROVIDER_MODELS.get(lane.endpoint, frozenset())
        or receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("candidate_identity_sha256")
        != candidate["candidate_identity_sha256"]
        or receipt.get("rubric_sha256") != RUBRIC_SHA256
        or receipt.get("requested_model") != lane.model
        or receipt.get("endpoint_origin") != lane.endpoint
        or receipt.get("request_reasoning_effort") is not None
        or receipt.get("api_key_persisted") is not False
        or receipt.get("raw_source_is_training_data") is not False
        or receipt.get("training_ready") is not False
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise IndependentReviewCompareError("independent review receipt differs")
    try:
        judgment = validate_normalized_judgment(receipt.get("judgment"), candidate)
    except RuntimeError as error:
        raise IndependentReviewCompareError(
            "independent review judgment differs"
        ) from error
    return judgment


def build_comparison(
    population_root: Path,
    primary_roots: dict[str, Path],
    review_lanes: list[ReviewLane],
    output: Path,
) -> dict[str, Any]:
    """Build a source-safe comparison and explicit adjudication queue."""

    if (
        output.exists()
        or output.is_symlink()
        or not primary_roots
        or not review_lanes
        or len({lane.name for lane in review_lanes}) != len(review_lanes)
    ):
        raise IndependentReviewCompareError("review comparison inputs differ")
    population_receipt = _load_json(population_root / "receipt.json")
    unsigned_population = {
        key: value
        for key, value in population_receipt.items()
        if key != "receipt_sha256"
    }
    candidates_path = population_root / "candidates.jsonl"
    candidates = _load_jsonl(candidates_path)
    descriptors = population_receipt.get("selected_descriptors")
    if (
        population_receipt.get("schema") != POPULATION_SCHEMA
        or population_receipt.get("status")
        != "complete_nontraining_independent_review_population"
        or population_receipt.get("receipt_sha256")
        != canonical_sha256(unsigned_population)
        or population_receipt.get("training_ready") is not False
        or population_receipt.get("population", {}).get("rows") != len(candidates)
        or population_receipt.get("population", {}).get("sha256")
        != sha256_file(candidates_path)
        or not isinstance(descriptors, list)
        or len(descriptors) != len(candidates)
    ):
        raise IndependentReviewCompareError("review population differs")
    by_identity = {row["candidate_identity_sha256"]: row for row in candidates}
    descriptor_by_identity = {
        row["candidate_identity_sha256"]: row for row in descriptors
    }
    if (
        len(by_identity) != len(candidates)
        or set(by_identity) != set(descriptor_by_identity)
    ):
        raise IndependentReviewCompareError("review population identities differ")

    provider_counts = {
        lane.name: Counter(
            {
                "expected_rows": len(candidates),
                "covered_rows": 0,
                "verdict_agreement_rows": 0,
                "route_agreement_rows": 0,
                "active_risk_agreement_rows": 0,
            }
        )
        for lane in review_lanes
    }
    comparison_rows = []
    for identity in sorted(by_identity):
        candidate = by_identity[identity]
        descriptor = descriptor_by_identity[identity]
        primary_root = primary_roots.get(descriptor["lane"])
        if primary_root is None:
            raise IndependentReviewCompareError("primary review lane differs")
        primary_receipt = _load_json(primary_root / f"{identity}.compiler.json")
        try:
            primary_receipt = _validate_compiler_receipt(
                primary_receipt, candidate
            )
        except RuntimeError as error:
            raise IndependentReviewCompareError(
                "primary review receipt differs"
            ) from error
        if (
            primary_receipt["receipt_sha256"]
            != descriptor["primary_receipt_sha256"]
            or primary_receipt["judgment"]["judgment_sha256"]
            != descriptor["primary_judgment_sha256"]
        ):
            raise IndependentReviewCompareError("primary review binding differs")
        reviews = {}
        review_receipt_hashes = {}
        for lane in review_lanes:
            path = lane.root / f"{identity}.independent-review.json"
            if not path.exists():
                reviews[lane.name] = None
                review_receipt_hashes[lane.name] = None
                continue
            receipt = _load_json(path)
            reviews[lane.name] = _validate_review_receipt(receipt, candidate, lane)
            review_receipt_hashes[lane.name] = receipt["receipt_sha256"]
        summary = summarize_row(primary_receipt["judgment"], reviews)
        for lane in review_lanes:
            independent = summary["reviews"][lane.name]
            if independent is None:
                continue
            counts = provider_counts[lane.name]
            counts["covered_rows"] += 1
            counts["verdict_agreement_rows"] += int(
                independent["verdict"] == summary["primary"]["verdict"]
            )
            counts["route_agreement_rows"] += int(
                independent["route"] == summary["primary"]["route"]
            )
            counts["active_risk_agreement_rows"] += int(
                independent["active_risks"] == summary["primary"]["active_risks"]
            )
        comparison_rows.append(
            {
                "candidate_identity_sha256": identity,
                "lane": descriptor["lane"],
                "stratum": descriptor["stratum"],
                "primary_receipt_sha256": primary_receipt["receipt_sha256"],
                "review_receipt_sha256s": review_receipt_hashes,
                **summary,
            }
        )
    counts = Counter()
    for row in comparison_rows:
        counts["rows"] += 1
        counts["complete_review_coverage_rows"] += int(
            row["complete_review_coverage"]
        )
        counts["unanimous_verdict_rows"] += int(
            row["all_available_verdict_agree"]
            and row["complete_review_coverage"]
        )
        counts["unanimous_route_rows"] += int(
            row["all_available_route_agree"]
            and row["complete_review_coverage"]
        )
        counts["unanimous_active_risk_rows"] += int(
            row["all_available_active_risks_agree"]
            and row["complete_review_coverage"]
        )
        counts["manual_adjudication_rows"] += int(
            row["manual_adjudication_required"]
        )
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_independent_review_comparison",
        "population": {
            "root_name": population_root.name,
            "receipt_sha256": population_receipt["receipt_sha256"],
            "rows": len(candidates),
        },
        "review_lanes": [
            {
                "name": lane.name,
                "model": lane.model,
                "endpoint": lane.endpoint,
                "root_name": lane.root.name,
            }
            for lane in review_lanes
        ],
        "counts": dict(sorted(counts.items())),
        "by_provider": {
            name: dict(sorted(values.items()))
            for name, values in sorted(provider_counts.items())
        },
        "rows": comparison_rows,
        "source_text_persisted": False,
        "comparison_is_automatic_training_admission": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def _mapping(value: str) -> tuple[str, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("mapping must be NAME=PATH")
    return parts[0], Path(parts[1])


def _review(value: str) -> ReviewLane:
    parts = value.split("=", 1)
    values = parts[1].split(",", 2) if len(parts) == 2 else []
    if len(values) != 3 or not parts[0] or not all(values):
        raise argparse.ArgumentTypeError("review must be NAME=MODEL,ENDPOINT,ROOT")
    return ReviewLane(parts[0], values[0], values[1], Path(values[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--primary", type=_mapping, action="append", required=True)
    parser.add_argument("--review", type=_review, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_comparison(
        args.population_root, dict(args.primary), args.review, args.output
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "by_provider": result["by_provider"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
