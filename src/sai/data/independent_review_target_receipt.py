"""Seal exact target and coverage for a stratified independent review lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.independent_review_compare import (
    POPULATION_SCHEMA,
    ReviewLane,
    _load_json,
    _validate_review_receipt,
)
from sai.data.nous_label_worker import _load_jsonl
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-independent-review-target-coverage-v1"


class IndependentReviewTargetError(RuntimeError):
    """The frozen population, target strata, or review custody differs."""


def build_target_receipt(
    population_root: Path,
    lane: ReviewLane,
    target_strata: set[str],
    output: Path,
) -> dict[str, Any]:
    """Write source-safe coverage for exactly the selected review strata."""

    if output.exists() or output.is_symlink() or not target_strata:
        raise IndependentReviewTargetError("targeted review inputs differ")
    population = _load_json(population_root / "receipt.json")
    unsigned = {
        key: value
        for key, value in population.items()
        if key != "receipt_sha256"
    }
    candidates_path = population_root / "candidates.jsonl"
    candidates = _load_jsonl(candidates_path)
    descriptors = population.get("selected_descriptors")
    if (
        population.get("schema") != POPULATION_SCHEMA
        or population.get("status")
        != "complete_nontraining_independent_review_population"
        or population.get("receipt_sha256") != canonical_sha256(unsigned)
        or population.get("training_ready") is not False
        or population.get("population", {}).get("rows") != len(candidates)
        or population.get("population", {}).get("sha256")
        != sha256_file(candidates_path)
        or not isinstance(descriptors, list)
        or len(descriptors) != len(candidates)
    ):
        raise IndependentReviewTargetError("review population differs")
    by_identity = {row["candidate_identity_sha256"]: row for row in candidates}
    selected = [row for row in descriptors if row.get("stratum") in target_strata]
    if (
        not selected
        or len(by_identity) != len(candidates)
        or any(
            row.get("candidate_identity_sha256") not in by_identity
            for row in selected
        )
    ):
        raise IndependentReviewTargetError("target identities differ")
    rows = []
    covered = 0
    for descriptor in sorted(
        selected, key=lambda row: row["candidate_identity_sha256"]
    ):
        identity = descriptor["candidate_identity_sha256"]
        path = lane.root / f"{identity}.independent-review.json"
        receipt_hash = None
        if path.exists():
            try:
                receipt = _load_json(path)
                _validate_review_receipt(receipt, by_identity[identity], lane)
            except RuntimeError as error:
                raise IndependentReviewTargetError(
                    "target review receipt differs"
                ) from error
            receipt_hash = receipt["receipt_sha256"]
            covered += 1
        rows.append(
            {
                "candidate_identity_sha256": identity,
                "lane": descriptor["lane"],
                "stratum": descriptor["stratum"],
                "review_receipt_sha256": receipt_hash,
            }
        )
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_targeted_review_coverage",
        "population": {
            "root_name": population_root.name,
            "receipt_sha256": population["receipt_sha256"],
            "rows": len(candidates),
        },
        "review_lane": {
            "name": lane.name,
            "model": lane.model,
            "endpoint": lane.endpoint,
            "root_name": lane.root.name,
        },
        "target_strata": sorted(target_strata),
        "counts": {
            "target_rows": len(rows),
            "covered_rows": covered,
            "missing_or_failed_rows": len(rows) - covered,
        },
        "rows": rows,
        "missing_rows_require_adjudication": True,
        "source_text_persisted": False,
        "automatic_training_admission": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--target-stratum", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_target_receipt(
        args.population_root,
        ReviewLane(args.name, args.model, args.endpoint, args.review_root),
        set(args.target_stratum),
        args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "counts": result["counts"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
