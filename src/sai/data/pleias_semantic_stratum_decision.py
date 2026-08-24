"""Require strong primary quality and cross-family support per PleIAs stratum."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.data_compiler_labeling import SCORE_KEYS
from sai.data.independent_review_compare import SCHEMA as COMPARISON_SCHEMA
from sai.data.reservoir_audit_aggregate import (
    _triage_route,
    _validate_compiler_receipt,
    load_population,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pleias-semantic-stratum-decision-v1"
MINIMUM_PRIMARY_ROWS = 16
MINIMUM_PRIMARY_REPRESENTATION_PPM = 875_000
MINIMUM_CORE_SCORE_MILLI = 3_750
MINIMUM_INDEPENDENT_ROWS = 4
MINIMUM_INDEPENDENT_REPRESENTATION_PPM = 875_000
MINIMUM_ROUTE_AGREEMENT_PPM = 875_000
CORE_SCORES = (
    "information_density",
    "educational_value",
    "source_reliability",
    "coherence",
)
BLOCKING_ROUTES = frozenset({"quarantine", "rights_hold"})


class PleiasSemanticStratumDecisionError(RuntimeError):
    """The semantic population, judgments, or cross-family evidence differs."""


def _ppm(numerator: int, denominator: int) -> int:
    return numerator * 1_000_000 // denominator if denominator else 0


def decide_strata(
    primary_rows: list[tuple[str, dict[str, Any]]],
    comparison_rows: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Apply the frozen conservative stratum-level advancement policy."""

    if not primary_rows:
        raise PleiasSemanticStratumDecisionError("primary evidence is empty")
    primary_counts: dict[str, Counter[str]] = defaultdict(Counter)
    score_sums: dict[str, Counter[str]] = defaultdict(Counter)
    pedagogy: dict[str, Counter[str]] = defaultdict(Counter)
    concepts: dict[str, Counter[str]] = defaultdict(Counter)
    prerequisites: dict[str, Counter[str]] = defaultdict(Counter)
    for stratum, judgment in primary_rows:
        if not isinstance(stratum, str) or not stratum:
            raise PleiasSemanticStratumDecisionError("primary stratum differs")
        route = _triage_route(judgment)
        primary_counts[stratum]["rows"] += 1
        primary_counts[stratum][f"route::{route}"] += 1
        for score in SCORE_KEYS:
            score_sums[stratum][score] += judgment["scores"][score]
        pedagogy[stratum]["difficulty_sum"] += judgment["difficulty"]
        pedagogy[stratum]["prerequisite_burden_sum"] += judgment[
            "prerequisite_burden"
        ]
        pedagogy[stratum][f"phase::{judgment['curriculum_phase']}"] += 1
        for domain in judgment["domains"]:
            pedagogy[stratum][f"domain::{domain}"] += 1
        for concept in judgment["concepts_taught"]:
            concepts[stratum][concept] += 1
        for concept in judgment["prerequisites_assumed"]:
            prerequisites[stratum][concept] += 1
    independent_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for stratum, row in comparison_rows:
        if not isinstance(stratum, str) or not stratum:
            raise PleiasSemanticStratumDecisionError("independent stratum differs")
        counts = independent_counts[stratum]
        counts["rows"] += 1
        complete = row.get("complete_review_coverage") is True
        counts["complete_rows"] += int(complete)
        counts["route_agreement_rows"] += int(
            complete and row.get("all_available_route_agree") is True
        )
        reviews = row.get("reviews")
        if not isinstance(reviews, dict):
            raise PleiasSemanticStratumDecisionError(
                "independent review evidence differs"
            )
        for review in reviews.values():
            if review is None:
                continue
            route = review.get("route")
            if not isinstance(route, str):
                raise PleiasSemanticStratumDecisionError("independent route differs")
            counts[f"route::{route}"] += 1
            counts["review_decisions"] += 1
    decisions = []
    for stratum in sorted(primary_counts):
        primary = primary_counts[stratum]
        independent = independent_counts[stratum]
        primary_rows_count = primary["rows"]
        primary_representation_ppm = _ppm(
            primary["route::representation_verification"], primary_rows_count
        )
        primary_blocking = sum(primary[f"route::{route}"] for route in BLOCKING_ROUTES)
        means = {
            score: score_sums[stratum][score] * 1000 // primary_rows_count
            for score in CORE_SCORES
        }
        phase_counts = {
            key.removeprefix("phase::"): value
            for key, value in sorted(pedagogy[stratum].items())
            if key.startswith("phase::")
        }
        dominant_phase = min(
            phase_counts,
            key=lambda phase: (-phase_counts[phase], phase),
        )
        independent_decisions = independent["review_decisions"]
        independent_representation_ppm = _ppm(
            independent["route::representation_verification"],
            independent_decisions,
        )
        independent_blocking = sum(
            independent[f"route::{route}"] for route in BLOCKING_ROUTES
        )
        route_agreement_ppm = _ppm(
            independent["route_agreement_rows"], independent["rows"]
        )
        reasons = []
        if primary_rows_count < MINIMUM_PRIMARY_ROWS:
            reasons.append("insufficient_primary_rows")
        if primary_representation_ppm < MINIMUM_PRIMARY_REPRESENTATION_PPM:
            reasons.append("primary_representation_rate_below_threshold")
        if primary_blocking:
            reasons.append("primary_blocking_route_present")
        if any(value < MINIMUM_CORE_SCORE_MILLI for value in means.values()):
            reasons.append("primary_core_score_below_threshold")
        if independent["rows"] < MINIMUM_INDEPENDENT_ROWS:
            reasons.append("insufficient_independent_rows")
        if independent["complete_rows"] != independent["rows"]:
            reasons.append("incomplete_independent_coverage")
        if independent_representation_ppm < MINIMUM_INDEPENDENT_REPRESENTATION_PPM:
            reasons.append("independent_representation_rate_below_threshold")
        if independent_blocking:
            reasons.append("independent_blocking_route_present")
        if route_agreement_ppm < MINIMUM_ROUTE_AGREEMENT_PPM:
            reasons.append("cross_family_route_agreement_below_threshold")
        decision = {
            "stratum": stratum,
            "decision": (
                "advance_to_full_candidate_decontamination"
                if not reasons
                else "hold_semantic_stratum"
            ),
            "reasons": reasons,
            "primary": {
                "rows": primary_rows_count,
                "route_counts": {
                    key.removeprefix("route::"): value
                    for key, value in sorted(primary.items())
                    if key.startswith("route::")
                },
                "representation_verification_ppm": primary_representation_ppm,
                "core_mean_scores_milli": means,
                "difficulty_mean_milli": (
                    pedagogy[stratum]["difficulty_sum"]
                    * 1000
                    // primary_rows_count
                ),
                "prerequisite_burden_mean_milli": (
                    pedagogy[stratum]["prerequisite_burden_sum"]
                    * 1000
                    // primary_rows_count
                ),
                "curriculum_phase_counts": phase_counts,
                "dominant_curriculum_phase": dominant_phase,
                "domain_counts": {
                    key.removeprefix("domain::"): value
                    for key, value in sorted(pedagogy[stratum].items())
                    if key.startswith("domain::")
                },
                "recurring_concepts": [
                    {"concept": concept, "votes": votes}
                    for concept, votes in sorted(concepts[stratum].items())
                    if votes >= 2
                ],
                "recurring_prerequisites": [
                    {"concept": concept, "votes": votes}
                    for concept, votes in sorted(prerequisites[stratum].items())
                    if votes >= 2
                ],
            },
            "independent": {
                "rows": independent["rows"],
                "complete_rows": independent["complete_rows"],
                "review_decisions": independent_decisions,
                "route_counts": {
                    key.removeprefix("route::"): value
                    for key, value in sorted(independent.items())
                    if key.startswith("route::")
                },
                "representation_verification_ppm": independent_representation_ppm,
                "route_agreement_ppm": route_agreement_ppm,
            },
            "automatic_training_admission": False,
        }
        decision["row_sha256"] = canonical_sha256(decision)
        decisions.append(decision)
    return decisions


def _load_comparison(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasSemanticStratumDecisionError("comparison is unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasSemanticStratumDecisionError("comparison is invalid") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not isinstance(value, dict)
        or value.get("schema") != COMPARISON_SCHEMA
        or value.get("status") != "complete_nontraining_independent_review_comparison"
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("training_ready") is not False
    ):
        raise PleiasSemanticStratumDecisionError("comparison differs")
    return value


def build_decision(
    population_root: Path,
    judgments_root: Path,
    comparison_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Replay exact model evidence and write a source-safe stratum decision."""

    if output.exists() or output.is_symlink():
        raise PleiasSemanticStratumDecisionError("decision output exists")
    candidates, lineage, population = load_population(population_root)
    primary_rows = []
    stratum_by_identity = {}
    primary_receipts = []
    for candidate, source in zip(candidates, lineage, strict=True):
        identity = candidate["candidate_identity_sha256"]
        path = judgments_root / f"{identity}.compiler.json"
        try:
            receipt = _validate_compiler_receipt(
                json.loads(path.read_bytes()), candidate
            )
        except (OSError, json.JSONDecodeError, RuntimeError) as error:
            raise PleiasSemanticStratumDecisionError(
                "primary receipt differs"
            ) from error
        primary_rows.append((source["stratum"], receipt["judgment"]))
        stratum_by_identity[identity] = source["stratum"]
        primary_receipts.append(receipt["receipt_sha256"])
    comparison = _load_comparison(comparison_path)
    comparison_rows = []
    for row in comparison.get("rows", []):
        identity = row.get("candidate_identity_sha256")
        if identity not in stratum_by_identity:
            raise PleiasSemanticStratumDecisionError("comparison identity differs")
        comparison_rows.append((stratum_by_identity[identity], row))
    decisions = decide_strata(primary_rows, comparison_rows)
    advanced = [
        row["stratum"]
        for row in decisions
        if row["decision"] == "advance_to_full_candidate_decontamination"
    ]
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_pleias_semantic_stratum_decision",
        "policy": {
            "minimum_primary_rows": MINIMUM_PRIMARY_ROWS,
            "minimum_primary_representation_ppm": MINIMUM_PRIMARY_REPRESENTATION_PPM,
            "minimum_core_score_milli": MINIMUM_CORE_SCORE_MILLI,
            "minimum_independent_rows": MINIMUM_INDEPENDENT_ROWS,
            "minimum_independent_representation_ppm": (
                MINIMUM_INDEPENDENT_REPRESENTATION_PPM
            ),
            "minimum_route_agreement_ppm": MINIMUM_ROUTE_AGREEMENT_PPM,
            "zero_primary_or_independent_blocking_routes": True,
        },
        "evidence": {
            "population_receipt_sha256": population["receipt_sha256"],
            "ordered_primary_receipts_sha256": canonical_sha256(primary_receipts),
            "comparison_file_sha256": sha256_file(comparison_path),
            "comparison_receipt_sha256": comparison["receipt_sha256"],
        },
        "counts": {
            "primary_rows": len(primary_rows),
            "independent_rows": len(comparison_rows),
            "strata": len(decisions),
            "advanced_strata": len(advanced),
            "held_strata": len(decisions) - len(advanced),
        },
        "decisions": decisions,
        "ordered_decisions_sha256": canonical_sha256(
            [row["row_sha256"] for row in decisions]
        ),
        "advanced_strata": advanced,
        "semantic_stratum_decision_is_row_admission": False,
        "full_content_decontamination_complete": False,
        "global_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_decision(
        args.population_root,
        args.judgments_root,
        args.comparison,
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
