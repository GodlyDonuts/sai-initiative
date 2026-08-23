"""Turn one completed coverage audit into a conservative source-work ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.reservoir_audit_aggregate import SCHEMA as AGGREGATE_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-reservoir-audit-source-decision-v1"
ROUTES = (
    "quarantine",
    "rights_hold",
    "factual_grounding_review",
    "translation_review",
    "cleanup_review",
    "transformation_review",
    "representation_verification",
)
LANES = {
    "quarantine": "quarantine_exclusion_review",
    "rights_hold": "rights_resolution_before_content_use",
    "factual_grounding_review": "source_grounding_verification",
    "translation_review": "english_translation_and_cultural_fidelity_review",
    "cleanup_review": "deterministic_cleanup_and_ocr_review",
    "transformation_review": "source_bound_transformation_verification",
    "representation_verification": "representation_verification",
}


class ReservoirAuditDecisionError(RuntimeError):
    """The completed aggregate or derived source decision differs."""


def _load_aggregate(path: Path) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size > 8 << 20
    ):
        raise ReservoirAuditDecisionError("aggregate input is missing or unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ReservoirAuditDecisionError(
            "aggregate input cannot be decoded"
        ) from error
    if not isinstance(payload, dict):
        raise ReservoirAuditDecisionError("aggregate input differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != AGGREGATE_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("coverage_first_not_statistical_acceptance_estimate") is not True
        or payload.get("independent_factual_verification_complete") is not False
        or payload.get("cross_source_deduplication_complete") is not False
        or payload.get("benchmark_decontamination_complete") is not False
        or payload.get("training_ready") is not False
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise ReservoirAuditDecisionError("aggregate receipt differs")
    return payload


def _nonnegative_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReservoirAuditDecisionError(f"{label} differs")
    return value


def _source_decisions(summary: Any) -> list[dict[str, Any]]:
    if not isinstance(summary, dict):
        raise ReservoirAuditDecisionError("aggregate summary differs")
    routes_by_source = summary.get("by_source_conservative_triage")
    verdicts_by_source = summary.get("by_source_verdict")
    rows = _nonnegative_count(summary.get("rows"), "aggregate rows")
    if (
        rows == 0
        or not isinstance(routes_by_source, dict)
        or not routes_by_source
        or not isinstance(verdicts_by_source, dict)
        or set(routes_by_source) != set(verdicts_by_source)
        or summary.get("model_judgments_are_verified_admissions") is not False
        or summary.get("representation_verification_is_training_admission") is not False
    ):
        raise ReservoirAuditDecisionError("aggregate source coverage differs")
    decisions = []
    covered_rows = 0
    for source_id in sorted(routes_by_source):
        raw_routes = routes_by_source[source_id]
        raw_verdicts = verdicts_by_source[source_id]
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(raw_routes, dict)
            or not raw_routes
            or any(route not in ROUTES for route in raw_routes)
            or not isinstance(raw_verdicts, dict)
            or not raw_verdicts
        ):
            raise ReservoirAuditDecisionError("aggregate source route differs")
        route_counts = {
            route: _nonnegative_count(raw_routes.get(route, 0), f"{source_id} route")
            for route in ROUTES
        }
        source_rows = sum(route_counts.values())
        verdict_rows = sum(
            _nonnegative_count(value, f"{source_id} verdict")
            for value in raw_verdicts.values()
        )
        if source_rows == 0 or source_rows != verdict_rows:
            raise ReservoirAuditDecisionError("aggregate source row custody differs")
        covered_rows += source_rows
        route_ppm = {
            route: (count * 1_000_000) // source_rows
            for route, count in route_counts.items()
        }
        blocking_ppm = route_ppm["quarantine"] + route_ppm["rights_hold"]
        if route_ppm["rights_hold"] >= 500_000:
            action = "rights_blocked_pending_source_specific_resolution"
        elif blocking_ppm >= 500_000:
            action = "bulk_expansion_paused_pending_stratified_confirmation"
        elif (
            route_ppm["representation_verification"] >= 300_000
            and route_ppm["quarantine"] <= 200_000
        ):
            action = "priority_targeted_verification"
        else:
            action = "targeted_recovery_and_verification"
        decisions.append(
            {
                "source_id": source_id,
                "observed_rows": source_rows,
                "route_counts": route_counts,
                "route_ppm": route_ppm,
                "dominant_routes": sorted(
                    (route for route, count in route_counts.items() if count),
                    key=lambda route: (-route_counts[route], ROUTES.index(route)),
                ),
                "required_work_lanes": [
                    LANES[route] for route in ROUTES if route_counts[route]
                ],
                "next_action": action,
                "bulk_training_admission": False,
            }
        )
    if covered_rows != rows:
        raise ReservoirAuditDecisionError(
            "aggregate source rows do not cover population"
        )
    return decisions


def build_decision(aggregate_path: Path, output_path: Path) -> dict[str, Any]:
    """Create a hash-bound work ledger without estimating a source yield."""

    if output_path.exists() or output_path.is_symlink():
        raise ReservoirAuditDecisionError("decision output already exists")
    aggregate = _load_aggregate(aggregate_path)
    decisions = _source_decisions(aggregate.get("summary"))
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "aggregate": {
            "path": aggregate_path.name,
            "bytes": aggregate_path.stat().st_size,
            "sha256": sha256_file(aggregate_path),
            "receipt_sha256": aggregate["receipt_sha256"],
        },
        "method": {
            "coverage_screen_not_acceptance_rate_estimate": True,
            "route_proportions_are_descriptive_only": True,
            "minimum_rights_hold_ppm_for_rights_block": 500_000,
            "minimum_quarantine_plus_rights_ppm_for_bulk_pause": 500_000,
            "minimum_representation_verification_ppm_for_priority": 300_000,
            "maximum_quarantine_ppm_for_priority": 200_000,
            "silent_source_reweighting_allowed": False,
        },
        "sources": decisions,
        "source_rows": sum(row["observed_rows"] for row in decisions),
        "source_count": len(decisions),
        "independent_verification_complete": False,
        "benchmark_decontamination_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_decision(args.aggregate, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
