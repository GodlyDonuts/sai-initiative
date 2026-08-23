"""Select source-disjoint confirmation lanes from two independent audit gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.benchmark_contamination_screen import SCHEMA as SCREEN_SCHEMA
from sai.data.decontamination import POLICY as CONTAMINATION_POLICY
from sai.data.reservoir_audit_decision import ROUTES, _load_aggregate
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-reservoir-audit-confirmation-plan-v2"
METHOD = {
    "minimum_observed_rows": 4,
    "minimum_representation_verification_ppm": 500_000,
    "maximum_contaminated_rows": 0,
    "maximum_quarantine_rows": 0,
    "maximum_rights_hold_rows": 0,
    "confirmation_rows_per_source": 32,
    "identity_disjoint_rows_required": True,
    "different_parent_when_available": True,
    "source_disjoint_parent_required": False,
    "coverage_screen_not_acceptance_rate_estimate": True,
    "bulk_training_admission_allowed": False,
}


class ReservoirAuditConfirmationPlanError(RuntimeError):
    """The aggregate, contamination screen, or confirmation plan differs."""


def _load_screen(path: Path) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size > 8 << 20
    ):
        raise ReservoirAuditConfirmationPlanError(
            "contamination screen is missing or unsafe"
        )
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ReservoirAuditConfirmationPlanError(
            "contamination screen cannot be decoded"
        ) from error
    if not isinstance(payload, dict):
        raise ReservoirAuditConfirmationPlanError("contamination screen differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    summary = payload.get("summary")
    if (
        payload.get("schema") != SCREEN_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("policy") != CONTAMINATION_POLICY
        or payload.get("policy_sha256") != canonical_sha256(CONTAMINATION_POLICY)
        or payload.get("benchmark_contamination_screen_complete") is not True
        or payload.get("full_source_population_decontaminated") is not False
        or payload.get("training_ready") is not False
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or not isinstance(summary, dict)
        or not isinstance(summary.get("by_source"), dict)
        or summary.get("individual_decisions_persisted") is not False
        or summary.get("source_text_persisted") is not False
    ):
        raise ReservoirAuditConfirmationPlanError("contamination screen differs")
    return payload


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ReservoirAuditConfirmationPlanError(f"{label} differs")
    return value


def _source_rows(
    aggregate: dict[str, Any], screen: dict[str, Any]
) -> list[dict[str, Any]]:
    routes_by_source = aggregate.get("summary", {}).get("by_source_conservative_triage")
    screen_by_source = screen["summary"]["by_source"]
    if (
        not isinstance(routes_by_source, dict)
        or not routes_by_source
        or set(routes_by_source) != set(screen_by_source)
    ):
        raise ReservoirAuditConfirmationPlanError("source population differs")
    results = []
    for source_id in sorted(routes_by_source):
        raw_routes = routes_by_source[source_id]
        raw_screen = screen_by_source[source_id]
        if not isinstance(raw_routes, dict) or not isinstance(raw_screen, dict):
            raise ReservoirAuditConfirmationPlanError("source evidence differs")
        unknown_routes = set(raw_routes) - set(ROUTES)
        if unknown_routes:
            raise ReservoirAuditConfirmationPlanError("source route differs")
        routes = {route: _count(raw_routes.get(route, 0), route) for route in ROUTES}
        observed = sum(routes.values())
        rows = _count(raw_screen.get("rows"), "screen rows")
        contaminated = _count(raw_screen.get("contaminated_rows"), "contaminated rows")
        if observed != rows or rows < METHOD["minimum_observed_rows"]:
            raise ReservoirAuditConfirmationPlanError("source row accounting differs")
        representation_ppm = (
            routes["representation_verification"] * 1_000_000
        ) // observed
        reasons = []
        if contaminated > METHOD["maximum_contaminated_rows"]:
            reasons.append("benchmark_overlap_observed")
        if routes["quarantine"] > METHOD["maximum_quarantine_rows"]:
            reasons.append("quarantine_observed")
        if routes["rights_hold"] > METHOD["maximum_rights_hold_rows"]:
            reasons.append("rights_hold_observed")
        if representation_ppm < METHOD["minimum_representation_verification_ppm"]:
            reasons.append("representation_signal_below_threshold")
        selected = not reasons
        results.append(
            {
                "source_id": source_id,
                "observed_rows": observed,
                "conservative_routes": routes,
                "representation_verification_ppm": representation_ppm,
                "contaminated_rows": contaminated,
                "selected_for_source_disjoint_confirmation": selected,
                "exclusion_reasons": reasons,
                "bulk_training_admission": False,
            }
        )
    if sum(row["observed_rows"] for row in results) != aggregate["summary"]["rows"]:
        raise ReservoirAuditConfirmationPlanError("aggregate row accounting differs")
    if sum(row["observed_rows"] for row in results) != screen["summary"]["rows"]:
        raise ReservoirAuditConfirmationPlanError("screen row accounting differs")
    return results


def build_plan(aggregate_path: Path, screen_path: Path, output_path: Path) -> dict:
    """Create one conservative, create-only confirmation plan."""

    if output_path.exists() or output_path.is_symlink():
        raise ReservoirAuditConfirmationPlanError("confirmation output exists")
    aggregate = _load_aggregate(aggregate_path)
    screen = _load_screen(screen_path)
    sources = _source_rows(aggregate, screen)
    selected = [
        row["source_id"]
        for row in sources
        if row["selected_for_source_disjoint_confirmation"]
    ]
    if not selected:
        raise ReservoirAuditConfirmationPlanError("confirmation selected no sources")
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "aggregate": {
            "path": aggregate_path.name,
            "bytes": aggregate_path.stat().st_size,
            "sha256": sha256_file(aggregate_path),
            "receipt_sha256": aggregate["receipt_sha256"],
        },
        "contamination_screen": {
            "path": screen_path.name,
            "bytes": screen_path.stat().st_size,
            "sha256": sha256_file(screen_path),
            "receipt_sha256": screen["receipt_sha256"],
        },
        "method": METHOD,
        "method_sha256": canonical_sha256(METHOD),
        "sources": sources,
        "selected_source_ids": selected,
        "selected_sources": len(selected),
        "confirmation_rows_per_source": METHOD["confirmation_rows_per_source"],
        "target_confirmation_rows": len(selected)
        * METHOD["confirmation_rows_per_source"],
        "bulk_training_admission": False,
        "benchmark_decontamination_of_confirmation_rows_complete": False,
        "independent_verification_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--contamination-screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_plan(args.aggregate, args.contamination_screen, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
