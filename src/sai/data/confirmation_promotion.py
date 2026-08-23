"""Combine independent confirmation evidence into bounded source-pilot decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.benchmark_contamination_screen import SCHEMA as SCREEN_SCHEMA
from sai.data.cross_population_duplicates import SCHEMA as DUPLICATE_SCHEMA
from sai.data.reservoir_audit_aggregate import SCHEMA as AGGREGATE_SCHEMA
from sai.data.reservoir_audit_population import SCHEMA as POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-confirmation-source-promotion-v1"
METHOD = {
    "minimum_confirmation_rows_per_source": 32,
    "minimum_representation_verification_ppm": 500_000,
    "maximum_quarantine_rows": 0,
    "maximum_rights_hold_rows": 0,
    "maximum_benchmark_contaminated_rows": 0,
    "maximum_exact_or_normalized_duplicate_pairs": 0,
    "exact_identity_disjointness_required": True,
    "exact_content_disjointness_required": True,
    "different_parent_required": False,
    "different_parent_preferred_when_available": True,
    "decision_scope": "bounded_streaming_source_pilot_only",
    "bulk_training_admission": False,
}


class ConfirmationPromotionError(RuntimeError):
    """A confirmation artifact, binding, or promotion decision differs."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    path = Path(path)
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size > 16 << 20
    ):
        raise ConfirmationPromotionError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ConfirmationPromotionError(f"{label} cannot be decoded") from error
    if not isinstance(payload, dict):
        raise ConfirmationPromotionError(f"{label} differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != canonical_sha256(unsigned):
        raise ConfirmationPromotionError(f"{label} receipt differs")
    return payload


def _validate_inputs(
    population: dict[str, Any],
    aggregate: dict[str, Any],
    screen: dict[str, Any],
    duplicates: dict[str, Any],
) -> None:
    population_sha = population.get("receipt_sha256")
    descriptors = duplicates.get("populations")
    confirmation_descriptors = (
        [row for row in descriptors if row.get("receipt_sha256") == population_sha]
        if isinstance(descriptors, list)
        else []
    )
    if (
        population.get("schema") != POPULATION_SCHEMA
        or population.get("status") != "complete"
        or population.get("identity_disjoint_from_discovery") is not True
        or population.get("exact_content_disjoint_from_discovery") is not True
        or population.get("training_ready") is not False
        or aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("status") != "complete"
        or aggregate.get("population_receipt_sha256") != population_sha
        or aggregate.get("coverage_first_not_statistical_acceptance_estimate")
        is not True
        or aggregate.get("training_ready") is not False
        or screen.get("schema") != SCREEN_SCHEMA
        or screen.get("status") != "complete"
        or screen.get("population", {}).get("receipt_sha256") != population_sha
        or screen.get("benchmark_contamination_screen_complete") is not True
        or screen.get("training_ready") is not False
        or duplicates.get("schema") != DUPLICATE_SCHEMA
        or duplicates.get("status") != "complete"
        or duplicates.get("sample_exact_duplicate_audit_complete") is not True
        or duplicates.get("training_ready") is not False
        or len(confirmation_descriptors) != 1
    ):
        raise ConfirmationPromotionError("confirmation evidence binding differs")


def decide_sources(
    population: dict[str, Any],
    aggregate: dict[str, Any],
    screen: dict[str, Any],
    duplicates: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return deterministic pilot decisions without granting corpus admission."""

    _validate_inputs(population, aggregate, screen, duplicates)
    summary = aggregate.get("summary")
    routes_by_source = (
        summary.get("by_source_conservative_triage")
        if isinstance(summary, dict)
        else None
    )
    contamination_by_source = screen.get("summary", {}).get("by_source")
    by_source = population.get("by_source")
    if (
        not isinstance(routes_by_source, dict)
        or not routes_by_source
        or not isinstance(contamination_by_source, dict)
        or set(routes_by_source) != set(contamination_by_source)
        or not isinstance(by_source, dict)
        or set(routes_by_source) != set(by_source)
    ):
        raise ConfirmationPromotionError("confirmation source coverage differs")
    global_duplicates_clean = (
        duplicates.get("flagged_pairs")
        == METHOD["maximum_exact_or_normalized_duplicate_pairs"]
        and duplicates.get("cross_population_flagged_pairs") == 0
    )
    decisions = []
    for source_id in sorted(routes_by_source):
        routes = routes_by_source[source_id]
        contamination = contamination_by_source[source_id]
        observed_rows = by_source[source_id]
        if (
            isinstance(observed_rows, bool)
            or not isinstance(observed_rows, int)
            or observed_rows <= 0
            or not isinstance(routes, dict)
            or sum(routes.values()) != observed_rows
            or not isinstance(contamination, dict)
            or contamination.get("rows") != observed_rows
        ):
            raise ConfirmationPromotionError("confirmation source row custody differs")
        representation_rows = routes.get("representation_verification", 0)
        representation_ppm = (representation_rows * 1_000_000) // observed_rows
        checks = {
            "minimum_rows": observed_rows
            >= METHOD["minimum_confirmation_rows_per_source"],
            "minimum_representation_verification": representation_ppm
            >= METHOD["minimum_representation_verification_ppm"],
            "zero_quarantine": routes.get("quarantine", 0)
            <= METHOD["maximum_quarantine_rows"],
            "zero_rights_hold": routes.get("rights_hold", 0)
            <= METHOD["maximum_rights_hold_rows"],
            "zero_benchmark_contamination": contamination.get(
                "contaminated_rows"
            )
            <= METHOD["maximum_benchmark_contaminated_rows"],
            "global_exact_duplicate_gate": global_duplicates_clean,
            "identity_disjoint": population["identity_disjoint_from_discovery"],
            "content_disjoint": population[
                "exact_content_disjoint_from_discovery"
            ],
        }
        authorized = all(checks.values())
        failed_checks = sorted(key for key, value in checks.items() if not value)
        decisions.append(
            {
                "source_id": source_id,
                "observed_rows": observed_rows,
                "representation_verification_rows": representation_rows,
                "representation_verification_ppm": representation_ppm,
                "quarantine_rows": routes.get("quarantine", 0),
                "rights_hold_rows": routes.get("rights_hold", 0),
                "benchmark_contaminated_rows": contamination.get(
                    "contaminated_rows"
                ),
                "checks": checks,
                "failed_checks": failed_checks,
                "bounded_streaming_source_pilot_authorized": authorized,
                "next_action": (
                    "build_bounded_streaming_source_pilot"
                    if authorized
                    else "hold_and_resolve_failed_confirmation_checks"
                ),
                "bulk_training_admission": False,
                "training_ready": False,
            }
        )
    return decisions


def build_decision(
    population_receipt_path: Path,
    aggregate_path: Path,
    screen_path: Path,
    duplicate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Seal a source-pilot decision from four independently replayable inputs."""

    if output_path.exists() or output_path.is_symlink():
        raise ConfirmationPromotionError("promotion output already exists")
    inputs = {
        "population": (
            population_receipt_path,
            _load_json(population_receipt_path, "population"),
        ),
        "aggregate": (aggregate_path, _load_json(aggregate_path, "aggregate")),
        "screen": (screen_path, _load_json(screen_path, "screen")),
        "duplicates": (duplicate_path, _load_json(duplicate_path, "duplicates")),
    }
    decisions = decide_sources(*(payload for _, payload in inputs.values()))
    selected = sorted(
        row["source_id"]
        for row in decisions
        if row["bounded_streaming_source_pilot_authorized"]
    )
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "method": METHOD,
        "method_sha256": canonical_sha256(METHOD),
        "inputs": {
            label: {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "receipt_sha256": artifact["receipt_sha256"],
            }
            for label, (path, artifact) in inputs.items()
        },
        "sources": decisions,
        "selected_source_ids": selected,
        "selected_sources": len(selected),
        "evaluated_sources": len(decisions),
        "bounded_streaming_source_pilots_authorized": bool(selected),
        "bulk_training_admission": False,
        "full_source_ingestion_authorized": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-receipt", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--duplicates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_decision(
        args.population_receipt,
        args.aggregate,
        args.screen,
        args.duplicates,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
