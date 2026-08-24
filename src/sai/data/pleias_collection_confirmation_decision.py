"""Derive conservative PleIAs collection work routes from independent reviews."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.independent_review_compare import SCHEMA as COMPARISON_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pleias-collection-confirmation-decision-v1"
REQUIRED_PROVIDER = "gemini-3.5-flash-lite"


class PleiasCollectionConfirmationDecisionError(RuntimeError):
    """The independent comparison or collection accounting differs."""


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasCollectionConfirmationDecisionError(
            "collection comparison is unsafe"
        )
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasCollectionConfirmationDecisionError(
            "collection comparison is invalid"
        ) from error
    if not isinstance(value, dict):
        raise PleiasCollectionConfirmationDecisionError(
            "collection comparison differs"
        )
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        value.get("schema") != COMPARISON_SCHEMA
        or value.get("status")
        != "complete_nontraining_independent_review_comparison"
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("training_ready") is not False
    ):
        raise PleiasCollectionConfirmationDecisionError(
            "collection comparison receipt differs"
        )
    return value


def _ppm(value: int, rows: int) -> int:
    return value * 1_000_000 // rows


def _decision(
    rows: int, primary: Counter[str], independent: Counter[str]
) -> tuple[str, dict[str, dict[str, int]]]:
    metrics = {}
    for name, routes in (("primary", primary), ("independent", independent)):
        blocking = routes["quarantine"] + routes["rights_hold"]
        recoverable = (
            routes["representation_verification"]
            + routes["cleanup_review"]
            + routes["quality_review"]
        )
        metrics[name] = {
            "blocking_ppm": _ppm(blocking, rows),
            "recoverable_ppm": _ppm(recoverable, rows),
            "translation_ppm": _ppm(routes["translation_review"], rows),
        }
    if all(values["blocking_ppm"] >= 500_000 for values in metrics.values()):
        return "hold_high_blocking_confirmation", metrics
    if all(
        values["blocking_ppm"] <= 250_000
        and values["recoverable_ppm"] >= 625_000
        for values in metrics.values()
    ):
        return "priority_targeted_verification", metrics
    if any(values["translation_ppm"] >= 500_000 for values in metrics.values()):
        return "translation_or_grounding_adjudication", metrics
    return "targeted_recovery_confirmation", metrics


def build_payload(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine disjoint collection screens into exact conservative decisions."""

    if not comparisons:
        raise PleiasCollectionConfirmationDecisionError("comparisons are empty")
    grouped: dict[str, dict[str, Any]] = {}
    comparison_receipts = []
    for comparison in comparisons:
        comparison_receipt = comparison["receipt_sha256"]
        if comparison_receipt in comparison_receipts:
            raise PleiasCollectionConfirmationDecisionError(
                "comparison receipt is duplicated"
            )
        comparison_receipts.append(comparison_receipt)
        provider = comparison.get("by_provider", {}).get(REQUIRED_PROVIDER)
        rows = comparison.get("rows")
        if not isinstance(provider, dict) or not isinstance(rows, list):
            raise PleiasCollectionConfirmationDecisionError(
                "required independent provider differs"
            )
        for row in rows:
            stratum = row.get("stratum")
            review = row.get("reviews", {}).get(REQUIRED_PROVIDER)
            primary = row.get("primary")
            if (
                not isinstance(stratum, str)
                or not stratum.startswith("collection::")
                or not isinstance(primary, dict)
                or not isinstance(review, dict)
            ):
                raise PleiasCollectionConfirmationDecisionError(
                    "collection confirmation row differs"
                )
            collection = stratum.split("::", 1)[1]
            cell = grouped.setdefault(
                collection,
                {
                    "rows": 0,
                    "primary_routes": Counter(),
                    "independent_routes": Counter(),
                    "auxiliary_provider_coverage": Counter(),
                },
            )
            cell["rows"] += 1
            cell["primary_routes"][primary["route"]] += 1
            cell["independent_routes"][review["route"]] += 1
            for name, candidate in row.get("reviews", {}).items():
                if name != REQUIRED_PROVIDER and candidate is not None:
                    cell["auxiliary_provider_coverage"][name] += 1
    decisions = []
    route_totals = Counter()
    for collection, cell in sorted(grouped.items()):
        rows = cell["rows"]
        if rows != 8 or sum(cell["independent_routes"].values()) != rows:
            raise PleiasCollectionConfirmationDecisionError(
                "collection confirmation coverage differs"
            )
        work_route, metrics = _decision(
            rows, cell["primary_routes"], cell["independent_routes"]
        )
        route_totals[work_route] += 1
        decision = {
            "collection": collection,
            "rows": rows,
            "primary_route_counts": dict(sorted(cell["primary_routes"].items())),
            "independent_provider": REQUIRED_PROVIDER,
            "independent_route_counts": dict(
                sorted(cell["independent_routes"].items())
            ),
            "auxiliary_provider_coverage": dict(
                sorted(cell["auxiliary_provider_coverage"].items())
            ),
            "metrics": metrics,
            "work_route": work_route,
            "automatic_exclusion": False,
            "automatic_training_admission": False,
        }
        decision["row_sha256"] = canonical_sha256(decision)
        decisions.append(decision)
    return {
        "schema": SCHEMA,
        "status": "complete_nontraining_collection_confirmation_decision",
        "method": {
            "required_full_coverage_independent_provider": REQUIRED_PROVIDER,
            "exact_rows_per_collection": 8,
            "high_blocking_ppm": 500_000,
            "maximum_priority_blocking_ppm": 250_000,
            "minimum_priority_recoverable_ppm": 625_000,
            "translation_adjudication_ppm": 500_000,
            "cleanup_is_direct_training_admission": False,
        },
        "comparison_receipt_sha256s": comparison_receipts,
        "collection_count": len(decisions),
        "route_counts": dict(sorted(route_totals.items())),
        "decisions": decisions,
        "ordered_decisions_sha256": canonical_sha256(
            [row["row_sha256"] for row in decisions]
        ),
        "source_text_persisted": False,
        "automatic_exclusion": False,
        "automatic_training_admission": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }


def build_decision(paths: list[Path], output: Path) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise PleiasCollectionConfirmationDecisionError("decision output exists")
    comparisons = [_load(path) for path in paths]
    payload = build_payload(comparisons)
    payload["comparison_file_sha256s"] = [sha256_file(path) for path in paths]
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_decision(args.comparison, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
