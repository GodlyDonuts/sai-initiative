"""Join exact PleIAs metadata geometry to calibrated audit work routes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.independent_review_compare import SCHEMA as CALIBRATION_SCHEMA
from sai.data.pleias_metadata_census import AGGREGATE_SCHEMA as CENSUS_SCHEMA
from sai.data.pleias_quality_strata import SCHEMA as QUALITY_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pleias-quality-core-policy-v1"
ROUTE_ORDER = (
    "priority_direct_representation_verification",
    "priority_cleanup_then_verification",
    "translation_value_review",
    "targeted_recovery_confirmation",
    "hold_nonenglish_for_translation_triage",
    "hold_high_blocking_signal",
    "hold_insufficient_audit_coverage",
)


class PleiasQualityCorePolicyError(RuntimeError):
    """The census, audit, calibration, or route accounting differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasQualityCorePolicyError("PleIAs policy input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasQualityCorePolicyError("PleIAs policy input is invalid") from error
    if not isinstance(payload, dict):
        raise PleiasQualityCorePolicyError("PleIAs policy input differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("training_ready") is not False
    ):
        raise PleiasQualityCorePolicyError("PleIAs policy receipt differs")
    return payload


def _count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PleiasQualityCorePolicyError(f"{label} differs")
    return value


def _ppm(numerator: int, denominator: int) -> int:
    return numerator * 1_000_000 // denominator if denominator else 0


def _route(
    language: str, screen_rows: int, routes: dict[str, int]
) -> tuple[str, dict[str, int]]:
    blocking = routes.get("quarantine", 0) + routes.get("rights_hold", 0)
    representation = routes.get("representation_verification", 0)
    cleanup = routes.get("cleanup_review", 0)
    quality = routes.get("quality_review", 0)
    grounding = routes.get("factual_grounding_review", 0)
    metrics = {
        "blocking_ppm": _ppm(blocking, screen_rows),
        "representation_verification_ppm": _ppm(representation, screen_rows),
        "cleanup_review_ppm": _ppm(cleanup, screen_rows),
        "quality_review_ppm": _ppm(quality, screen_rows),
        "factual_grounding_review_ppm": _ppm(grounding, screen_rows),
    }
    if screen_rows < 8:
        return "hold_insufficient_audit_coverage", metrics
    if metrics["blocking_ppm"] >= 500_000:
        return "hold_high_blocking_signal", metrics
    english = language.casefold() == "english"
    useful = representation + cleanup + quality + grounding
    if not english:
        if metrics["blocking_ppm"] <= 200_000 and _ppm(useful, screen_rows) >= 500_000:
            return "translation_value_review", metrics
        return "hold_nonenglish_for_translation_triage", metrics
    if (
        metrics["blocking_ppm"] <= 100_000
        and metrics["representation_verification_ppm"] >= 400_000
    ):
        return "priority_direct_representation_verification", metrics
    if (
        metrics["blocking_ppm"] <= 200_000
        and _ppm(representation + cleanup + quality, screen_rows) >= 600_000
    ):
        return "priority_cleanup_then_verification", metrics
    return "targeted_recovery_confirmation", metrics


def build_policy_payload(
    census: dict[str, Any],
    quality: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Build exact group routes without admitting or deleting any row."""

    if (
        census.get("schema") != CENSUS_SCHEMA
        or census.get("status")
        != "complete_nontraining_pleias_metadata_census"
        or census.get("training_ready") is not False
        or quality.get("schema") != QUALITY_SCHEMA
        or quality.get("status") != "complete_nontraining_quality_strata_report"
        or quality.get("training_ready") is not False
        or calibration.get("schema") != CALIBRATION_SCHEMA
        or calibration.get("status")
        != "complete_nontraining_independent_review_comparison"
        or calibration.get("training_ready") is not False
    ):
        raise PleiasQualityCorePolicyError("PleIAs policy evidence differs")
    census_groups = census.get("axes", {}).get("collection_language")
    quality_groups = quality.get("axes", {}).get("collection_language")
    if not isinstance(census_groups, dict) or not isinstance(quality_groups, list):
        raise PleiasQualityCorePolicyError("PleIAs group evidence differs")
    audits = {}
    for row in quality_groups:
        if not isinstance(row, dict) or not isinstance(row.get("value"), str):
            raise PleiasQualityCorePolicyError("PleIAs audit group differs")
        if row["value"] in audits:
            raise PleiasQualityCorePolicyError("PleIAs audit groups overlap")
        audits[row["value"]] = row
    groups = []
    route_totals: dict[str, Counter[str]] = {
        route: Counter() for route in ROUTE_ORDER
    }
    census_totals = Counter()
    for encoded, counts in census_groups.items():
        try:
            pair = json.loads(encoded)
        except (TypeError, json.JSONDecodeError) as error:
            raise PleiasQualityCorePolicyError(
                "PleIAs census group identity differs"
            ) from error
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(value, str) or not value for value in pair)
            or not isinstance(counts, dict)
        ):
            raise PleiasQualityCorePolicyError(
                "PleIAs census group identity differs"
            )
        collection, language = pair
        rows = _count(counts.get("rows"), "PleIAs group rows")
        words = _count(counts.get("word_count"), "PleIAs group words")
        tokens = _count(counts.get("token_count"), "PleIAs group tokens")
        files = _count(counts.get("files"), "PleIAs group files")
        audit = audits.get(f"{collection}::{language}")
        if audit is None:
            screen_rows = 0
            route_counts = {}
            active_risks = {}
        else:
            screen_rows = _count(audit.get("rows"), "PleIAs screen rows")
            raw_routes = audit.get("route_counts")
            raw_risks = audit.get("active_risk_counts")
            if not isinstance(raw_routes, dict) or not isinstance(raw_risks, dict):
                raise PleiasQualityCorePolicyError("PleIAs audit counts differ")
            route_counts = {
                key: _count(value, "PleIAs audit route")
                for key, value in raw_routes.items()
            }
            if sum(route_counts.values()) != screen_rows:
                raise PleiasQualityCorePolicyError("PleIAs audit coverage differs")
            active_risks = {
                key: _count(value, "PleIAs audit risk")
                for key, value in raw_risks.items()
            }
        work_route, metrics = _route(language, screen_rows, route_counts)
        group = {
            "collection": collection,
            "language": language,
            "metadata_rows": rows,
            "metadata_word_count": words,
            "metadata_token_count": tokens,
            "parent_files": files,
            "audit_rows": screen_rows,
            "audit_route_counts": dict(sorted(route_counts.items())),
            "audit_active_risk_counts": dict(sorted(active_risks.items())),
            "audit_metrics": metrics,
            "work_route": work_route,
            "automatic_exclusion": False,
            "automatic_training_admission": False,
        }
        group["row_sha256"] = canonical_sha256(group)
        groups.append(group)
        route_totals[work_route].update(
            {
                "groups": 1,
                "rows": rows,
                "word_count": words,
                "token_count": tokens,
            }
        )
        census_totals.update(
            {"rows": rows, "word_count": words, "token_count": tokens}
        )
    expected = census.get("totals")
    if not isinstance(expected, dict) or any(
        census_totals[field] != expected.get(field)
        for field in ("rows", "word_count", "token_count")
    ):
        raise PleiasQualityCorePolicyError("PleIAs policy totals differ")
    groups.sort(
        key=lambda row: (
            ROUTE_ORDER.index(row["work_route"]),
            -row["metadata_token_count"],
            row["collection"],
            row["language"],
        )
    )
    return {
        "schema": SCHEMA,
        "status": "complete_nontraining_pleias_quality_core_work_policy",
        "method": {
            "minimum_audit_rows": 8,
            "maximum_direct_blocking_ppm": 100_000,
            "minimum_direct_representation_ppm": 400_000,
            "maximum_cleanup_blocking_ppm": 200_000,
            "minimum_cleanup_useful_ppm": 600_000,
            "minimum_high_blocking_ppm": 500_000,
            "single_model_quarantine_is_automatic_exclusion": False,
            "route_is_training_admission": False,
        },
        "evidence": {
            "metadata_census_receipt_sha256": census.get("receipt_sha256"),
            "quality_strata_receipt_sha256": quality.get("receipt_sha256"),
            "independent_calibration_receipt_sha256": calibration.get(
                "receipt_sha256"
            ),
        },
        "totals": dict(sorted(census_totals.items())),
        "route_totals": {
            route: dict(sorted(route_totals[route].items()))
            for route in ROUTE_ORDER
        },
        "groups": groups,
        "ordered_group_rows_sha256": canonical_sha256(
            [row["row_sha256"] for row in groups]
        ),
        "source_text_persisted": False,
        "automatic_exclusion": False,
        "automatic_training_admission": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }


def build_policy(
    census_path: Path,
    quality_path: Path,
    calibration_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Load exact signed evidence and atomically write the work policy."""

    if output.exists() or output.is_symlink():
        raise PleiasQualityCorePolicyError("PleIAs policy output exists")
    census = _load_signed(census_path, CENSUS_SCHEMA)
    quality = _load_signed(quality_path, QUALITY_SCHEMA)
    calibration = _load_signed(calibration_path, CALIBRATION_SCHEMA)
    payload = build_policy_payload(census, quality, calibration)
    payload["input_file_sha256s"] = {
        "metadata_census": sha256_file(census_path),
        "quality_strata": sha256_file(quality_path),
        "independent_calibration": sha256_file(calibration_path),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-census", type=Path, required=True)
    parser.add_argument("--quality-strata", type=Path, required=True)
    parser.add_argument("--independent-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_policy(
        args.metadata_census,
        args.quality_strata,
        args.independent_calibration,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "totals": result["totals"],
                "route_totals": result["route_totals"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
