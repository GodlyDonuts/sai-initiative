"""Summarize a complete PleIAs audit by source metadata without source text."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.data_compiler_labeling import RISK_KEYS
from sai.data.reservoir_audit_aggregate import (
    SCHEMA as AGGREGATE_SCHEMA,
)
from sai.data.reservoir_audit_aggregate import (
    _triage_route,
    _validate_compiler_receipt,
    load_population,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pleias-quality-strata-report-v1"
AXES = ("collection", "language", "open_type", "collection_language")


class PleiasQualityStrataError(RuntimeError):
    """The complete audit or source-safe grouping contract differs."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasQualityStrataError("quality-strata input is missing or unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasQualityStrataError("quality-strata input is invalid") from error
    if not isinstance(value, dict):
        raise PleiasQualityStrataError("quality-strata input is invalid")
    return value


def _ppm(numerator: int, denominator: int) -> int:
    return numerator * 1_000_000 // denominator if denominator else 0


def _priority(rows: int, routes: Counter[str]) -> str:
    quarantine_or_rights = routes["quarantine"] + routes["rights_hold"]
    representation = routes["representation_verification"]
    if rows < 8:
        return "insufficient_screen_coverage"
    if (
        _ppm(quarantine_or_rights, rows) <= 50_000
        and _ppm(representation, rows) >= 600_000
    ):
        return "priority_targeted_materialization_screen"
    if (
        _ppm(quarantine_or_rights, rows) <= 150_000
        and _ppm(representation, rows) >= 400_000
    ):
        return "secondary_targeted_materialization_screen"
    return "hold_for_row_level_recovery"


def summarize(
    lineage: list[dict[str, Any]], judgments: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Return source-safe collection and language summaries."""

    if not lineage or len(lineage) != len(judgments):
        raise PleiasQualityStrataError("quality-strata row coverage differs")
    grouped: dict[str, dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]] = {
        axis: defaultdict(list) for axis in AXES
    }
    for source, judgment in zip(lineage, judgments, strict=True):
        locator = source.get("locator")
        if not isinstance(locator, dict):
            raise PleiasQualityStrataError("PleIAs locator differs")
        values = {}
        for key in ("collection", "language", "open_type"):
            value = locator.get(key)
            if value is None:
                values[key] = "__unknown__"
            elif isinstance(value, str) and value:
                values[key] = value
            else:
                raise PleiasQualityStrataError("PleIAs stratum metadata differs")
        values["collection_language"] = values["collection"] + "::" + values["language"]
        for axis in AXES:
            grouped[axis][values[axis]].append((source, judgment))

    result = {}
    for axis in AXES:
        rows = []
        for value, members in sorted(grouped[axis].items()):
            verdicts: Counter[str] = Counter()
            routes: Counter[str] = Counter()
            risks: Counter[str] = Counter()
            full_text_bytes = 0
            for source, judgment in members:
                size = source.get("full_text_bytes")
                if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                    raise PleiasQualityStrataError("PleIAs full-text bytes differ")
                full_text_bytes += size
                verdicts[judgment["verdict"]] += 1
                routes[_triage_route(judgment)] += 1
                for risk in RISK_KEYS:
                    if judgment["risks"][risk]:
                        risks[risk] += 1
            count = len(members)
            rows.append(
                {
                    "value": value,
                    "rows": count,
                    "full_text_bytes": full_text_bytes,
                    "verdict_counts": dict(sorted(verdicts.items())),
                    "route_counts": dict(sorted(routes.items())),
                    "route_ppm": {
                        key: _ppm(routes[key], count) for key in sorted(routes)
                    },
                    "active_risk_counts": dict(sorted(risks.items())),
                    "screen_priority": _priority(count, routes),
                    "bulk_training_admission": False,
                }
            )
        result[axis] = rows
    return result


def build_report(
    population_root: Path,
    judgments_root: Path,
    aggregate_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Build a tamper-evident report from a complete PleIAs audit."""

    if output.exists() or output.is_symlink():
        raise PleiasQualityStrataError("quality-strata output already exists")
    candidates, lineage, population = load_population(population_root)
    aggregate = _load_json(aggregate_path)
    unsigned_aggregate = {
        key: value for key, value in aggregate.items() if key != "receipt_sha256"
    }
    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("status") != "complete"
        or aggregate.get("receipt_sha256") != canonical_sha256(unsigned_aggregate)
        or aggregate.get("training_ready") is not False
        or aggregate.get("summary", {}).get("model_judgments_are_verified_admissions")
        is not False
        or aggregate.get("population_receipt_sha256") != population["receipt_sha256"]
        or aggregate.get("summary", {}).get("rows") != len(candidates)
    ):
        raise PleiasQualityStrataError("PleIAs aggregate custody differs")
    expected = {
        judgments_root / f"{row['candidate_identity_sha256']}.compiler.json"
        for row in candidates
    }
    if set(judgments_root.glob("*.compiler.json")) != expected:
        raise PleiasQualityStrataError("PleIAs judgment population differs")
    judgments = []
    receipt_hashes = []
    for candidate in candidates:
        identity = candidate["candidate_identity_sha256"]
        try:
            receipt = _validate_compiler_receipt(
                _load_json(judgments_root / f"{identity}.compiler.json"), candidate
            )
        except RuntimeError as error:
            raise PleiasQualityStrataError("PleIAs judgment custody differs") from error
        judgments.append(receipt["judgment"])
        receipt_hashes.append(receipt["receipt_sha256"])
    if aggregate.get("ordered_compiler_receipts_sha256") != canonical_sha256(
        receipt_hashes
    ):
        raise PleiasQualityStrataError("PleIAs judgment order differs")

    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_quality_strata_report",
        "population": {
            "root_name": population_root.name,
            "rows": len(candidates),
            "receipt_sha256": population["receipt_sha256"],
        },
        "aggregate": {
            "path": aggregate_path.name,
            "sha256": sha256_file(aggregate_path),
            "receipt_sha256": aggregate["receipt_sha256"],
        },
        "axes": summarize(lineage, judgments),
        "source_text_persisted": False,
        "screen_priorities_are_training_admissions": False,
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
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(
        args.population_root, args.judgments_root, args.aggregate, args.output
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
