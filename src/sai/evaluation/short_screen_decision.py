"""Turn the exact three-family 100M comparison into a predeclared decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from sai.evaluation.short_screen_compare import BENCHMARKS, FAMILIES, SCHEMA

DECISION_SCHEMA = "sai-100m-short-screen-decision-v1"
RECURRENT_FAMILIES = ("gdn_hybrid", "kda_mla_hybrid")
NONINFERIORITY_MARGIN_PERCENTAGE_POINTS = -1.0


class ShortScreenDecisionError(RuntimeError):
    """The comparison receipt or predeclared decision arithmetic differs."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _load_comparison(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ShortScreenDecisionError("comparison artifact is missing or unsafe")
    encoded = path.read_bytes()
    try:
        comparison = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShortScreenDecisionError("comparison artifact is unreadable") from error
    if not isinstance(comparison, dict):
        raise ShortScreenDecisionError("comparison payload differs")
    unsigned = dict(comparison)
    receipt = unsigned.pop("receipt_sha256", None)
    benchmarks = comparison.get("benchmarks")
    if (
        comparison.get("schema") != SCHEMA
        or comparison.get("status") != "complete"
        or comparison.get("development_only") is not True
        or comparison.get("iso_data_comparison") is not True
        or comparison.get("iso_flop_comparison") is not False
        or comparison.get("scientific_promotion_allowed") is not False
        or comparison.get("four_b_training_authorized") is not False
        or receipt != _canonical_sha256(unsigned)
        or not isinstance(benchmarks, dict)
        or set(benchmarks) != set(BENCHMARKS)
    ):
        raise ShortScreenDecisionError("comparison receipt differs")
    return comparison, hashlib.sha256(encoded).hexdigest()


def _candidate_minus_reference_interval(
    benchmark: dict[str, Any], candidate: str
) -> dict[str, float]:
    key = f"gated_gqa_minus_{candidate}"
    pair = benchmark.get("pairwise", {}).get(key, {}).get("paired_interval")
    if not isinstance(pair, dict) or pair.get("method") != "paired_normal_95ci":
        raise ShortScreenDecisionError("paired reference interval differs")
    values = (
        pair.get("delta_percentage_points"),
        pair.get("lower_percentage_points"),
        pair.get("upper_percentage_points"),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in values
    ):
        raise ShortScreenDecisionError("paired reference interval differs")
    delta, lower, upper = (float(value) for value in values)
    return {
        "delta_percentage_points": -delta,
        "lower_percentage_points": -upper,
        "upper_percentage_points": -lower,
    }


def decide(comparison: dict[str, Any], comparison_file_sha256: str) -> dict[str, Any]:
    """Apply the frozen capability floor and recurrent-vs-reference rule."""

    if (
        not isinstance(comparison_file_sha256, str)
        or len(comparison_file_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in comparison_file_sha256
        )
    ):
        raise ShortScreenDecisionError("comparison file identity differs")
    unsigned = dict(comparison)
    comparison_receipt = unsigned.pop("receipt_sha256", None)
    if comparison_receipt != _canonical_sha256(unsigned):
        raise ShortScreenDecisionError("comparison receipt differs")

    family_rows: dict[str, Any] = {}
    for family in FAMILIES:
        benchmark_rows = {}
        for benchmark_name in BENCHMARKS:
            benchmark = comparison["benchmarks"].get(benchmark_name)
            if not isinstance(benchmark, dict):
                raise ShortScreenDecisionError("benchmark comparison differs")
            baseline = benchmark.get("uniform_choice_baseline_accuracy")
            family_result = benchmark.get("families", {}).get(family)
            if (
                isinstance(baseline, bool)
                or not isinstance(baseline, (int, float))
                or not 0.0 < baseline < 1.0
                or not isinstance(family_result, dict)
                or isinstance(family_result.get("accuracy"), bool)
                or not isinstance(family_result.get("accuracy"), (int, float))
            ):
                raise ShortScreenDecisionError("benchmark family accuracy differs")
            accuracy = float(family_result["accuracy"])
            if not 0.0 <= accuracy <= 1.0:
                raise ShortScreenDecisionError("benchmark family accuracy differs")
            benchmark_rows[benchmark_name] = {
                "accuracy": accuracy,
                "uniform_choice_baseline_accuracy": float(baseline),
                "margin_over_uniform_percentage_points": 100.0
                * (accuracy - float(baseline)),
                "strictly_above_uniform": accuracy > float(baseline),
            }
        capability_floor = all(
            row["strictly_above_uniform"] for row in benchmark_rows.values()
        )
        family_rows[family] = {
            "benchmarks": benchmark_rows,
            "unweighted_macro_accuracy": sum(
                row["accuracy"] for row in benchmark_rows.values()
            )
            / len(BENCHMARKS),
            "capability_floor_pass": capability_floor,
            "role": (
                "conventional_reference"
                if family == "gated_gqa"
                else "recurrent_candidate"
            ),
        }

    eligible: list[str] = []
    for family in RECURRENT_FAMILIES:
        paired = {
            benchmark_name: _candidate_minus_reference_interval(
                comparison["benchmarks"][benchmark_name], family
            )
            for benchmark_name in BENCHMARKS
        }
        no_serious_regression = all(
            row["lower_percentage_points"] >= NONINFERIORITY_MARGIN_PERCENTAGE_POINTS
            for row in paired.values()
        )
        positive_lcb = any(
            row["lower_percentage_points"] > 0.0 for row in paired.values()
        )
        candidate_eligible = bool(
            family_rows[family]["capability_floor_pass"]
            and no_serious_regression
            and positive_lcb
        )
        family_rows[family].update(
            {
                "paired_candidate_minus_gated_gqa": paired,
                "paired_noninferior_every_benchmark": no_serious_regression,
                "positive_paired_lcb_any_benchmark": positive_lcb,
                "extended_screen_candidate_eligible": candidate_eligible,
            }
        )
        if candidate_eligible:
            eligible.append(family)
    family_rows["gated_gqa"]["extended_screen_candidate_eligible"] = False

    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda family: (
                min(
                    row["lower_percentage_points"]
                    for row in family_rows[family][
                        "paired_candidate_minus_gated_gqa"
                    ].values()
                ),
                family_rows[family]["unweighted_macro_accuracy"],
                -RECURRENT_FAMILIES.index(family),
            ),
        )
        action = "recurrent_candidate_selected_for_extended_screen"
    elif family_rows["gated_gqa"]["capability_floor_pass"]:
        action = "conventional_reference_retained_no_recurrent_win"
    else:
        action = "no_family_capability_qualified_data_extension_only"

    result = {
        "schema": DECISION_SCHEMA,
        "status": "complete",
        "development_only": True,
        "comparison_file_sha256": comparison_file_sha256,
        "comparison_receipt_sha256": comparison_receipt,
        "rules": {
            "capability_floor": "above_exact_uniform_every_benchmark",
            "noninferiority_margin_percentage_points": (
                NONINFERIORITY_MARGIN_PERCENTAGE_POINTS
            ),
            "positive_paired_lcb_any_benchmark_required": True,
            "paired_interval_method": "paired_normal_95ci",
        },
        "families": family_rows,
        "eligible_recurrent_candidates": eligible,
        "selected_recurrent_candidate": selected,
        "action": action,
        "next_longer_screen_is_data_starvation_diagnostic": selected is None,
        "scientific_promotion_allowed": False,
        "four_b_training_authorized": False,
    }
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def write_decision(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise ShortScreenDecisionError("decision output path differs")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison, file_sha256 = _load_comparison(args.comparison)
    write_decision(args.output, decide(comparison, file_sha256))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
