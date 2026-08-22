from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.evaluation.short_screen_decision import (
    ShortScreenDecisionError,
    decide,
    write_decision,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _comparison(
    *,
    gqa: tuple[float, float] = (0.13, 0.40),
    gdn: tuple[float, float] = (0.14, 0.41),
    kda: tuple[float, float] = (0.12, 0.39),
    gqa_minus_gdn: tuple[float, float, float] = (-1.0, -1.8, -0.2),
    gqa_minus_kda: tuple[float, float, float] = (1.0, 0.2, 1.8),
) -> dict[str, object]:
    families = {"gated_gqa": gqa, "gdn_hybrid": gdn, "kda_mla_hybrid": kda}
    benchmarks = {}
    for index, benchmark in enumerate(("mmlu_pro", "musr")):
        benchmarks[benchmark] = {
            "uniform_choice_baseline_accuracy": (0.11, 0.37)[index],
            "families": {
                family: {"accuracy": scores[index]}
                for family, scores in families.items()
            },
            "pairwise": {
                "gated_gqa_minus_gdn_hybrid": {
                    "paired_interval": {
                        "method": "paired_normal_95ci",
                        "delta_percentage_points": gqa_minus_gdn[0],
                        "lower_percentage_points": gqa_minus_gdn[1],
                        "upper_percentage_points": gqa_minus_gdn[2],
                    }
                },
                "gated_gqa_minus_kda_mla_hybrid": {
                    "paired_interval": {
                        "method": "paired_normal_95ci",
                        "delta_percentage_points": gqa_minus_kda[0],
                        "lower_percentage_points": gqa_minus_kda[1],
                        "upper_percentage_points": gqa_minus_kda[2],
                    }
                },
            },
        }
    result: dict[str, object] = {
        "schema": "sai-short-screen-family-comparison-v1",
        "status": "complete",
        "development_only": True,
        "iso_data_comparison": True,
        "iso_flop_comparison": False,
        "scientific_promotion_allowed": False,
        "four_b_training_authorized": False,
        "benchmarks": benchmarks,
    }
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def test_selects_only_candidate_above_floor_with_positive_paired_lcb() -> None:
    comparison = _comparison()
    result = decide(comparison, "a" * 64)
    assert result["selected_recurrent_candidate"] == "gdn_hybrid"
    assert result["eligible_recurrent_candidates"] == ["gdn_hybrid"]
    assert result["action"] == "recurrent_candidate_selected_for_extended_screen"
    assert not result["next_longer_screen_is_data_starvation_diagnostic"]
    assert not result["scientific_promotion_allowed"]
    unsigned = dict(result)
    receipt = unsigned.pop("receipt_sha256")
    assert receipt == _canonical_sha256(unsigned)


def test_rejects_all_when_capability_floor_fails() -> None:
    comparison = _comparison(
        gqa=(0.10, 0.36),
        gdn=(0.10, 0.36),
        kda=(0.10, 0.36),
    )
    result = decide(comparison, "b" * 64)
    assert result["selected_recurrent_candidate"] is None
    assert result["eligible_recurrent_candidates"] == []
    assert result["action"] == "no_family_capability_qualified_data_extension_only"
    assert result["next_longer_screen_is_data_starvation_diagnostic"]


def test_retains_reference_when_it_alone_clears_floor() -> None:
    comparison = _comparison(gdn=(0.10, 0.36), kda=(0.10, 0.36))
    result = decide(comparison, "c" * 64)
    assert result["selected_recurrent_candidate"] is None
    assert result["action"] == "conventional_reference_retained_no_recurrent_win"


def test_tamper_and_no_overwrite_fail_closed(tmp_path: Path) -> None:
    comparison = _comparison()
    comparison["receipt_sha256"] = "0" * 64
    with pytest.raises(ShortScreenDecisionError, match="receipt"):
        decide(comparison, "d" * 64)

    valid = _comparison()
    payload = decide(valid, "e" * 64)
    output = tmp_path / "decision.json"
    write_decision(output, payload)
    assert json.loads(output.read_text()) == payload
    with pytest.raises(ShortScreenDecisionError, match="output path"):
        write_decision(output, payload)
