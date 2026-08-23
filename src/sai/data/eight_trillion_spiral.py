"""Freeze Sai's moving-center-of-gravity schedule for an eight-trillion-token run."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _exact
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-eight-trillion-token-spiral-policy-v1"
TOTAL_TOKENS = 8_000_000_000_000
STAGES = ("foundation", "expansion", "depth", "synthesis", "annealing")
BANDS = ("foundational", "intermediate", "advanced", "expert")
BOUNDARIES = {
    "foundation": (0, 2_000_000_000_000),
    "expansion": (2_000_000_000_000, 4_800_000_000_000),
    "depth": (4_800_000_000_000, 6_800_000_000_000),
    "synthesis": (6_800_000_000_000, 7_600_000_000_000),
    "annealing": (7_600_000_000_000, 8_000_000_000_000),
}
BAND_SHARES_PPM = {
    "foundation": {
        "foundational": 550_000,
        "intermediate": 300_000,
        "advanced": 120_000,
        "expert": 30_000,
    },
    "expansion": {
        "foundational": 250_000,
        "intermediate": 450_000,
        "advanced": 230_000,
        "expert": 70_000,
    },
    "depth": {
        "foundational": 120_000,
        "intermediate": 280_000,
        "advanced": 420_000,
        "expert": 180_000,
    },
    "synthesis": {
        "foundational": 100_000,
        "intermediate": 200_000,
        "advanced": 350_000,
        "expert": 350_000,
    },
    "annealing": {
        "foundational": 100_000,
        "intermediate": 180_000,
        "advanced": 300_000,
        "expert": 420_000,
    },
}
CROSS_DOMAIN_MINIMUM_PPM = {
    "foundation": 10_000,
    "expansion": 40_000,
    "depth": 100_000,
    "synthesis": 300_000,
    "annealing": 200_000,
}


class EightTrillionSpiralError(RuntimeError):
    """The token boundary, moving mixture, or synthesis gate differs."""


def _allocate(total: int, shares: dict[str, int]) -> dict[str, int]:
    exact = {band: total * shares[band] for band in BANDS}
    result = {band: exact[band] // 1_000_000 for band in BANDS}
    remainder = total - sum(result.values())
    order = sorted(
        BANDS,
        key=lambda band: (-(exact[band] % 1_000_000), BANDS.index(band)),
    )
    for band in order[:remainder]:
        result[band] += 1
    return result


def build_policy() -> dict[str, Any]:
    """Create the prospective token schedule and synthetic-bridge boundary."""

    stage_tokens = {
        stage: BOUNDARIES[stage][1] - BOUNDARIES[stage][0] for stage in STAGES
    }
    payload = {
        "schema": SCHEMA,
        "status": "prospective",
        "total_tokens": TOTAL_TOKENS,
        "stage_order": list(STAGES),
        "bands": list(BANDS),
        "boundaries": {
            stage: {"start_inclusive": start, "end_exclusive": end}
            for stage, (start, end) in BOUNDARIES.items()
        },
        "stage_tokens": stage_tokens,
        "band_shares_ppm": BAND_SHARES_PPM,
        "stage_band_tokens": {
            stage: _allocate(stage_tokens[stage], BAND_SHARES_PPM[stage])
            for stage in STAGES
        },
        "cross_domain_minimum_ppm": CROSS_DOMAIN_MINIMUM_PPM,
        "moving_center_of_gravity": True,
        "foundations_present_in_every_stage": True,
        "expert_material_present_from_first_stage": True,
        "synthetic_bridge_policy": {
            "minimum_distinct_domains": 2,
            "source_anchors_required": True,
            "source_identity_hashes_required": True,
            "prerequisites_grounded_before_or_rehearsed_with_bridge": True,
            "novel_relationship_required": True,
            "independent_solution_or_deterministic_verification_required": True,
            "benchmark_derived_prompts_forbidden": True,
            "translation_lineage_required": True,
            "generic_ungrounded_generation_forbidden": True,
        },
        "annealing_selection_basis": (
            "source_disjoint_evaluation_marginal_gain_under_retention_vetoes"
        ),
        "fixed_origin_percentages_used": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return validate_policy(payload)


def validate_policy(payload: Any) -> dict[str, Any]:
    row = _exact(
        payload,
        {
            "schema",
            "status",
            "total_tokens",
            "stage_order",
            "bands",
            "boundaries",
            "stage_tokens",
            "band_shares_ppm",
            "stage_band_tokens",
            "cross_domain_minimum_ppm",
            "moving_center_of_gravity",
            "foundations_present_in_every_stage",
            "expert_material_present_from_first_stage",
            "synthetic_bridge_policy",
            "annealing_selection_basis",
            "fixed_origin_percentages_used",
            "training_authorized",
            "four_b_training_authorized",
            "receipt_sha256",
        },
        "eight-trillion spiral policy",
    )
    if (
        row["schema"] != SCHEMA
        or row["status"] != "prospective"
        or row["total_tokens"] != TOTAL_TOKENS
        or row["stage_order"] != list(STAGES)
        or row["bands"] != list(BANDS)
        or row["band_shares_ppm"] != BAND_SHARES_PPM
        or row["cross_domain_minimum_ppm"] != CROSS_DOMAIN_MINIMUM_PPM
        or row["moving_center_of_gravity"] is not True
        or row["foundations_present_in_every_stage"] is not True
        or row["expert_material_present_from_first_stage"] is not True
        or row["fixed_origin_percentages_used"] is not False
        or row["training_authorized"] is not False
        or row["four_b_training_authorized"] is not False
    ):
        raise EightTrillionSpiralError("eight-trillion spiral contract differs")
    boundaries = _exact(row["boundaries"], set(STAGES), "stage boundaries")
    stage_tokens = _exact(row["stage_tokens"], set(STAGES), "stage tokens")
    allocations = _exact(row["stage_band_tokens"], set(STAGES), "stage band tokens")
    cursor = 0
    for stage in STAGES:
        boundary = _exact(
            boundaries[stage], {"start_inclusive", "end_exclusive"}, stage
        )
        if (
            boundary["start_inclusive"] != cursor
            or (boundary["start_inclusive"], boundary["end_exclusive"])
            != BOUNDARIES[stage]
        ):
            raise EightTrillionSpiralError("stage boundaries differ")
        cursor = boundary["end_exclusive"]
        expected_tokens = cursor - boundary["start_inclusive"]
        if stage_tokens[stage] != expected_tokens:
            raise EightTrillionSpiralError("stage token budget differs")
        if sum(BAND_SHARES_PPM[stage].values()) != 1_000_000:
            raise EightTrillionSpiralError("band shares do not sum to one")
        expected = _allocate(expected_tokens, BAND_SHARES_PPM[stage])
        if allocations[stage] != expected:
            raise EightTrillionSpiralError("stage band allocation differs")
        if expected["foundational"] <= 0 or expected["expert"] <= 0:
            raise EightTrillionSpiralError("spiral endpoints disappeared")
    if cursor != TOTAL_TOKENS or sum(stage_tokens.values()) != TOTAL_TOKENS:
        raise EightTrillionSpiralError("total token budget differs")
    bridge = _exact(
        row["synthetic_bridge_policy"],
        {
            "minimum_distinct_domains",
            "source_anchors_required",
            "source_identity_hashes_required",
            "prerequisites_grounded_before_or_rehearsed_with_bridge",
            "novel_relationship_required",
            "independent_solution_or_deterministic_verification_required",
            "benchmark_derived_prompts_forbidden",
            "translation_lineage_required",
            "generic_ungrounded_generation_forbidden",
        },
        "synthetic bridge policy",
    )
    if bridge["minimum_distinct_domains"] < 2 or any(
        value is not True
        for key, value in bridge.items()
        if key != "minimum_distinct_domains"
    ):
        raise EightTrillionSpiralError("synthetic bridge policy differs")
    unsigned = {key: value for key, value in row.items() if key != "receipt_sha256"}
    if row["receipt_sha256"] != canonical_sha256(unsigned):
        raise EightTrillionSpiralError("spiral receipt hash differs")
    return row


def _atomic_create(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise EightTrillionSpiralError("spiral output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial.{uuid.uuid4().hex}"
    try:
        with temporary.open("x") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    policy = build_policy()
    _atomic_create(args.output, policy)
    print(json.dumps(policy, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
