import copy

import pytest

from sai.data.eight_trillion_spiral import (
    BAND_SHARES_PPM,
    BOUNDARIES,
    TOTAL_TOKENS,
    EightTrillionSpiralError,
    build_policy,
    validate_policy,
)
from sai.data.token_stream import canonical_sha256


def _resign(payload: dict) -> None:
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )


def test_exact_eight_trillion_boundaries_and_moving_center() -> None:
    policy = build_policy()
    assert policy["total_tokens"] == TOTAL_TOKENS == 8_000_000_000_000
    assert policy["boundaries"]["synthesis"] == {
        "start_inclusive": 6_800_000_000_000,
        "end_exclusive": 7_600_000_000_000,
    }
    assert policy["stage_band_tokens"]["foundation"]["expert"] > 0
    assert policy["stage_band_tokens"]["annealing"]["foundational"] > 0
    assert (
        BAND_SHARES_PPM["foundation"]["foundational"]
        > BAND_SHARES_PPM["annealing"]["foundational"]
    )
    assert (
        BAND_SHARES_PPM["foundation"]["expert"] < BAND_SHARES_PPM["annealing"]["expert"]
    )


def test_spiral_rejects_boundary_gap_and_missing_late_foundations() -> None:
    policy = copy.deepcopy(build_policy())
    policy["boundaries"]["depth"]["start_inclusive"] += 1
    _resign(policy)
    with pytest.raises(EightTrillionSpiralError, match="boundaries"):
        validate_policy(policy)
    policy = copy.deepcopy(build_policy())
    policy["band_shares_ppm"]["annealing"]["foundational"] = 0
    policy["band_shares_ppm"]["annealing"]["expert"] += 100_000
    _resign(policy)
    with pytest.raises(EightTrillionSpiralError, match="contract"):
        validate_policy(policy)


def test_spiral_rejects_ungrounded_synthetic_bridges() -> None:
    policy = copy.deepcopy(build_policy())
    policy["synthetic_bridge_policy"]["source_anchors_required"] = False
    _resign(policy)
    with pytest.raises(EightTrillionSpiralError, match="synthetic bridge"):
        validate_policy(policy)


def test_boundary_constants_are_contiguous() -> None:
    assert list(BOUNDARIES.values())[0][0] == 0
    values = list(BOUNDARIES.values())
    assert all(
        left[1] == right[0] for left, right in zip(values[:-1], values[1:], strict=True)
    )
