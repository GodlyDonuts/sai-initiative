import copy

import pytest

from sai.data.spiral_curriculum import (
    SHARES_PPM,
    SpiralCurriculumError,
    build_policy,
    validate_policy,
)
from sai.data.token_stream import canonical_sha256


def _budgets() -> dict[str, int]:
    return {
        "early": 10_000,
        "foundation": 10_000,
        "growth": 10_000,
        "advanced": 10_000,
        "annealing": 10_000,
    }


def _resign(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    return payload


def test_spiral_policy_preserves_fundamentals_and_three_complexity_axes() -> None:
    policy = build_policy(_budgets())
    assert policy["phase_band_sequences"]["early"] == {
        "basic": 6500,
        "intermediate": 2500,
        "advanced": 800,
        "expert": 200,
    }
    assert policy["phase_band_sequences"]["annealing"]["basic"] == 1000
    assert policy["scalar_difficulty_used"] is False
    assert len(policy["complexity_axes"]) == 3
    assert policy["training_authorized"] is False


def test_spiral_policy_rejects_easy_then_never_again_tamper() -> None:
    policy = copy.deepcopy(build_policy(_budgets()))
    policy["shares_ppm"]["annealing"]["basic"] = 0
    policy["shares_ppm"]["annealing"]["expert"] = 450_000
    _resign(policy)
    with pytest.raises(SpiralCurriculumError, match="contract"):
        validate_policy(policy)


def test_spiral_policy_rejects_scalar_or_unevidenced_graph() -> None:
    policy = copy.deepcopy(build_policy(_budgets()))
    policy["scalar_difficulty_used"] = True
    _resign(policy)
    with pytest.raises(SpiralCurriculumError, match="contract"):
        validate_policy(policy)
    policy = copy.deepcopy(build_policy(_budgets()))
    policy["prerequisite_admission"]["edge_evidence_required"] = False
    _resign(policy)
    with pytest.raises(SpiralCurriculumError, match="prerequisite"):
        validate_policy(policy)


def test_frozen_shares_each_sum_to_one_million() -> None:
    assert all(sum(shares.values()) == 1_000_000 for shares in SHARES_PPM.values())
