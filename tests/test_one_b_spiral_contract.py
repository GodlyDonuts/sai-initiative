from __future__ import annotations

import copy

import pytest

from sai.data.one_b_spiral_contract import (
    BANDS,
    SEQUENCE_LENGTH,
    TOTAL_TOKENS,
    OneBSpiralContractError,
    build_contract,
    validate_contract,
)


def test_builds_exact_four_trillion_token_spiral() -> None:
    payload = validate_contract(build_contract())
    assert payload["target_tokens"] == TOTAL_TOKENS
    assert payload["target_sequences"] * SEQUENCE_LENGTH == TOTAL_TOKENS
    assert sum(stage["tokens"] for stage in payload["stages"]) == TOTAL_TOKENS
    assert [stage["stage"] for stage in payload["stages"]] == [
        "foundation",
        "expansion",
        "depth",
        "synthesis",
        "annealing",
    ]
    assert all(
        tuple(stage["band_sequences"]) == BANDS for stage in payload["stages"]
    )
    assert all(
        sum(stage["band_sequences"].values()) == stage["sequences"]
        for stage in payload["stages"]
    )
    assert payload["one_b_training_authorized"] is False
    assert payload["four_b_target_retired"] is True


def test_rejects_one_sequence_of_silent_reweighting() -> None:
    payload = copy.deepcopy(build_contract())
    payload["stages"][0]["band_sequences"]["foundation"] -= 1
    payload["stages"][0]["band_sequences"]["expert"] += 1
    with pytest.raises(OneBSpiralContractError, match="differs"):
        validate_contract(payload)
