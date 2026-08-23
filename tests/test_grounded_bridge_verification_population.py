from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from sai.data.grounded_bridge_aggregate import build_candidate_row
from sai.data.grounded_bridge_verification_population import (
    GroundedBridgeVerificationPopulationError,
    build_candidate,
)
from sai.data.token_stream import canonical_sha256
from tests.test_grounded_bridge_aggregate import receipt
from tests.test_grounded_bridge_labeling import paired_candidate


def _bound_pair() -> dict:
    pair = paired_candidate()
    for key in ("anchor_a", "anchor_b"):
        pair[key]["source_content_sha256"] = hashlib.sha256(
            pair[key]["text"].encode()
        ).hexdigest()
    return pair


def test_build_candidate_restores_exact_anchors_and_remains_nontraining() -> None:
    pair = _bound_pair()
    generator = receipt(pair)
    generated = build_candidate_row(pair, generator)
    row = build_candidate(pair, generated, generator)
    assert row["anchor_a_text"] == pair["anchor_a"]["text"]
    assert row["anchor_b_text"] == pair["anchor_b"]["text"]
    assert row["generated"]["source_quotes_retained_in_candidate"] is False
    assert row["same_model_family_as_generator"] is True
    assert row["independent_request_verification_complete"] is False
    assert row["independent_model_family_verification_complete"] is False
    assert row["bridge_verified"] is False
    assert row["training_ready"] is False
    identity = row.pop("candidate_identity_sha256")
    assert identity == canonical_sha256(row)


def test_build_candidate_rejects_generated_bridge_drift() -> None:
    pair = _bound_pair()
    generator = receipt(pair)
    generated = build_candidate_row(pair, generator)
    drifted = deepcopy(generated)
    drifted["representations"][0]["text"] += " Unsupported drift."
    unsigned = {
        key: value
        for key, value in drifted.items()
        if key != "candidate_identity_sha256"
    }
    drifted["candidate_identity_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(
        GroundedBridgeVerificationPopulationError,
        match="binding",
    ):
        build_candidate(pair, drifted, generator)
