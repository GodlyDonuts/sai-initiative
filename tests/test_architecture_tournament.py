from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sai.architecture.tournament import TournamentError, validate

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "SAI_FRONTIER_ARCHITECTURE_TOURNAMENT.json"


def plan() -> dict:
    return json.loads(PLAN.read_text())


def test_frozen_prospective_plan_is_valid_but_authorizes_nothing() -> None:
    receipt = validate(plan())
    assert receipt["status"] == "prospective_plan_validated"
    assert not receipt["training_authorized"]
    assert receipt["official_training_order_required"]
    assert receipt["scale_order"] == ["mechanics", "100m", "300m", "1b", "4b"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_hold", False),
        ("training_authorized", True),
        ("official_training_order_received", True),
        ("gpu_jobs_submitted", 1),
        ("training_updates_completed", 1),
    ],
)
def test_any_execution_or_authorization_fails_closed(field: str, value: object) -> None:
    payload = plan()
    payload[field] = value
    with pytest.raises(TournamentError, match="no-training boundary"):
        validate(payload)


def test_skipping_the_scale_ladder_fails_closed() -> None:
    payload = plan()
    del payload["scales"][2]
    with pytest.raises(TournamentError, match="exact scale ladder"):
        validate(payload)


def test_unmatched_flops_fail_closed() -> None:
    payload = plan()
    payload["matching"]["primary_contrasts"]["iso_flop"]["same_model_flops"] = False
    with pytest.raises(TournamentError, match="matched-comparison"):
        validate(payload)


def test_premature_4b_prerequisite_claim_fails_closed() -> None:
    payload = plan()
    payload["four_b_prerequisites"]["300m_passed"] = True
    with pytest.raises(TournamentError, match="must remain unmet"):
        validate(payload)


def test_factor_set_is_exact() -> None:
    payload = copy.deepcopy(plan())
    payload["core_mixer_candidates"].append("fashionable_unmeasured_stack")
    with pytest.raises(TournamentError, match="core mixer tournament"):
        validate(payload)


def test_secondary_factor_order_is_frozen() -> None:
    payload = plan()
    payload["ordered_secondary_factors"].reverse()
    with pytest.raises(TournamentError, match="secondary factor order"):
        validate(payload)
