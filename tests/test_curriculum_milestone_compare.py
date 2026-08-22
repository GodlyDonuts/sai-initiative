from __future__ import annotations

import copy

import pytest

from sai.evaluation.curriculum_milestone_compare import (
    CurriculumMilestoneComparisonError,
    compare_curves,
)


def _observation(step: int, values: dict[str, float], state: str) -> dict:
    return {
        "optimizer_step": step,
        "model_state_sha256": state,
        "development_nll": {
            "strata": {
                phase: {"nll_per_target": value} for phase, value in values.items()
            }
        },
    }


def _receipt(*, curriculum: bool) -> dict:
    initial = {"grounding": 5.0, "integration": 5.2, "reasoning": 5.4}
    if curriculum:
        values = (
            initial,
            {"grounding": 3.0, "integration": 4.7, "reasoning": 5.0},
            {"grounding": 3.1, "integration": 3.3, "reasoning": 4.6},
            {"grounding": 3.2, "integration": 3.4, "reasoning": 3.7},
        )
    else:
        values = (
            initial,
            {"grounding": 3.2, "integration": 4.8, "reasoning": 5.1},
            {"grounding": 3.4, "integration": 3.6, "reasoning": 4.8},
            {"grounding": 3.6, "integration": 3.8, "reasoning": 4.0},
        )
    observations = [
        _observation(step, row, "1" * 64 if step == 0 else f"{step:x}" * 64)
        for step, row in zip((0, 2, 4, 6), values, strict=True)
    ]
    phases = {}
    for phase, completion in zip(
        ("grounding", "integration", "reasoning"), (2, 4, 6), strict=True
    ):
        by_step = {row["optimizer_step"]: row for row in observations}
        first = by_step[0]["development_nll"]["strata"][phase]["nll_per_target"]
        middle = by_step[completion]["development_nll"]["strata"][phase][
            "nll_per_target"
        ]
        terminal = by_step[6]["development_nll"]["strata"][phase]["nll_per_target"]
        phases[phase] = {
            "completion_step": completion,
            "initial_nll_per_target": first,
            "completion_nll_per_target": middle,
            "terminal_nll_per_target": terminal,
            "acquisition_delta": middle - first,
            "post_completion_forgetting_delta": terminal - middle,
            "terminal_delta_from_initialization": terminal - first,
        }
    return {
        "training_run": {
            "model_sha256": "2" * 64,
            "run_sha256": ("3" if curriculum else "4") * 64,
            "training_stream_identity_sha256": ("5" if curriculum else "6") * 64,
        },
        "development_stream": {"ordered_stream_identity_sha256": "7" * 64},
        "milestone_steps": [2, 4],
        "observations": observations,
        "learning_curve": {
            "observation_steps": [0, 2, 4, 6],
            "phase_order": ["grounding", "integration", "reasoning"],
            "phases": phases,
        },
    }


def test_requires_curriculum_to_win_acquisition_and_retention_phasewise() -> None:
    payload = compare_curves(_receipt(curriculum=True), _receipt(curriculum=False))
    assert payload["curriculum_progression_mechanics_supported"] is True
    assert payload["real_benchmark_confirmation_still_required"] is True
    assert payload["data_promotion_authorized"] is False
    assert all(
        row["curriculum_no_worse_at_completion"]
        and row["curriculum_no_more_forgetting"]
        and row["curriculum_no_worse_at_terminal"]
        for row in payload["phases"].values()
    )


def test_one_phase_vetoes_progression_without_becoming_authorization() -> None:
    curriculum = _receipt(curriculum=True)
    curriculum["observations"][2]["development_nll"]["strata"]["integration"][
        "nll_per_target"
    ] = 3.9
    payload = compare_curves(curriculum, _receipt(curriculum=False))
    assert payload["curriculum_progression_mechanics_supported"] is False
    assert payload["four_b_training_authorized"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda row: row.update(development_stream={"drift": True}),
        lambda row: row["milestone_steps"].append(5),
        lambda row: row["observations"][0].update(model_state_sha256="9" * 64),
        lambda row: row["training_run"].update(run_sha256="3" * 64),
    ),
)
def test_rejects_unmatched_control(mutation) -> None:
    control = copy.deepcopy(_receipt(curriculum=False))
    mutation(control)
    with pytest.raises(CurriculumMilestoneComparisonError):
        compare_curves(_receipt(curriculum=True), control)
