from __future__ import annotations

import copy

import pytest

from sai.evaluation.curriculum_milestone_nll import (
    CurriculumMilestoneNLLError,
    summarize_learning_curve,
)


def _observation(step: int, values: dict[str, float]) -> dict:
    return {
        "optimizer_step": step,
        "development_nll": {
            "strata": {
                phase: {"nll_per_target": value} for phase, value in values.items()
            }
        },
    }


def _curve() -> list[dict]:
    return [
        _observation(0, {"grounding": 5.0, "integration": 5.2, "reasoning": 5.4}),
        _observation(2, {"grounding": 3.0, "integration": 4.8, "reasoning": 5.1}),
        _observation(4, {"grounding": 3.1, "integration": 3.4, "reasoning": 4.7}),
        _observation(6, {"grounding": 3.2, "integration": 3.5, "reasoning": 3.8}),
    ]


def test_summarizes_phase_acquisition_and_forgetting() -> None:
    payload = summarize_learning_curve(
        _curve(), milestone_steps=(2, 4), optimizer_steps=6
    )
    assert payload["observation_steps"] == [0, 2, 4, 6]
    assert payload["phase_order"] == ["grounding", "integration", "reasoning"]
    assert payload["all_phases_acquired_by_completion"] is True
    assert payload["all_phases_better_than_initialization_at_terminal"] is True
    assert payload["phases"]["grounding"] == {
        "completion_step": 2,
        "initial_nll_per_target": 5.0,
        "completion_nll_per_target": 3.0,
        "terminal_nll_per_target": 3.2,
        "acquisition_delta": -2.0,
        "post_completion_forgetting_delta": pytest.approx(0.2),
        "terminal_delta_from_initialization": pytest.approx(-1.8),
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda rows: rows.pop(),
        lambda rows: rows[1].update(optimizer_step=3),
        lambda rows: rows[2]["development_nll"]["strata"].pop("reasoning"),
        lambda rows: rows[2]["development_nll"]["strata"]["grounding"].update(
            nll_per_target=float("nan")
        ),
    ),
)
def test_rejects_missing_reordered_or_nonfinite_curve(mutation) -> None:
    rows = copy.deepcopy(_curve())
    mutation(rows)
    with pytest.raises(CurriculumMilestoneNLLError):
        summarize_learning_curve(rows, milestone_steps=(2, 4), optimizer_steps=6)


def test_reports_failed_acquisition_without_reclassifying_evidence() -> None:
    rows = _curve()
    rows[2]["development_nll"]["strata"]["integration"]["nll_per_target"] = 5.3
    payload = summarize_learning_curve(rows, milestone_steps=(2, 4), optimizer_steps=6)
    assert payload["all_phases_acquired_by_completion"] is False
