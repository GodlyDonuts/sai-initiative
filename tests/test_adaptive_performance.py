from __future__ import annotations

import copy

import pytest

from sai.adaptive.config import WorkspaceConfig
from sai.adaptive.performance import (
    WorkspacePerformanceError,
    run_cpu_mechanics,
    validate_receipt,
)


def tiny_config() -> WorkspaceConfig:
    return WorkspaceConfig(
        hidden_size=16,
        workspace_size=16,
        num_slots=4,
        num_heads=4,
        reactor_layers=2,
        reactor_intermediate_size=32,
    )


@pytest.fixture(scope="module")
def receipt() -> dict:
    return run_cpu_mechanics(
        tiny_config(), sequence_length=8, iterations=2, warmups=1, samples=3
    )


def test_cpu_mechanics_is_valid_no_training_evidence(receipt: dict) -> None:
    assert validate_receipt(receipt) == receipt
    assert receipt["measurement_receipt_valid"]
    assert receipt["design_performance_gate_pass"] is None
    assert not receipt["production_qualified"]
    assert receipt["model_state_unchanged"]
    assert receipt["rng_state_unchanged"]
    assert receipt["call_counts"] == {
        "compiler": 3,
        "reactor": 12,
        "reader": 3,
    }
    assert len(set(receipt["raw_latency_ns"])) >= 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_authorized", True),
        ("gpu_jobs_submitted", 1),
        ("optimizer_steps", 1),
        ("backward_calls", 1),
        ("production_qualified", True),
        ("design_performance_gate_pass", True),
        ("dram_traffic_measured", True),
        ("model_state_unchanged", False),
    ],
)
def test_boundary_tampering_fails_closed(
    receipt: dict, field: str, value: object
) -> None:
    changed = copy.deepcopy(receipt)
    changed[field] = value
    with pytest.raises(WorkspacePerformanceError):
        validate_receipt(changed)


def test_resigned_latency_or_call_count_tampering_fails_closed(receipt: dict) -> None:
    from sai.adaptive.performance import canonical_sha256

    changed = copy.deepcopy(receipt)
    changed["call_counts"]["reader"] = 2
    changed["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "receipt_sha256"}
    )
    with pytest.raises(WorkspacePerformanceError):
        validate_receipt(changed)

    changed = copy.deepcopy(receipt)
    changed["raw_latency_ns"][0] += 1
    changed["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "receipt_sha256"}
    )
    with pytest.raises(WorkspacePerformanceError):
        validate_receipt(changed)


def test_invalid_measurement_geometry_fails_closed() -> None:
    with pytest.raises(WorkspacePerformanceError):
        run_cpu_mechanics(tiny_config(), sequence_length=0, iterations=2)
    with pytest.raises(WorkspacePerformanceError):
        run_cpu_mechanics(tiny_config(), sequence_length=8, iterations=0)
    with pytest.raises(WorkspacePerformanceError):
        run_cpu_mechanics(tiny_config(), sequence_length=8, iterations=2, samples=2)
