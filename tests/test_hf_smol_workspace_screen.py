from __future__ import annotations

import pytest

import sai.training.hf_smol_workspace_screen as screen
from sai.adaptive.config import workspace_parameter_ledger
from sai.training.runner import TrainingRunConfig


def _common() -> dict:
    return {
        "snapshot_tree_sha256": "1" * 64,
        "mechanics_file_sha256": "2" * 64,
        "stream_identity_sha256": "3" * 64,
        "source_manifest_sha256": "4" * 64,
        "training_sequences": 256,
        "training_utf8_bytes": 123,
        "optimizer": TrainingRunConfig(
            optimizer_steps=8,
            learning_rate=3e-4,
            warmup_steps=8,
        ),
        "code_sha256": "5" * 64,
        "environment_sha256": "6" * 64,
    }


def test_smol_geometry_and_matched_control_identity() -> None:
    assert screen.EXPECTED_WORKSPACE_PARAMETERS == 79_722_496
    assert workspace_parameter_ledger(screen.WORKSPACE_CONFIG)["total"] == (
        screen.EXPECTED_WORKSPACE_PARAMETERS
    )
    assert screen.WORKSPACE_CONFIG.hidden_size == 2048
    assert screen.WORKSPACE_CONFIG.workspace_size == 1024
    assert screen.EXPECTED_EOS_TOKEN_ID == 128_012

    recurrent_binding, recurrent = screen.make_bindings(
        state_mode="recurrent", **_common()
    )
    reset_binding, reset = screen.make_bindings(state_mode="reset_average", **_common())
    assert recurrent_binding.model_sha256 == reset_binding.model_sha256
    assert recurrent_binding.config_sha256 == reset_binding.config_sha256
    assert recurrent_binding.run_sha256 != reset_binding.run_sha256
    assert recurrent["parent"] == reset["parent"]
    assert recurrent["workspace_config"] == reset["workspace_config"]
    assert recurrent["optimizer"] == reset["optimizer"]
    assert recurrent["matched_control_contract"] == reset["matched_control_contract"]
    assert recurrent["matched_control_contract"]["only_changed_factor"] == (
        "reactor_state_carry_between_iterations"
    )
    assert recurrent["four_b_training_executed"] is False
    assert recurrent["four_b_training_authorized_by_this_result"] is False


def test_smol_binding_rejects_unfrozen_budget_or_mode() -> None:
    common = _common()
    common["training_sequences"] = 257
    with pytest.raises(screen.HFWorkspaceScreenError, match="prefix"):
        screen.make_bindings(state_mode="recurrent", **common)
    common["training_sequences"] = 256
    with pytest.raises(screen.HFWorkspaceScreenError, match="state mode"):
        screen.make_bindings(state_mode="other", **common)  # type: ignore[arg-type]
