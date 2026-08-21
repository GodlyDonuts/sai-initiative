from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sai.adaptive.config import (
    WorkspaceConfig,
    WorkspaceConfigError,
    default_workspace_config,
    workspace_activation_ledger,
    workspace_forward_flop_ledger,
    workspace_parameter_ledger,
)
from sai.adaptive.planner import WorkspacePlanError, build_plan, validate_plan
from sai.adaptive.reference import (
    AdaptiveSaiCausalLM,
    LatentWorkspace,
    WorkspaceReferenceError,
)
from sai.model.config import SaiModelConfig
from sai.model.reference import SaiCausalLM, exact_parameter_count

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_PLAN = ROOT / "docs" / "SAI_48K_SCALE_GEOMETRIES.json"
WORKSPACE_PLAN = ROOT / "docs" / "SAI_16_SLOT_WORKSPACE_MECHANICS.json"


def tiny_base_config() -> SaiModelConfig:
    return SaiModelConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        mixer_family="gdn_hybrid",
        mla_kv_rank=16,
        mla_qk_head_dim=8,
        mla_value_head_dim=8,
    )


def tiny_workspace_config() -> WorkspaceConfig:
    return WorkspaceConfig(
        hidden_size=32,
        workspace_size=16,
        num_slots=4,
        num_heads=4,
        reactor_layers=2,
        reactor_intermediate_size=32,
    )


def adaptive_model() -> AdaptiveSaiCausalLM:
    torch.manual_seed(20260821)
    return AdaptiveSaiCausalLM(
        SaiCausalLM(tiny_base_config()), LatentWorkspace(tiny_workspace_config())
    ).eval()


def test_workspace_parameter_ledger_matches_instantiated_reference() -> None:
    config = tiny_workspace_config()
    workspace = LatentWorkspace(config)
    assert workspace_parameter_ledger(config)["total"] == exact_parameter_count(
        workspace
    )


def test_frozen_300m_workspace_ledger_has_one_factor_and_no_controller() -> None:
    config = default_workspace_config(768)
    ledger = workspace_parameter_ledger(config)
    assert config.num_slots == 16
    assert config.reactor_layers == 4
    assert ledger["total"] == 11_218_176
    assert set(ledger) == {
        "learned_slots",
        "compiler",
        "reactor_per_layer",
        "reactor",
        "reader_zero_initialized_output_included",
        "total",
    }


def test_forced_fast_is_bitwise_direct_base_bypass() -> None:
    model = adaptive_model()
    tokens = torch.tensor([[2, 3, 5, 7, 11]], dtype=torch.long)
    with torch.no_grad():
        expected = model.base(tokens)
        observed, diagnostics = model(tokens, mode="fast", return_diagnostics=True)
    assert diagnostics is None
    assert torch.equal(observed, expected)


def test_zero_initialized_slow_reader_is_bitwise_fast_at_initialization() -> None:
    model = adaptive_model()
    tokens = torch.tensor([[2, 3, 5, 7, 11]], dtype=torch.long)
    with torch.no_grad():
        fast = model(tokens, mode="fast")
        slow, diagnostics = model(
            tokens, mode="slow", iterations=4, return_diagnostics=True
        )
    assert torch.equal(slow, fast)
    assert diagnostics is not None
    assert diagnostics.iterations == 4
    assert diagnostics.initial_slot_rms.item() > 0
    assert diagnostics.final_slot_rms.item() > 0
    assert diagnostics.last_update_rms.item() > 0
    assert diagnostics.output_delta_rms.item() == 0


def test_slow_path_can_change_only_last_position_when_reader_is_enabled() -> None:
    model = adaptive_model()
    torch.nn.init.normal_(model.workspace.reader.o_proj.weight, std=0.1)
    tokens = torch.tensor([[2, 3, 5, 7, 11]], dtype=torch.long)
    with torch.no_grad():
        fast = model(tokens, mode="fast")
        slow = model(tokens, mode="slow", iterations=2)
    assert torch.equal(slow[:, :-1], fast[:, :-1])
    assert not torch.equal(slow[:, -1], fast[:, -1])


def test_workspace_respects_final_packed_segment_boundary() -> None:
    model = adaptive_model()
    torch.nn.init.normal_(model.workspace.reader.o_proj.weight, std=0.1)
    segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.long)
    first = torch.tensor([[2, 3, 5, 7, 11, 13]], dtype=torch.long)
    second = torch.tensor([[71, 73, 79, 7, 11, 13]], dtype=torch.long)
    with torch.no_grad():
        first_logits = model(first, segment_ids, mode="slow", iterations=2)
        second_logits = model(second, segment_ids, mode="slow", iterations=2)
    torch.testing.assert_close(
        first_logits[:, 3:], second_logits[:, 3:], rtol=0, atol=0
    )


def test_recurrence_horizon_changes_compute_not_parameters() -> None:
    config = tiny_workspace_config()
    parameter_total = workspace_parameter_ledger(config)["total"]
    ledgers = [
        workspace_forward_flop_ledger(config, 128, value) for value in (1, 2, 4, 8)
    ]
    increment = ledgers[0]["reactor_per_iteration"]
    assert all(
        workspace_parameter_ledger(config)["total"] == parameter_total for _ in ledgers
    )
    assert (
        ledgers[1]["forced_slow_increment"] - ledgers[0]["forced_slow_increment"]
        == increment
    )
    assert (
        ledgers[2]["forced_slow_increment"] - ledgers[1]["forced_slow_increment"]
        == 2 * increment
    )
    assert (
        ledgers[3]["forced_slow_increment"] - ledgers[2]["forced_slow_increment"]
        == 4 * increment
    )


def test_activation_ledger_declares_reference_only_scope_and_is_monotonic() -> None:
    config = tiny_workspace_config()
    short = workspace_activation_ledger(config, 64)
    long = workspace_activation_ledger(config, 128)
    assert "excluding_backbone" in short["convention"]
    assert long["maximum_stage_elements"] > short["maximum_stage_elements"]
    assert long["maximum_stage_bytes"] == 2 * long["maximum_stage_elements"]


@pytest.mark.parametrize(
    "change",
    [
        {"workspace_size": 15},
        {"num_slots": 0},
        {"reactor_layers": -1},
        {"rms_norm_eps": 0.0},
    ],
)
def test_invalid_workspace_geometry_fails_closed(change: dict) -> None:
    values = tiny_workspace_config().as_dict()
    values.update(change)
    with pytest.raises(WorkspaceConfigError):
        WorkspaceConfig(**values)


def test_invalid_workspace_inputs_and_modes_fail_closed() -> None:
    model = adaptive_model()
    tokens = torch.tensor([[2, 3, 5]], dtype=torch.long)
    with pytest.raises(WorkspaceReferenceError, match="fast or slow"):
        model(tokens, mode="oracle")
    with pytest.raises(WorkspaceReferenceError, match="iterations"):
        model(tokens, mode="slow", iterations=0)


def test_workspace_plan_is_deterministic_and_retains_no_training_hold() -> None:
    geometry = json.loads(GEOMETRY_PLAN.read_text())
    first = build_plan(geometry)
    second = build_plan(geometry)
    assert first == second == validate_plan(first, geometry)
    assert first["selected_backbone_family"] is None
    assert first["primary_100m_screen_unchanged"]
    assert not first["training_authorized"]
    assert first["gpu_jobs_submitted"] == 0
    assert len(first["candidates"]) == 3
    assert all(
        not row
        for row in (
            first["workspace_factor"]["learned_regret_controller_included"],
            first["workspace_factor"]["semantic_memory_included"],
            first["workspace_factor"]["typed_side_channels_included"],
        )
    )


def test_checked_in_workspace_plan_is_exact() -> None:
    geometry = json.loads(GEOMETRY_PLAN.read_text())
    frozen = json.loads(WORKSPACE_PLAN.read_text())
    assert frozen == build_plan(geometry)
    assert validate_plan(frozen, geometry) == frozen


def test_tampered_workspace_plan_fails_closed() -> None:
    geometry = json.loads(GEOMETRY_PLAN.read_text())
    payload = build_plan(geometry)
    payload["candidates"][0]["workspace_parameter_ledger"]["total"] += 1
    with pytest.raises(WorkspacePlanError, match="identity or no-training"):
        validate_plan(payload, geometry)
