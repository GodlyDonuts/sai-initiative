from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from sai.model.config import (
    SCALE_TEMPLATES,
    SaiConfigError,
    SaiModelConfig,
    fit_scale_geometry,
    forward_flop_ledger,
    frozen_scale_geometries,
    parameter_ledger,
)
from sai.model.planner import SaiModelPlanError, build_plan, validate_plan
from sai.model.reference import (
    SaiCausalLM,
    causal_delta_recurrence,
    exact_parameter_count,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_GEOMETRIES = ROOT / "docs" / "SAI_48K_SCALE_GEOMETRIES.json"


def tiny_config(family: str) -> SaiModelConfig:
    return SaiModelConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        mixer_family=family,
        mla_kv_rank=16,
        mla_qk_head_dim=8,
        mla_value_head_dim=8,
    )


def test_exact_frozen_layer_patterns() -> None:
    assert tiny_config("gated_gqa").layer_types() == ["gated_gqa"] * 4
    assert tiny_config("gdn_hybrid").layer_types() == [
        "gated_deltanet",
        "gated_deltanet",
        "gated_deltanet",
        "gated_gqa",
    ]
    assert tiny_config("kda_mla_hybrid").layer_types() == [
        "kda",
        "kda",
        "kda",
        "gated_mla",
    ]


@pytest.mark.parametrize("family", ["gated_gqa", "gdn_hybrid", "kda_mla_hybrid"])
def test_analytical_ledger_matches_real_module(family: str) -> None:
    config = tiny_config(family)
    model = SaiCausalLM(config)
    assert parameter_ledger(config)["total"] == exact_parameter_count(model)
    assert model.lm_head_weight.data_ptr() == model.embed_tokens.weight.data_ptr()


def test_ledger_matches_non_square_attention_and_mla_projections() -> None:
    for family in ("gated_gqa", "kda_mla_hybrid"):
        config = SaiModelConfig(
            vocab_size=96,
            hidden_size=48,
            intermediate_size=80,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            mixer_family=family,
            mla_kv_rank=12,
            mla_qk_head_dim=6,
            mla_value_head_dim=10,
        )
        model = SaiCausalLM(config)
        assert parameter_ledger(config)["total"] == exact_parameter_count(model)
        assert model(torch.randint(0, 96, (1, 5))).shape == (1, 5, 96)


def test_kda_reduces_exactly_to_scalar_decay_when_channels_match() -> None:
    generator = torch.Generator().manual_seed(20260821)
    query = torch.randn(2, 7, 3, 4, generator=generator)
    key = torch.randn(2, 7, 3, 4, generator=generator)
    value = torch.randn(2, 7, 3, 5, generator=generator)
    scalar_alpha = torch.sigmoid(torch.randn(2, 7, 3, 1, generator=generator))
    beta = torch.sigmoid(torch.randn(2, 7, 3, 1, generator=generator))
    scalar_output, scalar_state = causal_delta_recurrence(
        query, key, value, scalar_alpha, beta
    )
    channel_output, channel_state = causal_delta_recurrence(
        query, key, value, scalar_alpha.expand(-1, -1, -1, 4), beta
    )
    torch.testing.assert_close(channel_output, scalar_output, rtol=0, atol=0)
    torch.testing.assert_close(channel_state, scalar_state, rtol=0, atol=0)


def test_recurrent_state_is_exact_across_chunk_boundary() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(1, 9, 2, 4, generator=generator)
    key = torch.randn(1, 9, 2, 4, generator=generator)
    value = torch.randn(1, 9, 2, 6, generator=generator)
    alpha = torch.sigmoid(torch.randn(1, 9, 2, 4, generator=generator))
    beta = torch.sigmoid(torch.randn(1, 9, 2, 1, generator=generator))
    full_output, full_state = causal_delta_recurrence(query, key, value, alpha, beta)
    first_output, first_state = causal_delta_recurrence(
        query[:, :4], key[:, :4], value[:, :4], alpha[:, :4], beta[:, :4]
    )
    second_output, second_state = causal_delta_recurrence(
        query[:, 4:],
        key[:, 4:],
        value[:, 4:],
        alpha[:, 4:],
        beta[:, 4:],
        first_state,
    )
    torch.testing.assert_close(
        torch.cat((first_output, second_output), dim=1), full_output, rtol=0, atol=0
    )
    torch.testing.assert_close(second_state, full_state, rtol=0, atol=0)


def test_recurrent_state_is_isolated_between_batch_members() -> None:
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(2, 6, 2, 4, generator=generator)
    key = torch.randn(2, 6, 2, 4, generator=generator)
    value = torch.randn(2, 6, 2, 5, generator=generator)
    alpha = torch.sigmoid(torch.randn(2, 6, 2, 4, generator=generator))
    beta = torch.sigmoid(torch.randn(2, 6, 2, 1, generator=generator))
    batched_output, batched_state = causal_delta_recurrence(
        query, key, value, alpha, beta
    )
    for batch_index in range(2):
        output, state = causal_delta_recurrence(
            query[batch_index : batch_index + 1],
            key[batch_index : batch_index + 1],
            value[batch_index : batch_index + 1],
            alpha[batch_index : batch_index + 1],
            beta[batch_index : batch_index + 1],
        )
        torch.testing.assert_close(output[0], batched_output[batch_index])
        torch.testing.assert_close(state[0], batched_state[batch_index])


@pytest.mark.parametrize("family", ["gated_gqa", "gdn_hybrid", "kda_mla_hybrid"])
def test_every_reference_family_is_causal_and_has_finite_gradients(family: str) -> None:
    torch.manual_seed(11)
    model = SaiCausalLM(tiny_config(family))
    tokens = torch.randint(0, 128, (2, 7))
    prefix_logits = model(tokens[:, :5])
    full_logits = model(tokens)
    torch.testing.assert_close(full_logits[:, :5], prefix_logits, rtol=2e-4, atol=5e-6)
    loss = full_logits.float().square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_scale_planner_stays_near_each_target_for_every_core_family() -> None:
    rows = frozen_scale_geometries(vocab_size=48_000)
    assert len(rows) == 12
    assert {row["scale"] for row in rows} == {"100m", "300m", "1b", "4b"}
    assert {row["mixer_family"] for row in rows} == {
        "gated_gqa",
        "gdn_hybrid",
        "kda_mla_hybrid",
    }
    assert all(abs(float(row["relative_error"])) < 0.007 for row in rows)
    assert all(row["flop_ledger_2048"]["forward"] > 0 for row in rows)


@pytest.mark.parametrize("template", SCALE_TEMPLATES)
@pytest.mark.parametrize("family", ["gated_gqa", "gdn_hybrid", "kda_mla_hybrid"])
def test_smaller_vocab_reinvests_in_larger_ffn_at_fixed_total(
    template, family: str
) -> None:
    sizes = [
        fit_scale_geometry(template, family, vocab).intermediate_size
        for vocab in (64_000, 48_000, 32_000)
    ]
    assert sizes[0] < sizes[1] < sizes[2]


@pytest.mark.parametrize(
    "change",
    [
        {"num_key_value_heads": 3},
        {"head_dim": 7},
        {"tie_word_embeddings": False},
        {"dropout": 0.1},
        {"rms_norm_eps": float("nan")},
    ],
)
def test_invalid_or_out_of_contract_geometry_is_rejected(change: dict) -> None:
    values = tiny_config("gated_gqa").as_dict()
    values.update(change)
    with pytest.raises(SaiConfigError):
        SaiModelConfig(**values)


def test_scale_targets_are_strictly_increasing() -> None:
    totals = [template.target_parameters for template in SCALE_TEMPLATES]
    assert totals == sorted(totals)
    assert all(math.isfinite(total) and total > 0 for total in totals)


def test_flop_ledger_declares_scope_and_counts_output_projection() -> None:
    ledger = forward_flop_ledger(tiny_config("gated_gqa"), 32)
    assert ledger["sequence_length"] == 32
    assert ledger["tied_output_logits"] == 2 * 32 * 32 * 128
    assert ledger["forward_plus_backward_approximation"] == 3 * ledger["forward"]
    assert "multiply_add" in ledger["convention"]


def test_geometry_plan_is_deterministic_and_retains_training_hold() -> None:
    first = build_plan(48_000)
    second = build_plan(48_000)
    assert first == second == validate_plan(first)
    assert not first["training_authorized"]
    assert first["gpu_jobs_submitted"] == 0
    assert len(first["geometries"]) == 12


def test_checked_in_48k_geometry_receipt_is_exact() -> None:
    frozen = json.loads(FROZEN_GEOMETRIES.read_text())
    assert frozen == build_plan(48_000)
    assert validate_plan(frozen) == frozen


def test_tampered_geometry_plan_fails_closed() -> None:
    payload = build_plan(48_000)
    payload["geometries"][0]["parameter_ledger"]["total"] += 1
    with pytest.raises(SaiModelPlanError, match="identity or no-training"):
        validate_plan(payload)
