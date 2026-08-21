from __future__ import annotations

import torch

from sai.model.initialization import POLICY_SHA256, initialize_sai_model
from sai.model.reference import RMSNorm, SaiCausalLM
from tests.test_model_reference import tiny_config


def state(model: SaiCausalLM) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_initialization_is_exactly_reproducible_and_seed_sensitive() -> None:
    first = SaiCausalLM(tiny_config("gdn_hybrid"))
    second = SaiCausalLM(tiny_config("gdn_hybrid"))
    third = SaiCausalLM(tiny_config("gdn_hybrid"))
    first_receipt = initialize_sai_model(first, seed=17)
    second_receipt = initialize_sai_model(second, seed=17)
    initialize_sai_model(third, seed=18)
    assert first_receipt == second_receipt
    assert first_receipt["policy_sha256"] == POLICY_SHA256
    assert all(
        torch.equal(state(first)[name], state(second)[name]) for name in state(first)
    )
    assert any(
        not torch.equal(state(first)[name], state(third)[name]) for name in state(first)
    )


def test_norms_and_direct_decay_parameters_follow_frozen_policy() -> None:
    model = SaiCausalLM(tiny_config("kda_mla_hybrid"))
    receipt = initialize_sai_model(model, seed=23)
    assert receipt["parameter_elements_initialized"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    for module in model.modules():
        if isinstance(module, RMSNorm):
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
    for name, parameter in model.named_parameters():
        if name.endswith(("alpha_log_scale", "alpha_bias")):
            torch.testing.assert_close(parameter, torch.zeros_like(parameter))


def test_initialized_random_token_loss_is_near_uniform_not_explosive() -> None:
    model = SaiCausalLM(tiny_config("gdn_hybrid"))
    initialize_sai_model(model, seed=20260821)
    tokens = torch.randint(0, model.config.vocab_size, (2, 8))
    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, model.config.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    assert 3.0 < float(loss.detach()) < 7.0
