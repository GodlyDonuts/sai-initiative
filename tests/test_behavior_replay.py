from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from sai.training.replay import (
    LoRALinear,
    adapters_enabled,
    behavior_replay_kl,
    matched_training_loss,
)


class Text(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(20, 8)
        self.projection = LoRALinear(nn.Linear(8, 8), rank=2, alpha=4.0)

    def forward(self, *, inputs_embeds, attention_mask, use_cache):
        del attention_mask, use_cache
        return SimpleNamespace(last_hidden_state=self.projection(inputs_embeds))


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_model = Text()
        self.lm_head = nn.Linear(8, 20, bias=False)
        self.lm_head.requires_grad_(False)


def test_adapter_switch_exposes_exact_parent_and_restores() -> None:
    torch.manual_seed(7)
    layer = LoRALinear(nn.Linear(5, 7, bias=False), rank=2, alpha=4.0)
    with torch.no_grad():
        layer.lora_b.weight.fill_(0.5)
    inputs = torch.randn(3, 5)
    adapted = layer(inputs)
    with adapters_enabled(layer, False):
        assert torch.equal(layer(inputs), layer.base(inputs))
    assert layer.adapter_enabled
    assert torch.equal(layer(inputs), adapted)


def test_replay_kl_starts_at_parent_and_penalizes_drift() -> None:
    model = Model()
    initial, tokens = behavior_replay_kl(model, [[1, 2, 3, 4]], 0)
    assert tokens == 3
    assert float(initial.detach()) == 0.0
    with torch.no_grad():
        model.text_model.projection.lora_b.weight.fill_(0.5)
    drift, _ = behavior_replay_kl(model, [[1, 2, 3, 4]], 0)
    assert float(drift.detach()) > 0
    drift.backward()
    assert model.text_model.projection.lora_b.weight.grad is not None
    assert model.text_model.projection.base.weight.grad is None


def test_zero_replay_weight_is_matched_equal_compute_control() -> None:
    task = torch.tensor(2.0, requires_grad=True)
    replay = torch.tensor(3.0, requires_grad=True)
    total = matched_training_loss(task, replay, 0.0)
    assert float(total.detach()) == 2.0
    total.backward()
    assert task.grad is not None
    assert replay.grad is not None
    assert float(replay.grad) == 0.0
