"""Frozen-parent behavior replay for matched Sai adapter training."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayError(RuntimeError):
    """The adapter or replay geometry is invalid."""


class LoRALinear(nn.Module):
    """Frozen linear projection plus a switchable low-rank residual."""

    is_sai_adapter = True

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ReplayError("LoRA rank and alpha must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.adapter_enabled = True
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        self.lora_b.to(device=base.weight.device, dtype=base.weight.dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        parent = self.base(inputs)
        if not self.adapter_enabled:
            return parent
        return parent + self.lora_b(self.lora_a(inputs)) * self.scale


def install_lora(module: nn.Module, rank: int, alpha: float) -> int:
    """Replace descendant linear projections and return their exact count."""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
            replaced += 1
        else:
            replaced += install_lora(child, rank, alpha)
    return replaced


@contextmanager
def adapters_enabled(model: nn.Module, enabled: bool) -> Iterator[None]:
    """Temporarily switch every explicitly marked Sai adapter."""

    adapters = [
        module
        for module in model.modules()
        if getattr(module, "is_sai_adapter", False)
        and hasattr(module, "adapter_enabled")
    ]
    if not adapters:
        raise ReplayError("model exposes no Sai adapters")
    previous = [bool(module.adapter_enabled) for module in adapters]
    try:
        for module in adapters:
            module.adapter_enabled = enabled
        yield
    finally:
        for module, state in zip(adapters, previous, strict=True):
            module.adapter_enabled = state


def pack_token_rows(
    embedding: nn.Module,
    rows: list[list[int]],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embed and right-pad complete replay sequences."""

    if not rows or any(len(row) < 2 for row in rows):
        raise ReplayError("each replay sequence needs at least two tokens")
    width = max(len(row) for row in rows)
    device = next(embedding.parameters()).device
    token_ids = torch.full(
        (len(rows), width), pad_token_id, device=device, dtype=torch.long
    )
    attention = torch.zeros((len(rows), width), device=device, dtype=torch.long)
    for index, row in enumerate(rows):
        token_ids[index, : len(row)] = torch.tensor(
            row, device=device, dtype=torch.long
        )
        attention[index, : len(row)] = 1
    return embedding(token_ids), attention


def behavior_replay_kl(
    model: Any,
    token_rows: list[list[int]],
    pad_token_id: int,
    *,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Return token-mean KL from adapted student to its frozen parent behavior."""

    if temperature <= 0:
        raise ReplayError("replay temperature must be positive")
    text_model = getattr(model, "text_model", None)
    lm_head = getattr(model, "lm_head", None)
    embedding = getattr(text_model, "embed_tokens", None)
    if text_model is None or lm_head is None or embedding is None:
        raise ReplayError("model does not expose the Sai causal language path")
    inputs, attention = pack_token_rows(embedding, token_rows, pad_token_id)
    valid = attention[:, 1:].bool()
    replay_tokens = int(valid.sum())
    if replay_tokens <= 0:
        raise ReplayError("replay contains no predicted tokens")

    with adapters_enabled(model, False), torch.no_grad():
        parent_hidden = text_model(
            inputs_embeds=inputs,
            attention_mask=attention,
            use_cache=False,
        ).last_hidden_state
        parent_log_probabilities = F.log_softmax(
            lm_head(parent_hidden[:, :-1]).float() / temperature,
            dim=-1,
        )
    student_hidden = text_model(
        inputs_embeds=inputs,
        attention_mask=attention,
        use_cache=False,
    ).last_hidden_state
    student_log_probabilities = F.log_softmax(
        lm_head(student_hidden[:, :-1]).float() / temperature,
        dim=-1,
    )
    token_kl = F.kl_div(
        student_log_probabilities,
        parent_log_probabilities,
        log_target=True,
        reduction="none",
    ).sum(dim=-1)
    return token_kl[valid].mean() * temperature**2, replay_tokens


def matched_training_loss(
    task_loss: torch.Tensor,
    replay_kl: torch.Tensor,
    replay_weight: float,
) -> torch.Tensor:
    """Combine losses while keeping weight zero as the equal-compute control."""

    if replay_weight < 0 or task_loss.ndim != 0 or replay_kl.ndim != 0:
        raise ReplayError("matched training loss geometry differs")
    return task_loss + replay_kl * replay_weight
