from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sai.adaptive.config import WorkspaceConfig
from sai.adaptive.hf_workspace import (
    FrozenHFWorkspaceSystem,
    matched_workspace_delta,
)
from sai.adaptive.reference import LatentWorkspace, WorkspaceReferenceError


class _Body(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding

    def forward(self, *, input_ids, attention_mask, use_cache):
        assert attention_mask.shape == input_ids.shape
        assert use_cache is False
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _Parent(nn.Module):
    def __init__(self, hidden: int = 8, vocab: int = 17) -> None:
        super().__init__()
        embedding = nn.Embedding(vocab, hidden)
        self.model = _Body(embedding)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(hidden_size=hidden)


def _workspace() -> LatentWorkspace:
    torch.manual_seed(7)
    return LatentWorkspace(
        WorkspaceConfig(
            hidden_size=8,
            workspace_size=8,
            num_slots=3,
            num_heads=2,
            reactor_layers=2,
            reactor_intermediate_size=12,
        )
    )


def test_frozen_parent_and_matched_state_modes() -> None:
    parent = _Parent()
    workspace = _workspace()
    with torch.no_grad():
        workspace.reader.o_proj.weight.normal_(std=0.1)
    system = FrozenHFWorkspaceSystem(parent, workspace)
    assert not any(parameter.requires_grad for parameter in parent.parameters())
    assert system.trainable_parameter_count() == sum(
        value.numel() for value in workspace.parameters()
    )
    system.train()
    assert system.training is True
    assert parent.training is False

    input_ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(input_ids)
    hidden = system.parent_hidden(input_ids, mask)
    segments = torch.tensor([[0, 0, 1, 1]])
    recurrent, parent_logits = system.logits_at(
        hidden,
        segments,
        position=3,
        iterations=2,
        state_mode="recurrent",
    )
    reset, same_parent = system.logits_at(
        hidden,
        segments,
        position=3,
        iterations=2,
        state_mode="reset_average",
    )
    assert torch.equal(parent_logits, same_parent)
    assert not torch.equal(recurrent, reset)


def test_state_modes_execute_identical_reactor_call_counts() -> None:
    workspace = _workspace()
    context = torch.randn(2, 5, 8)
    mask = torch.tensor([[False, True, True, True, True], [True] * 5])
    calls = 0

    def count(*_):
        nonlocal calls
        calls += 1

    handles = [block.register_forward_hook(count) for block in workspace.reactor]
    try:
        matched_workspace_delta(
            workspace,
            context,
            iterations=3,
            context_mask=mask,
            state_mode="recurrent",
        )
        recurrent_calls = calls
        calls = 0
        matched_workspace_delta(
            workspace,
            context,
            iterations=3,
            context_mask=mask,
            state_mode="reset_average",
        )
    finally:
        for handle in handles:
            handle.remove()
    assert recurrent_calls == calls == 3 * len(workspace.reactor)


def test_workspace_rejects_noncontiguous_context_and_unknown_mode() -> None:
    workspace = _workspace()
    context = torch.randn(1, 4, 8)
    with pytest.raises(WorkspaceReferenceError, match="contiguous suffix"):
        matched_workspace_delta(
            workspace,
            context,
            iterations=2,
            context_mask=torch.tensor([[True, False, True, True]]),
            state_mode="recurrent",
        )
    with pytest.raises(WorkspaceReferenceError, match="state mode"):
        matched_workspace_delta(
            workspace,
            context,
            iterations=2,
            context_mask=None,
            state_mode="unknown",  # type: ignore[arg-type]
        )
