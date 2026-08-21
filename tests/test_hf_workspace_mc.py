from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sai.adaptive.config import WorkspaceConfig
from sai.adaptive.hf_workspace import FrozenHFWorkspaceSystem
from sai.adaptive.reference import LatentWorkspace
from sai.evaluation.hf_workspace_mc import (
    HFWorkspaceChoiceAdapter,
    HFWorkspaceEvaluationError,
)


class _Body(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.calls = 0

    def forward(self, *, input_ids, attention_mask, use_cache):
        self.calls += 1
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        assert use_cache is False
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _Parent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        embedding = nn.Embedding(23, 8)
        self.model = _Body(embedding)
        self.lm_head = nn.Linear(8, 23, bias=False)
        self.config = SimpleNamespace(hidden_size=8)


def _adapter(state_mode: str = "recurrent") -> HFWorkspaceChoiceAdapter:
    torch.manual_seed(9)
    workspace = LatentWorkspace(
        WorkspaceConfig(
            hidden_size=8,
            workspace_size=8,
            num_slots=2,
            num_heads=2,
            reactor_layers=2,
            reactor_intermediate_size=12,
        )
    )
    with torch.no_grad():
        workspace.reader.o_proj.weight.normal_(std=0.1)
    return HFWorkspaceChoiceAdapter(
        FrozenHFWorkspaceSystem(_Parent(), workspace),
        state_mode=state_mode,  # type: ignore[arg-type]
    )


def test_choice_adapter_runs_parent_once_and_only_selected_positions() -> None:
    adapter = _adapter()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    logits = adapter.choice_logits(
        input_ids,
        torch.zeros_like(input_ids),
        start_position=2,
        token_count=2,
    )
    assert logits.shape == (2, 23)
    assert adapter.system.parent.model.calls == 1
    assert adapter.system.parent.training is False
    with pytest.raises(HFWorkspaceEvaluationError, match="selected choice"):
        adapter(input_ids, torch.zeros_like(input_ids))


@pytest.mark.parametrize(
    ("segments", "start", "count"),
    [([0, 1, 1], 1, 1), ([0, 0, 0], -1, 1), ([0, 0, 0], 2, 2)],
)
def test_choice_adapter_rejects_noncanonical_geometry(
    segments: list[int], start: int, count: int
) -> None:
    adapter = _adapter()
    input_ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(HFWorkspaceEvaluationError, match="choice geometry"):
        adapter.choice_logits(
            input_ids,
            torch.tensor([segments]),
            start_position=start,
            token_count=count,
        )


def test_choice_adapter_rejects_unknown_state_mode() -> None:
    with pytest.raises(HFWorkspaceEvaluationError, match="state mode"):
        _adapter("unknown")
