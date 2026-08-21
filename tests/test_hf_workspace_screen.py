from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import sai.training.hf_workspace_screen as screen
from sai.adaptive.config import WorkspaceConfig, workspace_parameter_ledger
from sai.adaptive.hf_workspace import FrozenHFWorkspaceSystem
from sai.adaptive.reference import LatentWorkspace
from sai.training.runner import TrainingRunConfig
from sai.training.stream import IGNORE_TARGET, StreamCursor, TrainingBatch


class _Body(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding
        self.seen_lengths: list[int] = []

    def forward(self, *, input_ids, attention_mask, use_cache):
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        assert use_cache is False
        self.seen_lengths.append(input_ids.shape[1])
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _Parent(nn.Module):
    def __init__(self, hidden: int = 8, vocab: int = 19) -> None:
        super().__init__()
        embedding = nn.Embedding(vocab, hidden)
        self.model = _Body(embedding)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(hidden_size=hidden)


def _batch(*, boundary_target: bool = False) -> TrainingBatch:
    length = screen.INPUT_SEQUENCE_LENGTH
    x = tuple(index % 19 for index in range(length))
    y = list((index + 1) % 19 for index in range(length))
    mask = [True] * length
    segments = [0, 0, *([1] * (length - 2))]
    if not boundary_target:
        y[1] = IGNORE_TARGET
        mask[1] = False
    identity = "a" * 64
    return TrainingBatch(
        x=(x,),
        y=(tuple(y),),
        segment_ids=(tuple(segments),),
        loss_mask=(tuple(mask),),
        first_sequence=0,
        resume_cursor=StreamCursor(identity, 1),
    )


def _system() -> FrozenHFWorkspaceSystem:
    torch.manual_seed(4)
    parent = _Parent()
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
    return FrozenHFWorkspaceSystem(parent, workspace)


def test_frozen_geometry_and_matched_mode_bindings() -> None:
    assert workspace_parameter_ledger(screen.WORKSPACE_CONFIG)["total"] == (
        screen.EXPECTED_WORKSPACE_PARAMETERS
    )
    optimizer = TrainingRunConfig(
        optimizer_steps=8,
        learning_rate=3e-4,
        warmup_steps=8,
    )
    common = dict(
        snapshot_tree_sha256="1" * 64,
        mechanics_file_sha256="2" * 64,
        stream_identity_sha256="3" * 64,
        source_manifest_sha256="4" * 64,
        training_sequences=256,
        training_utf8_bytes=123,
        optimizer=optimizer,
        code_sha256="5" * 64,
        environment_sha256="6" * 64,
    )
    recurrent_binding, recurrent = screen.make_bindings(
        state_mode="recurrent", **common
    )
    reset_binding, reset = screen.make_bindings(state_mode="reset_average", **common)
    assert recurrent_binding.model_sha256 == reset_binding.model_sha256
    assert recurrent_binding.config_sha256 == reset_binding.config_sha256
    assert recurrent_binding.run_sha256 != reset_binding.run_sha256
    assert recurrent["state_mode"] == "recurrent"
    assert reset["state_mode"] == "reset_average"
    assert recurrent["matched_control_contract"]["only_changed_factor"] == (
        "reactor_state_carry_between_iterations"
    )
    assert recurrent["four_b_training_executed"] is False


def test_objective_uses_only_current_document_and_admitted_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(screen, "POSITIONS", (1, 3))
    system = _system()
    batch = _batch()
    assert screen.selected_target_count(batch) == 1
    objective, observed = screen.matched_objective_sum(
        system,
        batch,
        state_mode="recurrent",
        device="cpu",
    )
    assert objective.requires_grad
    assert observed["targets"] == 1
    assert observed["cross_entropy_sum"] > 0
    assert observed["parent_kl_sum"] >= 0
    assert system.parent.model.seen_lengths == [2]
    objective.backward()
    assert any(
        parameter.grad is not None for parameter in system.workspace.parameters()
    )
    assert all(parameter.grad is None for parameter in system.parent.parameters())


def test_cross_document_target_is_not_silently_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(screen, "POSITIONS", (1,))
    batch = _batch(boundary_target=True)
    # The frozen stream contract, not an arbitrary target ID, determines whether
    # a cross-document prediction is trainable.
    batch = TrainingBatch(
        x=batch.x,
        y=batch.y,
        segment_ids=batch.segment_ids,
        loss_mask=(
            tuple(
                False if index == 1 else value
                for index, value in enumerate(batch.loss_mask[0])
            ),
        ),
        first_sequence=batch.first_sequence,
        resume_cursor=batch.resume_cursor,
    )
    with pytest.raises(screen.HFWorkspaceScreenError, match="target mask differs"):
        screen.matched_objective_sum(
            _system(), batch, state_mode="recurrent", device="cpu"
        )


def test_segment_start_rejects_reappearing_segment_identity() -> None:
    segments = torch.tensor([0, 1, 0], dtype=torch.long)
    with pytest.raises(screen.HFWorkspaceScreenError, match="not contiguous"):
        screen._segment_start(segments, 2)


def test_binding_rejects_nonfrozen_budget_and_mode() -> None:
    optimizer = TrainingRunConfig(optimizer_steps=1)
    common = dict(
        snapshot_tree_sha256="1" * 64,
        mechanics_file_sha256="2" * 64,
        stream_identity_sha256="3" * 64,
        source_manifest_sha256="4" * 64,
        training_sequences=257,
        training_utf8_bytes=1,
        optimizer=optimizer,
        code_sha256="5" * 64,
        environment_sha256="6" * 64,
    )
    with pytest.raises(screen.HFWorkspaceScreenError, match="prefix"):
        screen.make_bindings(state_mode="recurrent", **common)
    common["training_sequences"] = 256
    with pytest.raises(screen.HFWorkspaceScreenError, match="state mode"):
        screen.make_bindings(state_mode="other", **common)  # type: ignore[arg-type]
