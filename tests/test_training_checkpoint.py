from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from sai.training.checkpoint import (
    CheckpointBindings,
    MechanicsCheckpointError,
    TrainingCounters,
    checkpoint_manifest_path,
    load_mechanics_checkpoint,
    save_mechanics_checkpoint,
)
from sai.training.stream import StreamCursor


def bindings(salt: int = 0) -> CheckpointBindings:
    characters = "123456"
    return CheckpointBindings(
        model_sha256=characters[(salt + 0) % len(characters)] * 64,
        config_sha256=characters[(salt + 1) % len(characters)] * 64,
        ordered_stream_identity_sha256=characters[(salt + 2) % len(characters)] * 64,
        code_sha256=characters[(salt + 3) % len(characters)] * 64,
        environment_sha256=characters[(salt + 4) % len(characters)] * 64,
        run_sha256=characters[(salt + 5) % len(characters)] * 64,
    )


def step(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    inputs = torch.randn(3, 4)
    targets = torch.randn(3, 2)
    optimizer.zero_grad(set_to_none=True)
    loss = (model(inputs) - targets).square().mean()
    loss.backward()
    optimizer.step()


def assert_state_equal(left: nn.Module, right: nn.Module) -> None:
    assert set(left.state_dict()) == set(right.state_dict())
    for name, tensor in left.state_dict().items():
        assert torch.equal(tensor, right.state_dict()[name]), name


def test_interrupted_cpu_trajectory_exactly_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    identity = bindings()
    torch.manual_seed(20260821)
    uninterrupted = nn.Sequential(nn.Linear(4, 5), nn.GELU(), nn.Linear(5, 2))
    optimizer = torch.optim.AdamW(uninterrupted.parameters(), lr=1e-3)
    step(uninterrupted, optimizer)
    checkpoint = tmp_path / "mechanics.pt"
    manifest = save_mechanics_checkpoint(
        checkpoint,
        model=uninterrupted,
        optimizer=optimizer,
        bindings=identity,
        counters=TrainingCounters(optimizer_steps=1, sequences=2, targets=10),
        cursor=StreamCursor(identity.ordered_stream_identity_sha256, 2),
    )
    step(uninterrupted, optimizer)

    resumed = nn.Sequential(nn.Linear(4, 5), nn.GELU(), nn.Linear(5, 2))
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    restored = load_mechanics_checkpoint(
        checkpoint,
        model=resumed,
        optimizer=resumed_optimizer,
        expected_bindings=identity,
    )
    step(resumed, resumed_optimizer)

    assert_state_equal(uninterrupted, resumed)
    assert restored.counters == TrainingCounters(1, 2, 10)
    assert restored.cursor == StreamCursor(identity.ordered_stream_identity_sha256, 2)
    assert restored.checkpoint_sha256 == manifest["checkpoint"]["sha256"]
    assert restored.checkpoint_bytes == manifest["checkpoint"]["bytes"]
    assert not list(tmp_path.glob(".*.tmp"))


def test_checkpoint_byte_mutation_fails_before_target_mutation(tmp_path: Path) -> None:
    identity = bindings()
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "mechanics.pt"
    save_mechanics_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        bindings=identity,
        counters=TrainingCounters(0, 0, 0),
        cursor=StreamCursor(identity.ordered_stream_identity_sha256, 0),
    )
    target = nn.Linear(3, 2)
    target_optimizer = torch.optim.AdamW(target.parameters())
    before = {name: tensor.clone() for name, tensor in target.state_dict().items()}
    payload = bytearray(checkpoint.read_bytes())
    payload[len(payload) // 2] ^= 1
    checkpoint.write_bytes(payload)

    with pytest.raises(MechanicsCheckpointError, match="artifact differs"):
        load_mechanics_checkpoint(
            checkpoint,
            model=target,
            optimizer=target_optimizer,
            expected_bindings=identity,
        )
    assert all(
        torch.equal(before[name], tensor)
        for name, tensor in target.state_dict().items()
    )


def test_binding_and_manifest_mutations_fail_closed(tmp_path: Path) -> None:
    identity = bindings()
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "mechanics.pt"
    save_mechanics_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        bindings=identity,
        counters=TrainingCounters(0, 0, 0),
        cursor=StreamCursor(identity.ordered_stream_identity_sha256, 0),
    )
    with pytest.raises(MechanicsCheckpointError, match="bindings"):
        load_mechanics_checkpoint(
            checkpoint,
            model=nn.Linear(3, 2),
            optimizer=torch.optim.AdamW(nn.Linear(3, 2).parameters()),
            expected_bindings=bindings(1),
        )

    sidecar = checkpoint_manifest_path(checkpoint)
    manifest = json.loads(sidecar.read_text())
    manifest["counters"]["targets"] = 1
    sidecar.write_text(json.dumps(manifest))
    target = nn.Linear(3, 2)
    with pytest.raises(MechanicsCheckpointError, match="payload and manifest"):
        load_mechanics_checkpoint(
            checkpoint,
            model=target,
            optimizer=torch.optim.AdamW(target.parameters()),
            expected_bindings=identity,
        )


def test_cursor_and_counter_invariants_fail_before_writing(tmp_path: Path) -> None:
    identity = bindings()
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "mechanics.pt"
    with pytest.raises(MechanicsCheckpointError, match="counter and stream cursor"):
        save_mechanics_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            bindings=identity,
            counters=TrainingCounters(optimizer_steps=1, sequences=2, targets=4),
            cursor=StreamCursor(identity.ordered_stream_identity_sha256, 3),
        )
    assert not checkpoint.exists()
    assert not checkpoint_manifest_path(checkpoint).exists()


def test_wrong_model_geometry_fails_before_target_mutation(tmp_path: Path) -> None:
    identity = bindings()
    source = nn.Linear(3, 2)
    checkpoint = tmp_path / "mechanics.pt"
    save_mechanics_checkpoint(
        checkpoint,
        model=source,
        optimizer=torch.optim.AdamW(source.parameters()),
        bindings=identity,
        counters=TrainingCounters(0, 0, 0),
        cursor=StreamCursor(identity.ordered_stream_identity_sha256, 0),
    )
    target = nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(target.parameters())
    before = target.weight.detach().clone()
    with pytest.raises(MechanicsCheckpointError, match="model tensor"):
        load_mechanics_checkpoint(
            checkpoint,
            model=target,
            optimizer=optimizer,
            expected_bindings=identity,
        )
    assert torch.equal(before, target.weight)
