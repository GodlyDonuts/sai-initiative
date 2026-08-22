from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sai.training.checkpoint import CheckpointBindings, TrainingCounters
from sai.training.milestone import (
    MilestoneSnapshotError,
    load_validated_milestone_state,
    milestone_path,
    parse_milestone_steps,
    prepare_milestone_root,
    publish_or_validate_milestone,
    state_sha256,
    validate_milestone,
    validate_milestone_population,
)
from sai.training.stream import CURSOR_SCHEMA, StreamCursor


def _bindings() -> CheckpointBindings:
    return CheckpointBindings(
        model_sha256="1" * 64,
        config_sha256="2" * 64,
        ordered_stream_identity_sha256="3" * 64,
        code_sha256="4" * 64,
        environment_sha256="5" * 64,
        run_sha256="6" * 64,
    )


def _cursor(step: int) -> StreamCursor:
    return StreamCursor(
        schema=CURSOR_SCHEMA,
        ordered_stream_identity_sha256="3" * 64,
        next_sequence=step * 8,
    )


def test_parses_only_prospective_unique_nonterminal_steps() -> None:
    assert parse_milestone_steps("2,5,9", maximum_step=10) == (2, 5, 9)
    assert parse_milestone_steps("2:5:9", maximum_step=10) == (2, 5, 9)
    assert parse_milestone_steps("", maximum_step=10) == ()
    for value in ("0", "2,2", "5,2", "2,10", "2,,5", "2:5,9", " 2"):
        with pytest.raises(MilestoneSnapshotError, match="steps differ"):
            parse_milestone_steps(value, maximum_step=10)


def test_publishes_and_replays_one_create_only_model_snapshot(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Linear(5, 2))
    checkpoint = tmp_path / "run.checkpoint.pt"
    root = prepare_milestone_root(checkpoint, resume=False)
    counters = TrainingCounters(optimizer_steps=2, sequences=16, targets=29)
    first = publish_or_validate_milestone(
        root,
        model=model,
        bindings=_bindings(),
        counters=counters,
        cursor=_cursor(2),
    )
    assert first["optimizer_step"] == 2
    assert first["sequences"] == 16
    assert first["targets"] == 29
    assert first["bytes"] > 0
    assert len(first["sha256"]) == 64
    assert len(first["model_state_sha256"]) == 64
    assert (
        validate_milestone(
            milestone_path(root, 2),
            expected_bindings=_bindings(),
            expected_step=2,
        )
        == first
    )
    assert (
        publish_or_validate_milestone(
            root,
            model=model,
            bindings=_bindings(),
            counters=counters,
            cursor=_cursor(2),
        )
        == first
    )

    restored = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Linear(5, 2))
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
    assert (
        load_validated_milestone_state(
            milestone_path(root, 2),
            model=restored,
            expected_bindings=_bindings(),
            expected_descriptor=first,
        )
        == first
    )
    assert state_sha256(restored.state_dict()) == first["model_state_sha256"]

    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(MilestoneSnapshotError, match="lineage differs"):
        publish_or_validate_milestone(
            root,
            model=model,
            bindings=_bindings(),
            counters=counters,
            cursor=_cursor(2),
        )


def test_population_requires_exact_completed_membership(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    root = prepare_milestone_root(tmp_path / "run.pt", resume=False)
    publish_or_validate_milestone(
        root,
        model=model,
        bindings=_bindings(),
        counters=TrainingCounters(2, 16, 20),
        cursor=_cursor(2),
    )
    rows = validate_milestone_population(
        root,
        expected_steps=(2, 4),
        expected_bindings=_bindings(),
        maximum_completed_step=2,
    )
    assert [row["optimizer_step"] for row in rows] == [2]

    (root / "undeclared.pt").write_bytes(b"drift")
    with pytest.raises(MilestoneSnapshotError, match="membership differs"):
        validate_milestone_population(
            root,
            expected_steps=(2, 4),
            expected_bindings=_bindings(),
            maximum_completed_step=2,
        )


def test_rejects_rewritten_snapshot_metadata_and_tensor_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    root = prepare_milestone_root(tmp_path / "run.pt", resume=False)
    publish_or_validate_milestone(
        root,
        model=model,
        bindings=_bindings(),
        counters=TrainingCounters(2, 16, 20),
        cursor=_cursor(2),
    )
    path = milestone_path(root, 2)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["scope"] = "rewritten"
    torch.save(payload, path)
    with pytest.raises(MilestoneSnapshotError, match="lineage differs"):
        validate_milestone(
            path,
            expected_bindings=_bindings(),
            expected_step=2,
        )


def test_resume_requires_a_real_existing_root(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run.pt"
    with pytest.raises(MilestoneSnapshotError, match="missing"):
        prepare_milestone_root(checkpoint, resume=True)
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "run.pt.milestones"
    root.symlink_to(target, target_is_directory=True)
    with pytest.raises(MilestoneSnapshotError, match="unsafe"):
        prepare_milestone_root(checkpoint, resume=True)
