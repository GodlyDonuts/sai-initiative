"""Create-only model snapshots at predeclared training milestones."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sai.data.token_stream import canonical_sha256
from sai.training.checkpoint import CheckpointBindings, TrainingCounters
from sai.training.stream import StreamCursor

SCHEMA = "sai-model-milestone-snapshot-v1"


class MilestoneSnapshotError(RuntimeError):
    """A milestone step, artifact, model state, or run binding differs."""


def parse_milestone_steps(value: str | None, *, maximum_step: int) -> tuple[int, ...]:
    """Parse one prospective, strictly increasing milestone-step list."""

    if (
        isinstance(maximum_step, bool)
        or not isinstance(maximum_step, int)
        or maximum_step <= 0
    ):
        raise MilestoneSnapshotError("milestone optimizer budget differs")
    if value is None or value == "":
        return ()
    if not isinstance(value, str):
        raise MilestoneSnapshotError("milestone steps differ")
    if "," in value and ":" in value:
        raise MilestoneSnapshotError("milestone steps differ")
    fields = value.split(":" if ":" in value else ",")
    if any(
        not field or not field.isascii() or not field.isdecimal() for field in fields
    ):
        raise MilestoneSnapshotError("milestone steps differ")
    steps = tuple(int(field) for field in fields)
    if (
        any(step <= 0 or step >= maximum_step for step in steps)
        or tuple(sorted(set(steps))) != steps
    ):
        raise MilestoneSnapshotError("milestone steps differ")
    return steps


def milestone_root(checkpoint: Path) -> Path:
    """Return the relocation-safe sibling directory for one run's snapshots."""

    checkpoint = Path(checkpoint)
    return checkpoint.with_name(f"{checkpoint.name}.milestones")


def milestone_path(root: Path, step: int) -> Path:
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise MilestoneSnapshotError("milestone step differs")
    return Path(root) / f"step-{step:06d}.model.pt"


def prepare_milestone_root(checkpoint: Path, *, resume: bool) -> Path:
    """Create a new snapshot root, or reopen the exact root during resume."""

    root = milestone_root(checkpoint)
    if root.is_symlink():
        raise MilestoneSnapshotError("milestone root is unsafe")
    if resume:
        if not root.is_dir():
            raise MilestoneSnapshotError("milestone root is missing")
    else:
        try:
            root.mkdir(mode=0o700)
        except FileExistsError as error:
            raise MilestoneSnapshotError("milestone root already exists") from error
    return root


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    if not isinstance(model, nn.Module):
        raise MilestoneSnapshotError("milestone model differs")
    state = model.state_dict()
    if not state:
        raise MilestoneSnapshotError("milestone model state is empty")
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash one exact, sorted tensor-state projection."""

    if not isinstance(state, dict) or not state:
        raise MilestoneSnapshotError("milestone model state differs")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise MilestoneSnapshotError("milestone model state differs")
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _artifact_sha256(handle) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1 << 20):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _read_artifact(path: Path) -> tuple[dict[str, Any], int, str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise MilestoneSnapshotError(
            "milestone artifact is missing or unsafe"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
        ):
            raise MilestoneSnapshotError("milestone artifact is missing or unsafe")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            artifact_sha256 = _artifact_sha256(handle)
            try:
                payload = torch.load(handle, map_location="cpu", weights_only=True)
            except Exception as error:
                raise MilestoneSnapshotError(
                    "milestone artifact is unreadable"
                ) from error
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    def identity(row):
        return (
            row.st_dev,
            row.st_ino,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
        )

    if identity(before) != identity(after):
        raise MilestoneSnapshotError("milestone artifact changed while reading")
    return payload, before.st_size, artifact_sha256


def validate_milestone(
    path: Path,
    *,
    expected_bindings: CheckpointBindings,
    expected_step: int,
    expected_state_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay one model-only milestone artifact and return its descriptor."""

    payload, size, artifact_sha256 = _read_artifact(path)
    expected_keys = {
        "schema",
        "bindings",
        "counters",
        "cursor",
        "optimizer_step",
        "model_state_sha256",
        "model_state_dict",
        "scope",
        "receipt_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise MilestoneSnapshotError("milestone payload fields differ")
    try:
        bindings = CheckpointBindings.from_dict(payload["bindings"])
        counters = TrainingCounters.from_dict(payload["counters"])
        cursor = StreamCursor.from_dict(payload["cursor"])
    except Exception as error:
        raise MilestoneSnapshotError("milestone lineage differs") from error
    state = payload["model_state_dict"]
    observed_state_sha256 = state_sha256(state)
    unsigned = {
        key: value
        for key, value in payload.items()
        if key not in {"model_state_dict", "receipt_sha256"}
    }
    if (
        payload["schema"] != SCHEMA
        or bindings != expected_bindings
        or payload["optimizer_step"] != expected_step
        or counters.optimizer_steps != expected_step
        or counters.sequences != cursor.next_sequence
        or cursor.ordered_stream_identity_sha256
        != expected_bindings.ordered_stream_identity_sha256
        or payload["model_state_sha256"] != observed_state_sha256
        or payload["scope"] != "evaluation_only_model_state_without_optimizer_or_rng"
        or payload["receipt_sha256"] != canonical_sha256(unsigned)
        or (
            expected_state_sha256 is not None
            and observed_state_sha256 != expected_state_sha256
        )
    ):
        raise MilestoneSnapshotError("milestone lineage differs")
    return {
        "path": path.name,
        "bytes": size,
        "sha256": artifact_sha256,
        "optimizer_step": expected_step,
        "sequences": counters.sequences,
        "targets": counters.targets,
        "model_state_sha256": observed_state_sha256,
    }


def load_validated_milestone_state(
    path: Path,
    *,
    model: nn.Module,
    expected_bindings: CheckpointBindings,
    expected_descriptor: dict[str, Any],
) -> dict[str, Any]:
    """Load only one exact, receipt-bound model state into an instantiated model."""

    if not isinstance(expected_descriptor, dict) or set(expected_descriptor) != {
        "path",
        "bytes",
        "sha256",
        "optimizer_step",
        "sequences",
        "targets",
        "model_state_sha256",
    }:
        raise MilestoneSnapshotError("milestone descriptor differs")
    step = expected_descriptor["optimizer_step"]
    if (
        expected_descriptor["path"] != Path(path).name
        or isinstance(expected_descriptor["bytes"], bool)
        or not isinstance(expected_descriptor["bytes"], int)
        or expected_descriptor["bytes"] <= 0
        or not isinstance(expected_descriptor["sha256"], str)
        or len(expected_descriptor["sha256"]) != 64
        or isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
    ):
        raise MilestoneSnapshotError("milestone descriptor differs")
    payload, size, artifact_sha256 = _read_artifact(path)
    observed_descriptor = validate_milestone(
        path,
        expected_bindings=expected_bindings,
        expected_step=step,
        expected_state_sha256=expected_descriptor["model_state_sha256"],
    )
    if (
        observed_descriptor != expected_descriptor
        or size != expected_descriptor["bytes"]
        or artifact_sha256 != expected_descriptor["sha256"]
    ):
        raise MilestoneSnapshotError("milestone descriptor differs")
    saved = payload.get("model_state_dict")
    current = model.state_dict()
    if not isinstance(saved, dict) or set(saved) != set(current):
        raise MilestoneSnapshotError("milestone model state differs")
    for name, target in current.items():
        source = saved[name]
        if (
            not isinstance(source, torch.Tensor)
            or source.shape != target.shape
            or source.dtype != target.dtype
        ):
            raise MilestoneSnapshotError(f"milestone model tensor {name} differs")
    model.load_state_dict(saved, strict=True)
    if state_sha256(model.state_dict()) != expected_descriptor["model_state_sha256"]:
        raise MilestoneSnapshotError("milestone model state differs")
    return observed_descriptor


def publish_or_validate_milestone(
    root: Path,
    *,
    model: nn.Module,
    bindings: CheckpointBindings,
    counters: TrainingCounters,
    cursor: StreamCursor,
) -> dict[str, Any]:
    """Atomically publish one snapshot or prove an identical resumed artifact."""

    if counters.optimizer_steps <= 0 or counters.sequences != cursor.next_sequence:
        raise MilestoneSnapshotError("milestone position differs")
    path = milestone_path(root, counters.optimizer_steps)
    state = _clone_state(model)
    state_identity = state_sha256(state)
    if path.exists() or path.is_symlink():
        return validate_milestone(
            path,
            expected_bindings=bindings,
            expected_step=counters.optimizer_steps,
            expected_state_sha256=state_identity,
        )
    payload = {
        "schema": SCHEMA,
        "bindings": bindings.as_dict(),
        "counters": counters.as_dict(),
        "cursor": cursor.as_dict(),
        "optimizer_step": counters.optimizer_steps,
        "model_state_sha256": state_identity,
        "model_state_dict": state,
        "scope": "evaluation_only_model_state_without_optimizer_or_rng",
    }
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "model_state_dict"}
    )
    temporary = Path(root) / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return validate_milestone(
        path,
        expected_bindings=bindings,
        expected_step=counters.optimizer_steps,
        expected_state_sha256=state_identity,
    )


def validate_milestone_population(
    root: Path,
    *,
    expected_steps: tuple[int, ...],
    expected_bindings: CheckpointBindings,
    maximum_completed_step: int,
) -> list[dict[str, Any]]:
    """Require exactly the milestone files that should exist at one position."""

    if not Path(root).is_dir() or Path(root).is_symlink():
        raise MilestoneSnapshotError("milestone root is missing or unsafe")
    expected_existing = tuple(
        step for step in expected_steps if step <= maximum_completed_step
    )
    expected_names = {milestone_path(root, step).name for step in expected_existing}
    observed_names = set()
    for entry in os.scandir(root):
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise MilestoneSnapshotError("milestone root membership differs")
        observed_names.add(entry.name)
    if observed_names != expected_names:
        raise MilestoneSnapshotError("milestone root membership differs")
    return [
        validate_milestone(
            milestone_path(root, step),
            expected_bindings=expected_bindings,
            expected_step=step,
        )
        for step in expected_existing
    ]
