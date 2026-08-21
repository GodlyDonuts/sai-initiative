"""Atomic, receipt-bound checkpoints for interruptible mechanics training."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sai.training.stream import CURSOR_SCHEMA, StreamCursor, TrainingStreamError

CHECKPOINT_SCHEMA = "sai-mechanics-checkpoint-v1"
MANIFEST_SCHEMA = "sai-mechanics-checkpoint-manifest-v1"


class MechanicsCheckpointError(RuntimeError):
    """A checkpoint artifact, binding, counter, or target state differs."""


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MechanicsCheckpointError(f"{field} must be a lowercase SHA256")
    return value


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MechanicsCheckpointError(f"{field} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class CheckpointBindings:
    """Immutable identities that make a checkpoint valid for exactly one run."""

    model_sha256: str
    config_sha256: str
    ordered_stream_identity_sha256: str
    code_sha256: str
    environment_sha256: str
    run_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "model_sha256",
            "config_sha256",
            "ordered_stream_identity_sha256",
            "code_sha256",
            "environment_sha256",
            "run_sha256",
        ):
            _sha256(getattr(self, field), field.replace("_", " "))

    def as_dict(self) -> dict[str, str]:
        return {
            "model_sha256": self.model_sha256,
            "config_sha256": self.config_sha256,
            "ordered_stream_identity_sha256": (self.ordered_stream_identity_sha256),
            "code_sha256": self.code_sha256,
            "environment_sha256": self.environment_sha256,
            "run_sha256": self.run_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CheckpointBindings:
        fields = {
            "model_sha256",
            "config_sha256",
            "ordered_stream_identity_sha256",
            "code_sha256",
            "environment_sha256",
            "run_sha256",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise MechanicsCheckpointError("checkpoint bindings differ")
        return cls(**value)


@dataclass(frozen=True)
class TrainingCounters:
    """Cumulative work completed before the next receipt-bound sequence."""

    optimizer_steps: int
    sequences: int
    targets: int

    def __post_init__(self) -> None:
        for field in ("optimizer_steps", "sequences", "targets"):
            _nonnegative_integer(getattr(self, field), field.replace("_", " "))

    def as_dict(self) -> dict[str, int]:
        return {
            "optimizer_steps": self.optimizer_steps,
            "sequences": self.sequences,
            "targets": self.targets,
        }

    @classmethod
    def from_dict(cls, value: Any) -> TrainingCounters:
        if not isinstance(value, dict) or set(value) != {
            "optimizer_steps",
            "sequences",
            "targets",
        }:
            raise MechanicsCheckpointError("training counters differ")
        return cls(**value)


@dataclass(frozen=True)
class RestoredCheckpoint:
    """Validated resume position and exact cumulative work counters."""

    bindings: CheckpointBindings
    counters: TrainingCounters
    cursor: StreamCursor
    checkpoint_sha256: str
    checkpoint_bytes: int


def checkpoint_manifest_path(checkpoint_path: Path) -> Path:
    """Return the mandatory sidecar path for one checkpoint artifact."""

    path = Path(checkpoint_path)
    return path.with_name(f"{path.name}.manifest.json")


def _validate_cursor(
    value: StreamCursor | dict[str, Any], bindings: CheckpointBindings
) -> StreamCursor:
    try:
        cursor = (
            value if isinstance(value, StreamCursor) else StreamCursor.from_dict(value)
        )
    except TrainingStreamError as error:
        raise MechanicsCheckpointError("stream cursor differs") from error
    if (
        cursor.schema != CURSOR_SCHEMA
        or cursor.ordered_stream_identity_sha256
        != bindings.ordered_stream_identity_sha256
        or isinstance(cursor.next_sequence, bool)
        or not isinstance(cursor.next_sequence, int)
        or cursor.next_sequence < 0
    ):
        raise MechanicsCheckpointError("stream cursor differs")
    return cursor


def _validate_position(counters: TrainingCounters, cursor: StreamCursor) -> None:
    if counters.sequences != cursor.next_sequence:
        raise MechanicsCheckpointError("sequence counter and stream cursor differ")
    if counters.optimizer_steps > counters.sequences:
        raise MechanicsCheckpointError("optimizer-step counter exceeds sequences")


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {_clone_to_cpu(key): _clone_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise MechanicsCheckpointError(
        f"checkpoint state contains unsupported type {type(value).__name__}"
    )


def _safe_target(path: Path) -> Path:
    path = Path(path)
    if not path.name or path.name in {".", ".."}:
        raise MechanicsCheckpointError("checkpoint target differs")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise MechanicsCheckpointError("checkpoint parent is missing or unsafe")
    if path.is_symlink():
        raise MechanicsCheckpointError("checkpoint target is a symlink")
    return path


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, writer: Any) -> None:
    path = _safe_target(path)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _read_regular(path: Path) -> tuple[io.BufferedReader, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise MechanicsCheckpointError(
            "checkpoint artifact is missing or unsafe"
        ) from error
    handle = os.fdopen(descriptor, "rb")
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        handle.close()
        raise MechanicsCheckpointError(
            "checkpoint artifact is not a unique regular file"
        )
    return handle, metadata


def _sha256_handle(handle: io.BufferedReader) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _read_manifest(path: Path) -> dict[str, Any]:
    with _read_regular(path)[0] as handle:
        try:
            value = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise MechanicsCheckpointError(
                "checkpoint manifest is unreadable"
            ) from error
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "checkpoint",
        "bindings",
        "counters",
        "cursor",
    }:
        raise MechanicsCheckpointError("checkpoint manifest differs")
    if value["schema"] != MANIFEST_SCHEMA:
        raise MechanicsCheckpointError("checkpoint manifest schema differs")
    descriptor = value["checkpoint"]
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise MechanicsCheckpointError("checkpoint artifact descriptor differs")
    _sha256(descriptor.get("sha256"), "checkpoint artifact SHA256")
    if (
        isinstance(descriptor.get("bytes"), bool)
        or not isinstance(descriptor.get("bytes"), int)
        or descriptor["bytes"] <= 0
    ):
        raise MechanicsCheckpointError("checkpoint artifact bytes differ")
    return value


def _manifest(
    checkpoint_path: Path,
    *,
    checkpoint_bytes: int,
    checkpoint_sha256: str,
    bindings: CheckpointBindings,
    counters: TrainingCounters,
    cursor: StreamCursor,
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "checkpoint": {
            "path": checkpoint_path.name,
            "bytes": checkpoint_bytes,
            "sha256": checkpoint_sha256,
        },
        "bindings": bindings.as_dict(),
        "counters": counters.as_dict(),
        "cursor": cursor.as_dict(),
    }


def save_mechanics_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    bindings: CheckpointBindings,
    counters: TrainingCounters,
    cursor: StreamCursor,
) -> dict[str, Any]:
    """Durably replace a checkpoint, publishing its manifest only when complete."""

    if not isinstance(model, nn.Module):
        raise MechanicsCheckpointError("checkpoint model type differs")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise MechanicsCheckpointError("checkpoint optimizer type differs")
    if not isinstance(bindings, CheckpointBindings):
        raise MechanicsCheckpointError("checkpoint bindings type differs")
    if not isinstance(counters, TrainingCounters):
        raise MechanicsCheckpointError("training counters type differs")
    cursor = _validate_cursor(cursor, bindings)
    _validate_position(counters, cursor)
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_rng_states = torch.cuda.get_rng_state_all() if cuda_available else []
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "bindings": bindings.as_dict(),
        "counters": counters.as_dict(),
        "cursor": cursor.as_dict(),
        "model_state_dict": _clone_to_cpu(model.state_dict()),
        "optimizer_state_dict": _clone_to_cpu(optimizer.state_dict()),
        "cpu_rng_state": torch.random.get_rng_state().clone(),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_rng_states": _clone_to_cpu(cuda_rng_states),
    }
    checkpoint_path = _safe_target(Path(checkpoint_path))
    _atomic_write(checkpoint_path, lambda handle: torch.save(payload, handle))
    with _read_regular(checkpoint_path)[0] as handle:
        checkpoint_sha256 = _sha256_handle(handle)
        checkpoint_bytes = os.fstat(handle.fileno()).st_size
    manifest = _manifest(
        checkpoint_path,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=checkpoint_sha256,
        bindings=bindings,
        counters=counters,
        cursor=cursor,
    )
    encoded = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    _atomic_write(
        checkpoint_manifest_path(checkpoint_path), lambda handle: handle.write(encoded)
    )
    return manifest


def _validate_model_state(model: nn.Module, saved: Any) -> None:
    current = model.state_dict()
    if not isinstance(saved, dict) or set(saved) != set(current):
        raise MechanicsCheckpointError("checkpoint model state membership differs")
    for name, target in current.items():
        source = saved[name]
        if (
            not isinstance(source, torch.Tensor)
            or source.shape != target.shape
            or source.dtype != target.dtype
        ):
            raise MechanicsCheckpointError(f"checkpoint model tensor {name} differs")


def _validate_optimizer_state(optimizer: torch.optim.Optimizer, saved: Any) -> None:
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise MechanicsCheckpointError("checkpoint optimizer state differs")
    current_groups = optimizer.state_dict()["param_groups"]
    saved_groups = saved["param_groups"]
    if not isinstance(saved_groups, list) or len(saved_groups) != len(current_groups):
        raise MechanicsCheckpointError("checkpoint optimizer groups differ")
    for current, candidate in zip(current_groups, saved_groups, strict=True):
        if not isinstance(candidate, dict) or not isinstance(
            candidate.get("params"), list
        ):
            raise MechanicsCheckpointError("checkpoint optimizer group differs")
        if len(candidate["params"]) != len(current["params"]):
            raise MechanicsCheckpointError("checkpoint optimizer parameters differ")
    if not isinstance(saved["state"], dict):
        raise MechanicsCheckpointError("checkpoint optimizer tensors differ")


def load_mechanics_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_bindings: CheckpointBindings,
) -> RestoredCheckpoint:
    """Validate every receipt and target shape before restoring state and RNG."""

    if not isinstance(model, nn.Module):
        raise MechanicsCheckpointError("checkpoint model type differs")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise MechanicsCheckpointError("checkpoint optimizer type differs")
    if not isinstance(expected_bindings, CheckpointBindings):
        raise MechanicsCheckpointError("expected bindings type differs")
    checkpoint_path = Path(checkpoint_path)
    manifest = _read_manifest(checkpoint_manifest_path(checkpoint_path))
    descriptor = manifest["checkpoint"]
    if descriptor["path"] != checkpoint_path.name:
        raise MechanicsCheckpointError("checkpoint artifact path differs")
    bindings = CheckpointBindings.from_dict(manifest["bindings"])
    if bindings != expected_bindings:
        raise MechanicsCheckpointError("checkpoint bindings do not match this run")
    counters = TrainingCounters.from_dict(manifest["counters"])
    cursor = _validate_cursor(manifest["cursor"], bindings)
    _validate_position(counters, cursor)

    with _read_regular(checkpoint_path)[0] as handle:
        actual_bytes = os.fstat(handle.fileno()).st_size
        actual_sha256 = _sha256_handle(handle)
        if actual_bytes != descriptor["bytes"] or actual_sha256 != descriptor["sha256"]:
            raise MechanicsCheckpointError("checkpoint artifact differs from manifest")
        try:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception as error:
            raise MechanicsCheckpointError(
                "checkpoint payload is unreadable"
            ) from error

    expected_keys = {
        "schema",
        "bindings",
        "counters",
        "cursor",
        "model_state_dict",
        "optimizer_state_dict",
        "cpu_rng_state",
        "cuda_available",
        "cuda_device_count",
        "cuda_rng_states",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise MechanicsCheckpointError("checkpoint payload differs")
    if payload["schema"] != CHECKPOINT_SCHEMA:
        raise MechanicsCheckpointError("checkpoint payload schema differs")
    if (
        payload["bindings"] != manifest["bindings"]
        or payload["counters"] != manifest["counters"]
        or payload["cursor"] != manifest["cursor"]
    ):
        raise MechanicsCheckpointError("checkpoint payload and manifest differ")
    cpu_rng_state = payload["cpu_rng_state"]
    if (
        not isinstance(cpu_rng_state, torch.Tensor)
        or cpu_rng_state.dtype != torch.uint8
        or cpu_rng_state.ndim != 1
    ):
        raise MechanicsCheckpointError("CPU RNG state differs")
    cuda_available = payload["cuda_available"]
    cuda_device_count = payload["cuda_device_count"]
    cuda_rng_states = payload["cuda_rng_states"]
    if (
        not isinstance(cuda_available, bool)
        or isinstance(cuda_device_count, bool)
        or not isinstance(cuda_device_count, int)
        or cuda_device_count < 0
        or not isinstance(cuda_rng_states, list)
        or cuda_available != torch.cuda.is_available()
        or cuda_device_count != (torch.cuda.device_count() if cuda_available else 0)
        or len(cuda_rng_states) != cuda_device_count
        or any(
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            for state in cuda_rng_states
        )
    ):
        raise MechanicsCheckpointError("CUDA RNG topology or state differs")
    _validate_model_state(model, payload["model_state_dict"])
    _validate_optimizer_state(optimizer, payload["optimizer_state_dict"])

    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    torch.random.set_rng_state(cpu_rng_state)
    if cuda_available:
        torch.cuda.set_rng_state_all(cuda_rng_states)
    return RestoredCheckpoint(
        bindings=bindings,
        counters=counters,
        cursor=cursor,
        checkpoint_sha256=actual_sha256,
        checkpoint_bytes=actual_bytes,
    )
