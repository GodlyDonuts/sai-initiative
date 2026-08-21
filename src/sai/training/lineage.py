"""Replay completed training lineage before any result can use a checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "sai-completed-run-lineage-receipt-v1"
AUTHORIZATION_SCHEMA = "sai-training-authorization-receipt-v1"
PLAN_SCHEMA = "sai-300m-adaptive-experiment-plan-v1"
ROLES = ("workspace_treatment", "equal_flop_fast_control")
TOP_LEVEL_KEYS = {
    "schema",
    "status",
    "scientific_status",
    "training_authorized",
    "official_training_order_received",
    "terminal_public_board_accessed",
    "created_at_utc",
    "role",
    "run_identity_sha256",
    "comparison_group_sha256",
    "changed_factor",
    "plan",
    "authorization",
    "parent",
    "immutable_inputs",
    "execution",
    "checkpoint_tree",
    "state_projections",
    "receipt_sha256",
}


class CompletedRunLineageError(RuntimeError):
    """A completed run cannot be replayed exactly."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CompletedRunLineageError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise CompletedRunLineageError(f"{field} differs") from error
    return value


def _positive_integer(value: Any, field: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CompletedRunLineageError(f"{field} differs")
    return value


def _finite(value: Any, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompletedRunLineageError(f"{field} differs")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise CompletedRunLineageError(f"{field} differs")
    return result


def _relative_path(value: Any, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise CompletedRunLineageError(f"{field} differs")
    result = PurePosixPath(value)
    if result.is_absolute() or ".." in result.parts or "." in result.parts:
        raise CompletedRunLineageError(f"{field} is unsafe")
    return result


def _safe_file(root: Path, relative: PurePosixPath, field: str) -> Path:
    path = root.joinpath(*relative.parts)
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise CompletedRunLineageError(f"{field} is missing or unsafe")
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if resolved.parent != resolved_root and resolved_root not in resolved.parents:
        raise CompletedRunLineageError(f"{field} escapes the artifact root")
    return path


def _load_json_artifact(
    descriptor: Any, root: Path, field: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise CompletedRunLineageError(f"{field} artifact descriptor differs")
    relative = _relative_path(descriptor.get("path"), field)
    path = _safe_file(root, relative, field)
    size = _positive_integer(descriptor.get("bytes"), f"{field} bytes")
    digest = _sha256(descriptor.get("sha256"), f"{field} SHA256")
    if path.stat().st_size != size or sha256_file(path) != digest:
        raise CompletedRunLineageError(f"{field} artifact differs")
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompletedRunLineageError(f"{field} artifact is unreadable") from error
    if not isinstance(payload, dict):
        raise CompletedRunLineageError(f"{field} artifact must be an object")
    return payload, {"path": str(relative), "bytes": size, "sha256": digest}


def _validate_plan(
    descriptor: Any, root: Path, run_identity: str, comparison_group: str, role: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, artifact = _load_json_artifact(descriptor, root, "plan")
    if (
        payload.get("schema") != PLAN_SCHEMA
        or payload.get("status") != "frozen"
        or payload.get("training_authorized") is not False
        or payload.get("official_training_order_received") is not False
    ):
        raise CompletedRunLineageError("prospective plan boundary differs")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs:
        raise CompletedRunLineageError("prospective plan runs are missing")
    matches = [row for row in runs if row.get("run_identity_sha256") == run_identity]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise CompletedRunLineageError("completed run is absent from the plan")
    planned = matches[0]
    required_planned = {
        "run_identity_sha256",
        "comparison_group_sha256",
        "role",
        "changed_factor",
        "scale",
        "mixer_family",
        "contrast",
        "seed",
        "parent_completed_run_receipt_sha256",
        "parent_checkpoint_tree_sha256",
        "parent_fast_path_state_sha256",
        "training_budget",
        "tokenizer_sha256",
        "ordered_stream_sha256",
        "environment_sha256",
        "system_config_sha256",
        "workspace_plan_sha256",
        "workspace_candidate_identity_sha256",
    }
    if (
        set(planned) != required_planned
        or planned.get("comparison_group_sha256") != comparison_group
        or planned.get("role") != role
        or planned.get("scale") != "300m"
        or not isinstance(planned.get("mixer_family"), str)
        or not planned["mixer_family"]
        or planned.get("contrast") != "iso_flop"
        or isinstance(planned.get("seed"), bool)
        or not isinstance(planned.get("seed"), int)
        or planned["seed"] < 0
    ):
        raise CompletedRunLineageError("planned comparison role differs")
    if payload.get("plan_sha256") != canonical_sha256(
        {key: value for key, value in payload.items() if key != "plan_sha256"}
    ):
        raise CompletedRunLineageError("prospective plan identity differs")
    return planned, artifact


def _validate_authorization(
    descriptor: Any, root: Path, plan_sha256: str, run_identity: str
) -> dict[str, Any]:
    payload, artifact = _load_json_artifact(descriptor, root, "authorization")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != AUTHORIZATION_SCHEMA
        or payload.get("status") != "authorized"
        or payload.get("official_training_order_received") is not True
        or payload.get("training_authorized") is not True
        or payload.get("terminal_public_board_accessed") is not False
        or payload.get("plan_sha256") != plan_sha256
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise CompletedRunLineageError("training authorization differs")
    authorized = payload.get("authorized_run_identities")
    if (
        not isinstance(authorized, list)
        or len(authorized) != len(set(authorized))
        or run_identity not in authorized
        or any(_sha256(item, "authorized run identity") != item for item in authorized)
    ):
        raise CompletedRunLineageError("completed run was not authorized")
    return artifact


def _validate_checkpoint_tree(payload: Any, root: Path) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "root",
        "format",
        "files",
        "file_count",
        "total_bytes",
        "tree_sha256",
    }:
        raise CompletedRunLineageError("checkpoint tree descriptor differs")
    relative_root = _relative_path(payload.get("root"), "checkpoint root")
    checkpoint_root = root.joinpath(*relative_root.parts)
    if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
        raise CompletedRunLineageError("checkpoint root is missing or unsafe")
    resolved_root = root.resolve(strict=True)
    resolved_checkpoint = checkpoint_root.resolve(strict=True)
    if (
        resolved_checkpoint != resolved_root
        and resolved_root not in resolved_checkpoint.parents
    ):
        raise CompletedRunLineageError("checkpoint root escapes the artifact root")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise CompletedRunLineageError("checkpoint manifest is empty")
    normalized = []
    names = set()
    for row in files:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise CompletedRunLineageError("checkpoint member differs")
        relative = _relative_path(row.get("path"), "checkpoint member")
        name = str(relative)
        if name in names:
            raise CompletedRunLineageError("checkpoint member is duplicated")
        names.add(name)
        member = _safe_file(checkpoint_root, relative, "checkpoint member")
        size = _positive_integer(row.get("bytes"), "checkpoint member bytes")
        digest = _sha256(row.get("sha256"), "checkpoint member SHA256")
        if member.stat().st_size != size or sha256_file(member) != digest:
            raise CompletedRunLineageError("checkpoint member differs")
        normalized.append({"path": name, "bytes": size, "sha256": digest})
    normalized.sort(key=lambda row: row["path"])
    actual = set()
    for candidate in checkpoint_root.rglob("*"):
        if candidate.is_symlink():
            raise CompletedRunLineageError("checkpoint tree contains a symlink")
        if candidate.is_file():
            if candidate.stat().st_nlink != 1:
                raise CompletedRunLineageError("checkpoint member has multiple links")
            actual.add(candidate.relative_to(checkpoint_root).as_posix())
        elif not candidate.is_dir():
            raise CompletedRunLineageError("checkpoint tree contains a special file")
    if actual != names:
        raise CompletedRunLineageError("checkpoint tree membership differs")
    total = sum(row["bytes"] for row in normalized)
    if (
        payload.get("format") not in {"safetensors", "pytorch_state_dict"}
        or payload.get("file_count") != len(normalized)
        or payload.get("total_bytes") != total
        or payload.get("tree_sha256") != canonical_sha256(normalized)
    ):
        raise CompletedRunLineageError("checkpoint tree receipt differs")
    return {
        "root": str(relative_root),
        "format": payload["format"],
        "files": normalized,
        "file_count": len(normalized),
        "total_bytes": total,
        "tree_sha256": payload["tree_sha256"],
    }


def _validate_state_projection(
    descriptor: Any, root: Path, component: str
) -> dict[str, Any]:
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "bytes",
        "sha256",
        "state_sha256",
    }:
        raise CompletedRunLineageError("state projection descriptor differs")
    artifact_descriptor = {key: descriptor[key] for key in ("path", "bytes", "sha256")}
    payload, artifact = _load_json_artifact(
        artifact_descriptor, root, f"{component} state projection"
    )
    if (
        set(payload) != {"schema", "status", "component", "tensors", "state_sha256"}
        or payload.get("schema") != "sai-tensor-state-projection-v1"
        or payload.get("status") != "complete"
        or payload.get("component") != component
    ):
        raise CompletedRunLineageError("state projection identity differs")
    tensors = payload.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise CompletedRunLineageError("state projection tensors are missing")
    normalized = []
    names = set()
    for tensor in tensors:
        if not isinstance(tensor, dict) or set(tensor) != {
            "name",
            "dtype",
            "shape",
            "raw_little_endian_sha256",
        }:
            raise CompletedRunLineageError("state projection tensor differs")
        name = tensor.get("name")
        dtype = tensor.get("dtype")
        shape = tensor.get("shape")
        if (
            not isinstance(name, str)
            or not name
            or name in names
            or not isinstance(dtype, str)
            or not dtype
            or not isinstance(shape, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            )
        ):
            raise CompletedRunLineageError("state projection tensor metadata differs")
        names.add(name)
        normalized.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "raw_little_endian_sha256": _sha256(
                    tensor.get("raw_little_endian_sha256"),
                    "state tensor raw bytes",
                ),
            }
        )
    normalized.sort(key=lambda row: row["name"])
    state_sha256 = _sha256(payload.get("state_sha256"), "state projection")
    if (
        tensors != normalized
        or state_sha256 != canonical_sha256(normalized)
        or descriptor.get("state_sha256") != state_sha256
    ):
        raise CompletedRunLineageError("state projection receipt differs")
    return {**artifact, "state_sha256": state_sha256}


def _validate_execution(payload: Any, planned: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != {
        "attempts",
        "optimizer_steps",
        "sequences",
        "valid_tokens",
        "admitted_utf8_bytes",
        "modeled_training_flops",
        "skipped_updates",
        "nonfinite_updates",
        "overflow_updates",
        "measured_gpu_seconds",
        "update_ledger_sha256",
    }:
        raise CompletedRunLineageError("execution receipt differs")
    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise CompletedRunLineageError("execution attempts are missing")
    normalized_attempts = []
    expected_start = 0
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or set(attempt) != {
            "job_id",
            "attempt_index",
            "host",
            "gpu_identity_sha256",
            "exit_code",
            "signal",
            "restarts",
            "committed_update_start",
            "committed_update_end",
            "stdout_sha256",
            "stderr_sha256",
            "accounting_sha256",
        }:
            raise CompletedRunLineageError("execution attempt differs")
        start = _positive_integer(
            attempt.get("committed_update_start"),
            "committed update start",
            allow_zero=True,
        )
        end = _positive_integer(
            attempt.get("committed_update_end"),
            "committed update end",
            allow_zero=True,
        )
        exit_code = attempt.get("exit_code")
        signal = attempt.get("signal")
        if (
            isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or exit_code < 0
            or isinstance(signal, bool)
            or not isinstance(signal, int)
            or signal < 0
        ):
            raise CompletedRunLineageError("execution attempt termination differs")
        succeeded = exit_code == 0 and signal == 0
        if (
            attempt.get("attempt_index") != index
            or start != expected_start
            or (succeeded and end <= start)
            or (not succeeded and end != start)
            or attempt.get("restarts") != 0
            or not isinstance(attempt.get("job_id"), str)
            or not attempt["job_id"]
            or not isinstance(attempt.get("host"), str)
            or not attempt["host"]
        ):
            raise CompletedRunLineageError("execution attempt status or ranges differ")
        for key in (
            "gpu_identity_sha256",
            "stdout_sha256",
            "stderr_sha256",
            "accounting_sha256",
        ):
            _sha256(attempt.get(key), key)
        normalized_attempts.append(dict(attempt))
        expected_start = end
    expected_budget = planned.get("training_budget")
    if not isinstance(expected_budget, dict):
        raise CompletedRunLineageError("planned training budget is missing")
    budget_fields = (
        "optimizer_steps",
        "sequences",
        "valid_tokens",
        "admitted_utf8_bytes",
        "modeled_training_flops",
    )
    observed = {key: _positive_integer(payload.get(key), key) for key in budget_fields}
    if any(expected_budget.get(key) != observed[key] for key in budget_fields):
        raise CompletedRunLineageError("observed training budget differs from plan")
    if (
        expected_start != observed["optimizer_steps"]
        or attempts[-1]["exit_code"] != 0
        or attempts[-1]["signal"] != 0
    ):
        raise CompletedRunLineageError("attempt update ranges do not cover the run")
    if any(
        payload.get(key) != 0
        for key in ("skipped_updates", "nonfinite_updates", "overflow_updates")
    ):
        raise CompletedRunLineageError("training contains skipped or invalid updates")
    return {
        "attempts": normalized_attempts,
        **observed,
        "skipped_updates": 0,
        "nonfinite_updates": 0,
        "overflow_updates": 0,
        "measured_gpu_seconds": _finite(
            payload.get("measured_gpu_seconds"), "measured GPU seconds"
        ),
        "update_ledger_sha256": _sha256(
            payload.get("update_ledger_sha256"), "update ledger"
        ),
    }


def validate_receipt(payload: Any, artifact_root: Path) -> dict[str, Any]:
    """Reopen a portable artifact bundle and validate one completed run."""

    if not isinstance(payload, dict) or set(payload) != TOP_LEVEL_KEYS:
        raise CompletedRunLineageError("completed-run receipt keys differ")
    if not artifact_root.is_dir() or artifact_root.is_symlink():
        raise CompletedRunLineageError("artifact root is missing or unsafe")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "complete"
        or payload.get("scientific_status") != "complete"
        or payload.get("training_authorized") is not True
        or payload.get("official_training_order_received") is not True
        or payload.get("terminal_public_board_accessed") is not False
        or payload.get("role") not in ROLES
        or payload.get("changed_factor")
        not in {"latent_workspace", "fast_path_capacity"}
        or not isinstance(payload.get("created_at_utc"), str)
        or not payload["created_at_utc"].endswith("Z")
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise CompletedRunLineageError("completed-run identity or boundary differs")
    run_identity = _sha256(payload.get("run_identity_sha256"), "run identity")
    comparison_group = _sha256(
        payload.get("comparison_group_sha256"), "comparison group"
    )
    planned, plan_artifact = _validate_plan(
        payload.get("plan"),
        artifact_root,
        run_identity,
        comparison_group,
        payload["role"],
    )
    authorization_artifact = _validate_authorization(
        payload.get("authorization"),
        artifact_root,
        plan_artifact["sha256"],
        run_identity,
    )
    parent = payload.get("parent")
    if not isinstance(parent, dict) or set(parent) != {
        "completed_run_receipt_sha256",
        "checkpoint_tree_sha256",
        "fast_path_state_sha256",
    }:
        raise CompletedRunLineageError("parent lineage differs")
    normalized_parent = {
        key: _sha256(value, f"parent {key}") for key, value in parent.items()
    }
    if (
        planned["parent_completed_run_receipt_sha256"]
        != normalized_parent["completed_run_receipt_sha256"]
        or planned["parent_checkpoint_tree_sha256"]
        != normalized_parent["checkpoint_tree_sha256"]
        or planned["parent_fast_path_state_sha256"]
        != normalized_parent["fast_path_state_sha256"]
        or planned["changed_factor"] != payload["changed_factor"]
    ):
        raise CompletedRunLineageError("planned parent or changed factor differs")
    immutable = payload.get("immutable_inputs")
    required_inputs = {
        "architecture_sha256",
        "geometry_sha256",
        "tokenizer_sha256",
        "ordered_stream_sha256",
        "environment_sha256",
        "source_tree_sha256",
        "runtime_sha256",
        "kernel_contract_sha256",
        "system_config_sha256",
        "workspace_plan_sha256",
        "workspace_candidate_identity_sha256",
    }
    if not isinstance(immutable, dict) or set(immutable) != required_inputs:
        raise CompletedRunLineageError("immutable input lineage differs")
    normalized_inputs = {key: _sha256(value, key) for key, value in immutable.items()}
    for key in (
        "tokenizer_sha256",
        "ordered_stream_sha256",
        "environment_sha256",
        "system_config_sha256",
        "workspace_plan_sha256",
        "workspace_candidate_identity_sha256",
    ):
        if planned.get(key) != normalized_inputs[key]:
            raise CompletedRunLineageError(f"planned {key} differs")
    execution = _validate_execution(payload.get("execution"), planned)
    checkpoint = _validate_checkpoint_tree(
        payload.get("checkpoint_tree"), artifact_root
    )
    projections = payload.get("state_projections")
    if not isinstance(projections, dict) or set(projections) != {
        "system_state_sha256",
        "fast_path_state_sha256",
        "slow_path_state_sha256",
    }:
        raise CompletedRunLineageError("state projection receipt differs")
    normalized_projections = {
        key: _validate_state_projection(descriptor, artifact_root, key)
        for key, descriptor in projections.items()
    }
    observed_fast_state = normalized_projections["fast_path_state_sha256"][
        "state_sha256"
    ]
    if payload["role"] == "workspace_treatment":
        if observed_fast_state != normalized_parent["fast_path_state_sha256"]:
            raise CompletedRunLineageError("trained run changed the frozen fast path")
    elif observed_fast_state == normalized_parent["fast_path_state_sha256"]:
        raise CompletedRunLineageError(
            "equal-FLOP control did not change fast capacity"
        )
    # Validation above replays every artifact. Preserve the signed payload byte
    # semantics rather than returning a normalized object with a stale self-hash.
    _ = (
        authorization_artifact,
        normalized_inputs,
        execution,
        checkpoint,
        normalized_projections,
    )
    return payload


def load_and_validate_receipt(path: Path, artifact_root: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise CompletedRunLineageError("completed-run receipt is missing or unsafe")
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompletedRunLineageError("completed-run receipt is unreadable") from error
    return validate_receipt(payload, artifact_root)


def write_receipt(payload: dict[str, Any], output: Path, artifact_root: Path) -> None:
    validate_receipt(payload, artifact_root)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(encoded)
