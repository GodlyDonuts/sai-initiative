"""Validate the initial bridge screen graph and stage its confirmation handoff."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file


SCREEN_LAUNCH_SCHEMA = "sai-bridge-transfer-newton-launch-v1"
SCHEMA = "sai-bridge-transfer-confirmation-stage-v1"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_SCREEN_KEYS = {
    "schema",
    "status",
    "runtime_commit",
    "launcher_job_id",
    "newton_jobs",
    "one_h100_per_arm",
    "matched_token_budget",
    "four_b_training_authorized",
    "receipt_sha256",
}
_SCREEN_JOBS = {"unchanged", "source_control", "connections", "aggregate"}


class BridgeTransferConfirmationStageError(RuntimeError):
    """The screen graph or confirmation handoff identity differs."""


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise BridgeTransferConfirmationStageError(f"{label} commit differs")
    return value


def _positive_job(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BridgeTransferConfirmationStageError(f"{label} job differs")
    return value


def load_screen_launch(path: Path, expected_runtime_commit: str) -> dict[str, Any]:
    """Replay the signed initial-screen launch receipt and return its exact graph."""

    expected_runtime_commit = _commit(
        expected_runtime_commit, "expected screen runtime"
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BridgeTransferConfirmationStageError(
            "screen launch receipt is unavailable"
        ) from error
    if not isinstance(payload, dict) or set(payload) != _SCREEN_KEYS:
        raise BridgeTransferConfirmationStageError("screen launch receipt differs")
    claimed = payload.get("receipt_sha256")
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256")
    if claimed != canonical_sha256(unsigned):
        raise BridgeTransferConfirmationStageError("screen launch signature differs")
    if (
        payload.get("schema") != SCREEN_LAUNCH_SCHEMA
        or payload.get("status") != "complete_newton_transfer_graph_launch"
        or payload.get("runtime_commit") != expected_runtime_commit
        or payload.get("one_h100_per_arm") is not True
        or payload.get("matched_token_budget") is not True
        or payload.get("four_b_training_authorized") is not False
    ):
        raise BridgeTransferConfirmationStageError("screen launch contract differs")
    _positive_job(payload.get("launcher_job_id"), "screen launcher")
    jobs = payload.get("newton_jobs")
    if not isinstance(jobs, dict) or set(jobs) != _SCREEN_JOBS:
        raise BridgeTransferConfirmationStageError("screen job coverage differs")
    job_ids = [_positive_job(jobs[key], key) for key in sorted(_SCREEN_JOBS)]
    if len(set(job_ids)) != len(job_ids):
        raise BridgeTransferConfirmationStageError("screen job identities overlap")
    return payload


def write_stage_receipt(
    *,
    output: Path,
    screen_launch_path: Path,
    screen_runtime_commit: str,
    confirmation_runtime_commit: str,
    confirmation_launcher_job: int,
) -> dict[str, Any]:
    """Write one create-only receipt binding the screen aggregate to its handoff."""

    screen = load_screen_launch(screen_launch_path, screen_runtime_commit)
    confirmation_runtime_commit = _commit(
        confirmation_runtime_commit, "confirmation runtime"
    )
    confirmation_launcher_job = _positive_job(
        confirmation_launcher_job, "confirmation launcher"
    )
    if output.exists():
        raise BridgeTransferConfirmationStageError("stage receipt already exists")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_bridge_transfer_confirmation_staging",
        "screen_launch": {
            "bytes": screen_launch_path.stat().st_size,
            "sha256": sha256_file(screen_launch_path),
            "receipt_sha256": screen["receipt_sha256"],
            "runtime_commit": screen["runtime_commit"],
            "aggregate_job": screen["newton_jobs"]["aggregate"],
        },
        "confirmation_runtime_commit": confirmation_runtime_commit,
        "confirmation_launcher_job": confirmation_launcher_job,
        "dependency": f"afterok:{screen['newton_jobs']['aggregate']}",
        "one_h100_per_confirmation_arm": True,
        "matched_token_budget": True,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--screen-launch", type=Path, required=True)
    inspect_parser.add_argument("--screen-runtime-commit", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--screen-launch", type=Path, required=True)
    record_parser.add_argument("--screen-runtime-commit", required=True)
    record_parser.add_argument("--confirmation-runtime-commit", required=True)
    record_parser.add_argument("--confirmation-launcher-job", type=int, required=True)
    record_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        receipt = load_screen_launch(args.screen_launch, args.screen_runtime_commit)
        print(receipt["newton_jobs"]["aggregate"])
        return
    write_stage_receipt(
        output=args.output,
        screen_launch_path=args.screen_launch,
        screen_runtime_commit=args.screen_runtime_commit,
        confirmation_runtime_commit=args.confirmation_runtime_commit,
        confirmation_launcher_job=args.confirmation_launcher_job,
    )


if __name__ == "__main__":
    main()
