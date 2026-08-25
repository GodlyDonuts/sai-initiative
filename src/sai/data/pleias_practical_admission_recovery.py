"""Validate a completed primary admission or clean its exact failed partial."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_practical_admission import (
    SCHEMA as ADMISSION_SCHEMA,
)
from sai.data.pleias_practical_admission import (
    _load_signed,
)
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-pleias-practical-admission-recovery-cleanup-v1"


class PleiasPracticalAdmissionRecoveryError(RuntimeError):
    """The primary receipt, terminal state, or exact cleanup target differs."""


def validate_primary(receipt_path: Path) -> dict[str, Any]:
    """Return a complete signed primary receipt or fail closed."""

    payload = _load_signed(receipt_path, ADMISSION_SCHEMA)
    if (
        payload.get("status")
        != "complete_practical_pleias_pretraining_admission"
        or payload.get("global_exact_content_deduplication_complete") is not True
        or payload.get("known_quarantine_exclusions_applied") is not True
        or payload.get("training_ready") is not True
        or payload.get("four_b_training_authorized") is not False
    ):
        raise PleiasPracticalAdmissionRecoveryError(
            "primary admission receipt is not complete"
        )
    return payload


def cleanup_failed_primary(
    root: Path,
    expected_root: Path,
    destination: Path,
    primary_job: int,
    primary_state: str,
) -> dict[str, Any]:
    """Remove only an incomplete exact root and durably record what was removed."""

    if (
        isinstance(primary_job, bool)
        or not isinstance(primary_job, int)
        or primary_job <= 0
        or not isinstance(primary_state, str)
        or not primary_state
        or primary_state.startswith(("RUNNING", "PENDING"))
        or root != expected_root
        or root.is_symlink()
        or destination.exists()
        or destination.is_symlink()
    ):
        raise PleiasPracticalAdmissionRecoveryError(
            "recovery cleanup target differs"
        )
    entries = []
    partial_bytes = 0
    root_present = root.exists()
    if root_present:
        if not root.is_dir():
            raise PleiasPracticalAdmissionRecoveryError(
                "recovery cleanup target differs"
            )
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise PleiasPracticalAdmissionRecoveryError(
                    "recovery partial contains a symlink"
                )
            relative = str(path.relative_to(root))
            if path.is_file():
                size = path.stat().st_size
                partial_bytes += size
                entries.append({"path": relative, "type": "file", "bytes": size})
            elif path.is_dir():
                entries.append({"path": relative, "type": "directory"})
            else:
                raise PleiasPracticalAdmissionRecoveryError(
                    "recovery partial contains an unsupported entry"
                )
        shutil.rmtree(root)
    payload = {
        "schema": SCHEMA,
        "status": "complete_failed-primary-partial-cleanup",
        "primary_admission_job": primary_job,
        "primary_admission_state": primary_state,
        "exact_cleanup_root": str(root),
        "partial_root_present": root_present,
        "partial_entries": len(entries),
        "partial_file_bytes_removed": partial_bytes,
        "ordered_partial_entries_sha256": canonical_sha256(entries),
        "partial_scientific_release_complete": False,
        "primary_receipt_valid": False,
        "recovery_admission_required": True,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(destination, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    cleanup = subparsers.add_parser("cleanup")
    cleanup.add_argument("--root", type=Path, required=True)
    cleanup.add_argument("--expected-root", type=Path, required=True)
    cleanup.add_argument("--destination", type=Path, required=True)
    cleanup.add_argument("--primary-job", type=int, required=True)
    cleanup.add_argument("--primary-state", required=True)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate_primary(args.receipt)
    else:
        result = cleanup_failed_primary(
            args.root,
            args.expected_root,
            args.destination,
            args.primary_job,
            args.primary_state,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
