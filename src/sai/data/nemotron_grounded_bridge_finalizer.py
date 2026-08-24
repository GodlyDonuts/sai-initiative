"""Serialize the independent bridge aggregate and benchmark screen."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
from typing import Any

from sai.data.data_yield_ledger import _load_receipt
from sai.data.grounded_bridge_decontamination import SCHEMA as DECONTAMINATION_SCHEMA
from sai.data.grounded_bridge_decontamination import build_screen
from sai.data.nemotron_grounded_bridge_verification_aggregate import (
    SCHEMA as AGGREGATE_SCHEMA,
)
from sai.data.nemotron_grounded_bridge_verification_aggregate import build_aggregate


class NemotronGroundedBridgeFinalizerError(RuntimeError):
    """The finalizer lock or an already-created output differs."""


def _sealed_receipt(root: Path, schema: str, status: str) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise NemotronGroundedBridgeFinalizerError(
            f"existing output root is unsafe: {root}"
        )
    receipt_path = root / "receipt.json"
    if not receipt_path.is_file():
        raise NemotronGroundedBridgeFinalizerError(
            f"existing output lacks a sealed receipt: {root}"
        )
    receipt = _load_receipt(receipt_path)
    if receipt.get("schema") != schema or receipt.get("status") != status:
        raise NemotronGroundedBridgeFinalizerError(
            f"existing output receipt differs: {root}"
        )
    return receipt


def finalize(
    population_root: Path,
    same_family_aggregate_root: Path,
    judgments_root: Path,
    aggregate_root: Path,
    boundary_indexes: list[Path],
    decontamination_root: Path,
    lock_path: Path,
    *,
    logical_shards: int,
) -> dict[str, Any]:
    """Allow exactly one process at a time to create the two terminal outputs."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if aggregate_root.exists() or aggregate_root.is_symlink():
            aggregate = _sealed_receipt(
                aggregate_root,
                AGGREGATE_SCHEMA,
                "complete_independent_model_family_bridge_verification_routes",
            )
            aggregate_created = False
        else:
            aggregate = build_aggregate(
                population_root,
                same_family_aggregate_root,
                judgments_root,
                aggregate_root,
                logical_shards=logical_shards,
            )
            aggregate_created = True

        if decontamination_root.exists() or decontamination_root.is_symlink():
            decontamination = _sealed_receipt(
                decontamination_root,
                DECONTAMINATION_SCHEMA,
                "complete_post_generation_bridge_benchmark_screen",
            )
            decontamination_created = False
        else:
            decontamination = build_screen(
                aggregate_root,
                boundary_indexes,
                decontamination_root,
            )
            decontamination_created = True

        return {
            "schema": "sai-nemotron-grounded-bridge-finalization-v1",
            "aggregate_created": aggregate_created,
            "aggregate_receipt_sha256": aggregate["receipt_sha256"],
            "decontamination_created": decontamination_created,
            "decontamination_receipt_sha256": decontamination["receipt_sha256"],
            "serialized_by_process_lock": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--same-family-aggregate-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--boundary-index", type=Path, action="append", required=True)
    parser.add_argument("--decontamination-root", type=Path, required=True)
    parser.add_argument("--lock-path", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, default=64)
    args = parser.parse_args()
    result = finalize(
        args.population_root,
        args.same_family_aggregate_root,
        args.judgments_root,
        args.aggregate_root,
        args.boundary_index,
        args.decontamination_root,
        args.lock_path,
        logical_shards=args.logical_shards,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
