"""Seal verified completion of a recoverable Hugging Face source removal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.hf_source_removal_plan import SCHEMA as PLAN_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-hf-source-removal-receipt-v1"


class HfSourceRemovalReceiptError(RuntimeError):
    """The removal plan or observed post-removal state differs."""


def _load_plan(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise HfSourceRemovalReceiptError("source removal plan is missing or unsafe")
    try:
        plan = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise HfSourceRemovalReceiptError("source removal plan is invalid") from error
    if not isinstance(plan, dict):
        raise HfSourceRemovalReceiptError("source removal plan differs")
    unsigned = {key: value for key, value in plan.items() if key != "receipt_sha256"}
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("status") != "planned_recoverable_source_prefix_removal"
        or plan.get("receipt_sha256") != canonical_sha256(unsigned)
        or plan.get("deletion_executed") is not False
        or plan.get("source_text_persisted") is not False
        or plan.get("upstream_recovery", {}).get("exact_redownload_available")
        is not True
    ):
        raise HfSourceRemovalReceiptError("source removal plan differs")
    return plan


def build_receipt(
    *,
    plan: dict[str, Any],
    plan_file_sha256: str,
    plan_publication_revision: str,
    deletion_revision: str,
    verified_current_revision: str,
    remaining_prefix_files: int,
    post_source_files: int,
    post_source_bytes: int,
    post_data_files: int,
    post_data_bytes: int,
) -> dict[str, Any]:
    """Build a receipt only for a fully absent planned prefix."""

    integers = (
        remaining_prefix_files,
        post_source_files,
        post_source_bytes,
        post_data_files,
        post_data_bytes,
    )
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in integers)
        or remaining_prefix_files != 0
        or min(post_source_files, post_source_bytes, post_data_files, post_data_bytes)
        <= 0
        or not all(
            isinstance(value, str) and value
            for value in (
                plan_file_sha256,
                plan_publication_revision,
                deletion_revision,
                verified_current_revision,
            )
        )
        or deletion_revision != verified_current_revision
    ):
        raise HfSourceRemovalReceiptError("post-removal verification differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete_verified_recoverable_source_prefix_removal",
        "repository": plan["repository"],
        "repo_type": plan["repo_type"],
        "prefix": plan["prefix"],
        "plan": {
            "base_revision": plan["base_revision"],
            "publication_revision": plan_publication_revision,
            "file_sha256": plan_file_sha256,
            "receipt_sha256": plan["receipt_sha256"],
        },
        "deletion_revision": deletion_revision,
        "verified_current_revision": verified_current_revision,
        "removed_objects": plan["object_count"],
        "removed_bytes": plan["total_bytes"],
        "remaining_prefix_files": remaining_prefix_files,
        "post_removal_source_tree": {
            "files": post_source_files,
            "bytes": post_source_bytes,
            "data_files": post_data_files,
            "data_bytes": post_data_bytes,
        },
        "upstream_recovery": plan["upstream_recovery"],
        "recoverable_from_repository_history": True,
        "source_text_persisted_in_receipt": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-publication-revision", required=True)
    parser.add_argument("--deletion-revision", required=True)
    parser.add_argument("--verified-current-revision", required=True)
    parser.add_argument("--remaining-prefix-files", type=int, required=True)
    parser.add_argument("--post-source-files", type=int, required=True)
    parser.add_argument("--post-source-bytes", type=int, required=True)
    parser.add_argument("--post-data-files", type=int, required=True)
    parser.add_argument("--post-data-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise HfSourceRemovalReceiptError("source removal receipt already exists")
    plan = _load_plan(args.plan)
    payload = build_receipt(
        plan=plan,
        plan_file_sha256=sha256_file(args.plan),
        plan_publication_revision=args.plan_publication_revision,
        deletion_revision=args.deletion_revision,
        verified_current_revision=args.verified_current_revision,
        remaining_prefix_files=args.remaining_prefix_files,
        post_source_files=args.post_source_files,
        post_source_bytes=args.post_source_bytes,
        post_data_files=args.post_data_files,
        post_data_bytes=args.post_data_bytes,
    )
    _atomic_create(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
