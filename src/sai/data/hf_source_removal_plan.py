"""Seal an exact, recoverable Hugging Face source-prefix removal plan."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, RepoFile

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-hf-source-removal-plan-v1"


class HfSourceRemovalPlanError(RuntimeError):
    """The remote source prefix or expected removal boundary differs."""


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def normalize_remote_files(items: Iterable[Any], prefix: str) -> list[dict[str, Any]]:
    """Normalize only source-safe remote object identities."""

    if not isinstance(prefix, str) or not prefix or prefix.startswith("/"):
        raise HfSourceRemovalPlanError("source removal prefix differs")
    normalized = []
    for item in items:
        if not isinstance(item, RepoFile):
            continue
        lfs = item.lfs
        if (
            not item.path.startswith(prefix.rstrip("/") + "/")
            or isinstance(item.size, bool)
            or not isinstance(item.size, int)
            or item.size <= 0
            or lfs is None
            or lfs.size != item.size
            or not _sha256(lfs.sha256)
            or not isinstance(item.blob_id, str)
            or not item.blob_id
            or not _sha256(item.xet_hash)
        ):
            raise HfSourceRemovalPlanError("remote source object differs")
        normalized.append(
            {
                "path": item.path,
                "bytes": item.size,
                "lfs_sha256": lfs.sha256,
                "xet_hash": item.xet_hash,
                "git_blob_id": item.blob_id,
            }
        )
    normalized.sort(key=lambda row: row["path"])
    if not normalized or len({row["path"] for row in normalized}) != len(normalized):
        raise HfSourceRemovalPlanError("remote source object coverage differs")
    return normalized


def build_payload(
    *,
    repository: str,
    base_revision: str,
    prefix: str,
    upstream_repository: str,
    upstream_revision: str,
    files: list[dict[str, Any]],
    expected_files: int,
    expected_bytes: int,
) -> dict[str, Any]:
    """Build a non-destructive plan that must precede remote deletion."""

    if (
        not isinstance(repository, str)
        or not repository
        or not isinstance(base_revision, str)
        or not base_revision
        or not isinstance(upstream_repository, str)
        or not upstream_repository
        or not isinstance(upstream_revision, str)
        or not upstream_revision
        or isinstance(expected_files, bool)
        or not isinstance(expected_files, int)
        or expected_files <= 0
        or isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes <= 0
        or len(files) != expected_files
        or sum(row["bytes"] for row in files) != expected_bytes
    ):
        raise HfSourceRemovalPlanError("source removal expectation differs")
    payload = {
        "schema": SCHEMA,
        "status": "planned_recoverable_source_prefix_removal",
        "repository": repository,
        "repo_type": "dataset",
        "base_revision": base_revision,
        "prefix": prefix.rstrip("/"),
        "objects": files,
        "object_count": len(files),
        "total_bytes": expected_bytes,
        "ordered_object_identities_sha256": canonical_sha256(files),
        "upstream_recovery": {
            "repository": upstream_repository,
            "revision": upstream_revision,
            "exact_redownload_available": True,
        },
        "source_text_persisted": False,
        "deletion_executed": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--upstream-repository", required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--expected-files", type=int, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise HfSourceRemovalPlanError("source removal output already exists")
    token = os.environ.get(args.token_env, "")
    if not token:
        raise HfSourceRemovalPlanError(f"{args.token_env} is required")
    api = HfApi(token=token)
    info = api.repo_info(args.repository, repo_type="dataset")
    files = normalize_remote_files(
        api.list_repo_tree(
            args.repository,
            repo_type="dataset",
            path_in_repo=args.prefix,
            recursive=True,
            expand=True,
        ),
        args.prefix,
    )
    payload = build_payload(
        repository=args.repository,
        base_revision=info.sha,
        prefix=args.prefix,
        upstream_repository=args.upstream_repository,
        upstream_revision=args.upstream_revision,
        files=files,
        expected_files=args.expected_files,
        expected_bytes=args.expected_bytes,
    )
    _atomic_create(args.output, payload)
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "objects"},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
