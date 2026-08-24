"""Verify all rewritten PleIAs shards and their final remote LFS identities."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_production_materializer import (
    DESTINATION_REPOSITORY,
    _load_signed,
)
from sai.data.pleias_production_materializer import (
    SHARD_SCHEMA as MATERIALIZED_SCHEMA,
)
from sai.data.pleias_subdocument_rewrite import (
    DESTINATION_PREFIX,
)
from sai.data.pleias_subdocument_rewrite import (
    SHARD_SCHEMA as REWRITE_SCHEMA,
)
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-pleias-subdocument-rewritten-aggregate-v1"


class PleiasSubdocumentRewriteAggregateError(RuntimeError):
    """Rewritten shard coverage, accounting, or remote custody differs."""


def build_aggregate(
    materialized_root: Path,
    rewrite_root: Path,
    output: Path,
    logical_shards: int,
    token: str,
) -> dict[str, Any]:
    """Replay one rewritten output per source shard from one remote snapshot."""

    if output.exists() or output.is_symlink() or logical_shards <= 0 or not token:
        raise PleiasSubdocumentRewriteAggregateError("aggregate arguments differ")
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise PleiasSubdocumentRewriteAggregateError(
            "huggingface_hub is required"
        ) from error
    totals: Counter[str] = Counter()
    receipts = []
    remotes = {}
    for shard_index in range(logical_shards):
        materialized = _load_signed(
            materialized_root / "shards" / f"shard_{shard_index:05d}" / "receipt.json",
            MATERIALIZED_SCHEMA,
        )
        rewritten = _load_signed(
            rewrite_root / "shards" / f"shard_{shard_index:05d}" / "receipt.json",
            REWRITE_SCHEMA,
        )
        remote = rewritten.get("remote_output")
        counts = rewritten.get("counts")
        if (
            rewritten.get("logical_shards") != logical_shards
            or rewritten.get("shard_index") != shard_index
            or rewritten.get("source", {}).get("materialized_receipt_sha256")
            != materialized["receipt_sha256"]
            or rewritten.get("pleias_global_subdocument_rewrite_complete") is not True
            or rewritten.get("local_payload_removed_after_remote_verification")
            is not True
            or not isinstance(counts, dict)
            or counts.get("documents")
            != materialized.get("counts", {}).get("retained_rows", 0)
            or counts.get("input_text_utf8_bytes")
            != materialized.get("counts", {}).get("retained_text_utf8_bytes", 0)
            or counts.get("output_text_utf8_bytes", 0)
            > counts.get("input_text_utf8_bytes", 0)
            or not isinstance(remote, dict)
            or remote.get("repository") != DESTINATION_REPOSITORY
            or not isinstance(remote.get("path"), str)
            or not remote["path"].startswith(f"{DESTINATION_PREFIX}/")
            or remote["path"] in remotes
            or not isinstance(remote.get("bytes"), int)
            or remote["bytes"] <= 0
            or not isinstance(remote.get("sha256"), str)
            or len(remote["sha256"]) != 64
        ):
            raise PleiasSubdocumentRewriteAggregateError("rewritten shard differs")
        for key, value in counts.items():
            totals[key] += value
        totals["remote_output_bytes"] += remote["bytes"]
        remotes[remote["path"]] = remote
        receipts.append(rewritten["receipt_sha256"])
    api = HfApi(token=token)
    info = api.dataset_info(DESTINATION_REPOSITORY, files_metadata=True)
    siblings = {item.rfilename: item for item in info.siblings or []}
    for path, expected in remotes.items():
        sibling = siblings.get(path)
        lfs = None if sibling is None else sibling.lfs
        if (
            sibling is None
            or lfs is None
            or sibling.size != expected["bytes"]
            or lfs.size != expected["bytes"]
            or lfs.sha256 != expected["sha256"]
        ):
            raise PleiasSubdocumentRewriteAggregateError(
                "rewritten remote LFS identity differs"
            )
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_pleias_subdocument_rewritten",
        "shards": {
            "logical_shards": logical_shards,
            "ordered_receipts_sha256": canonical_sha256(receipts),
            "remote_repository": DESTINATION_REPOSITORY,
            "remote_revision_verified": info.sha,
            "remote_prefix": DESTINATION_PREFIX,
        },
        "totals": dict(sorted(totals.items())),
        "complete_materialized_document_coverage": True,
        "all_remote_lfs_identities_verified": True,
        "pleias_global_subdocument_deduplication_complete": True,
        "cross_source_subdocument_deduplication_complete": False,
        "token_count_requires_recomputation": True,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--rewrite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    result = build_aggregate(
        args.materialized_root,
        args.rewrite_root,
        args.output,
        args.logical_shards,
        os.environ.get(args.token_env, ""),
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
