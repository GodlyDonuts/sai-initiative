from __future__ import annotations

import pytest

from sai.data.hf_source_removal_plan import build_payload
from sai.data.hf_source_removal_receipt import (
    HfSourceRemovalReceiptError,
    build_receipt,
)


def _plan() -> dict:
    files = [
        {
            "path": "sources/example/r1/a.parquet",
            "bytes": 10,
            "lfs_sha256": "a" * 64,
            "xet_hash": "b" * 64,
            "git_blob_id": "blob",
        }
    ]
    return build_payload(
        repository="owner/data",
        base_revision="base",
        prefix="sources/example/r1",
        upstream_repository="upstream/data",
        upstream_revision="revision",
        files=files,
        expected_files=1,
        expected_bytes=10,
    )


def test_completed_removal_is_recoverable_and_source_safe() -> None:
    receipt = build_receipt(
        plan=_plan(),
        plan_file_sha256="c" * 64,
        plan_publication_revision="published",
        deletion_revision="deleted",
        verified_current_revision="deleted",
        remaining_prefix_files=0,
        post_source_files=9,
        post_source_bytes=90,
        post_data_files=8,
        post_data_bytes=80,
    )
    assert receipt["removed_objects"] == 1
    assert receipt["removed_bytes"] == 10
    assert receipt["remaining_prefix_files"] == 0
    assert receipt["recoverable_from_repository_history"] is True
    assert receipt["source_text_persisted_in_receipt"] is False


def test_completed_removal_rejects_remaining_files_or_revision_drift() -> None:
    for remaining, current in ((1, "deleted"), (0, "other")):
        with pytest.raises(HfSourceRemovalReceiptError, match="verification"):
            build_receipt(
                plan=_plan(),
                plan_file_sha256="c" * 64,
                plan_publication_revision="published",
                deletion_revision="deleted",
                verified_current_revision=current,
                remaining_prefix_files=remaining,
                post_source_files=9,
                post_source_bytes=90,
                post_data_files=8,
                post_data_bytes=80,
            )
