from __future__ import annotations

import pytest
from huggingface_hub import RepoFile

from sai.data.hf_source_removal_plan import (
    HfSourceRemovalPlanError,
    build_payload,
    normalize_remote_files,
)


def _file(path: str, size: int, digest: str) -> RepoFile:
    return RepoFile(
        path=path,
        size=size,
        oid="blob-" + digest[:8],
        lfs={"size": size, "oid": digest, "pointerSize": 132},
        xetHash=digest,
    )


def test_removal_plan_binds_every_recoverable_object_without_text() -> None:
    prefix = "sources/example/r1"
    files = normalize_remote_files(
        [
            _file(prefix + "/b.parquet", 20, "b" * 64),
            _file(prefix + "/a.parquet", 10, "a" * 64),
        ],
        prefix,
    )
    payload = build_payload(
        repository="owner/data",
        base_revision="base",
        prefix=prefix,
        upstream_repository="upstream/data",
        upstream_revision="revision",
        files=files,
        expected_files=2,
        expected_bytes=30,
    )
    assert [row["path"] for row in payload["objects"]] == [
        prefix + "/a.parquet",
        prefix + "/b.parquet",
    ]
    assert payload["upstream_recovery"]["exact_redownload_available"] is True
    assert payload["deletion_executed"] is False
    assert payload["source_text_persisted"] is False
    assert all("text" not in row for row in payload["objects"])


def test_removal_plan_rejects_byte_or_count_drift() -> None:
    files = normalize_remote_files(
        [_file("sources/example/r1/a.parquet", 10, "a" * 64)],
        "sources/example/r1",
    )
    with pytest.raises(HfSourceRemovalPlanError, match="expectation"):
        build_payload(
            repository="owner/data",
            base_revision="base",
            prefix="sources/example/r1",
            upstream_repository="upstream/data",
            upstream_revision="revision",
            files=files,
            expected_files=1,
            expected_bytes=11,
        )


def test_removal_plan_rejects_path_escape() -> None:
    with pytest.raises(HfSourceRemovalPlanError, match="object"):
        normalize_remote_files(
            [_file("sources/other/a.parquet", 10, "a" * 64)],
            "sources/example/r1",
        )
