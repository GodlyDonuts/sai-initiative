from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.quarantine_exclusion_registry import (
    QuarantineExclusionRegistryError,
    build_registry,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _manifest(root: Path, identity: str, content: str) -> None:
    root.mkdir()
    row = {
        "schema": "sai-audit-quarantine-exclusion-v1",
        "candidate_identity_sha256": identity,
        "source_content_sha256": content,
        "route": "quarantine",
        "dataset_materialization_allowed": False,
        "source_text_persisted": False,
    }
    row["record_sha256"] = canonical_sha256(row)
    path = root / "quarantine_exclusions.jsonl"
    path.write_text(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    receipt = {
        "schema": "sai-audit-quarantine-manifest-receipt-v1",
        "status": "complete_audit_quarantine_exclusion_manifest",
        "manifest": {
            "path": path.name,
            "rows": 1,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "source_text_persisted": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (root / "receipt.json").write_text(json.dumps(receipt))


def test_registry_merges_text_free_unique_quarantines(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _manifest(first, "a" * 64, "c" * 64)
    _manifest(second, "b" * 64, "d" * 64)
    output = tmp_path / "registry"
    result = build_registry([first, second], output)
    rows = [json.loads(line) for line in (output / "quarantine_registry.jsonl").open()]
    assert result["source_rows"] == 2
    assert result["unique_quarantine_rows"] == 2
    assert len(rows) == 2
    assert all(row["dataset_materialization_allowed"] is False for row in rows)
    assert all(row["source_text_persisted"] is False for row in rows)
    assert all("text" not in row for row in rows)


def test_registry_rejects_duplicate_candidate_identity(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _manifest(first, "a" * 64, "c" * 64)
    _manifest(second, "a" * 64, "c" * 64)
    with pytest.raises(QuarantineExclusionRegistryError, match="duplicated"):
        build_registry([first, second], tmp_path / "registry")
    assert not (tmp_path / "registry").exists()


def test_registry_rejects_manifest_byte_tamper(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _manifest(source, "a" * 64, "c" * 64)
    with (source / "quarantine_exclusions.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(QuarantineExclusionRegistryError, match="bytes"):
        build_registry([source], tmp_path / "registry")
    assert not (tmp_path / "registry").exists()


def test_registry_rejects_source_text_field(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _manifest(source, "a" * 64, "c" * 64)
    path = source / "quarantine_exclusions.jsonl"
    row = json.loads(path.read_text())
    row["source_text"] = "must not persist"
    row["record_sha256"] = canonical_sha256(
        {key: value for key, value in row.items() if key != "record_sha256"}
    )
    path.write_text(json.dumps(row) + "\n")
    receipt = json.loads((source / "receipt.json").read_text())
    receipt["manifest"]["bytes"] = path.stat().st_size
    receipt["manifest"]["sha256"] = sha256_file(path)
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    (source / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(QuarantineExclusionRegistryError, match="record"):
        build_registry([source], tmp_path / "registry")
    assert not (tmp_path / "registry").exists()
