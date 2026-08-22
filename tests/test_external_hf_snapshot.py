from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from sai.data.external_hf_snapshot import (
    ExternalSnapshotError,
    ExternalSnapshotSpec,
    validate_external_snapshot,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path: Path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_bytes(b"{}\n")
    (root / "weights.bin").write_bytes(b"weights")
    for path in root.iterdir():
        os.chmod(path, 0o444)
    os.chmod(root, 0o555)
    rows = [
        {"path": "config.json", "bytes": 3, "sha256": _sha(b"{}\n")},
        {"path": "weights.bin", "bytes": 7, "sha256": _sha(b"weights")},
    ]
    manifest = tmp_path / "model.manifest.json"
    manifest_payload = {
        "schema": "shohin-dense-model-manifest-v1",
        "repository": "example/model",
        "revision": "1" * 40,
        "tree_sha256": "2" * 64,
        "bytes": 10,
        "files": rows,
    }
    manifest.write_text(json.dumps(manifest_payload, sort_keys=True))
    receipt = tmp_path / "model.receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "shohin-dense-model-restoration-v1",
                "status": "complete",
                "repository": "example/model",
                "model_revision": "1" * 40,
                "tree_sha256": "2" * 64,
                "manifest": str(manifest.resolve()),
                "manifest_sha256": _sha(manifest.read_bytes()),
                "manifest_verified": True,
                "model_root": str(root.resolve()),
                "bytes": 10,
                "files": 2,
                "symlinks": 0,
                "special_files": 0,
            },
            sort_keys=True,
        )
    )
    spec = ExternalSnapshotSpec(
        repository="example/model",
        revision="1" * 40,
        tree_sha256="2" * 64,
        manifest_sha256=_sha(manifest.read_bytes()),
        receipt_sha256=_sha(receipt.read_bytes()),
    )
    return root, manifest, receipt, spec


def test_external_snapshot_reopens_every_member(tmp_path: Path) -> None:
    root, manifest, receipt, spec = _fixture(tmp_path)
    result = validate_external_snapshot(
        root, manifest_path=manifest, receipt_path=receipt, spec=spec
    )
    assert result["file_count"] == 2
    assert result["total_bytes"] == 10
    assert result["tree_sha256"] == "2" * 64


def test_external_snapshot_rejects_member_tamper(tmp_path: Path) -> None:
    root, manifest, receipt, spec = _fixture(tmp_path)
    os.chmod(root / "weights.bin", 0o644)
    (root / "weights.bin").write_bytes(b"changed")
    os.chmod(root / "weights.bin", 0o444)
    with pytest.raises(ExternalSnapshotError, match="member differs"):
        validate_external_snapshot(
            root, manifest_path=manifest, receipt_path=receipt, spec=spec
        )


def test_external_snapshot_rejects_resigned_manifest(tmp_path: Path) -> None:
    root, manifest, receipt, spec = _fixture(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["repository"] = "different/model"
    manifest.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(ExternalSnapshotError, match="boundary differs"):
        validate_external_snapshot(
            root, manifest_path=manifest, receipt_path=receipt, spec=spec
        )
