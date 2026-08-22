"""Replay an immutable externally restored Hugging Face model snapshot.

This adapter lets Sai reuse an already authenticated, sealed model tree without
copying multi-gigabyte weights or trusting a path alone.  The external manifest
and restoration receipt are both byte-pinned, and every model member is reopened
and hashed before the snapshot is admitted.
"""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ExternalSnapshotError(RuntimeError):
    """An external manifest, receipt, or model member differs."""


@dataclass(frozen=True)
class ExternalSnapshotSpec:
    repository: str
    revision: str
    tree_sha256: str
    manifest_sha256: str
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not self.repository or not self.revision:
            raise ExternalSnapshotError("external snapshot identity is empty")
        for value in (
            self.tree_sha256,
            self.manifest_sha256,
            self.receipt_sha256,
        ):
            if (
                len(value) != 64
                or value != value.lower()
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ExternalSnapshotError("external snapshot SHA256 differs")


def sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ExternalSnapshotError("external snapshot artifact is missing or unsafe")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalSnapshotError("external snapshot JSON is unreadable") from error
    if not isinstance(payload, dict):
        raise ExternalSnapshotError("external snapshot JSON must be an object")
    return payload


def validate_external_snapshot(
    root: Path,
    *,
    manifest_path: Path,
    receipt_path: Path,
    spec: ExternalSnapshotSpec,
) -> dict[str, Any]:
    """Reopen every member and return normalized immutable snapshot evidence."""

    root = Path(root)
    manifest_path = Path(manifest_path)
    receipt_path = Path(receipt_path)
    if (
        not root.is_dir()
        or root.is_symlink()
        or stat.S_IMODE(root.stat().st_mode) & 0o222
        or sha256_file(manifest_path) != spec.manifest_sha256
        or sha256_file(receipt_path) != spec.receipt_sha256
    ):
        raise ExternalSnapshotError("external snapshot boundary differs")
    manifest = _load_object(manifest_path)
    receipt = _load_object(receipt_path)
    rows = manifest.get("files")
    if (
        manifest.get("schema") != "shohin-dense-model-manifest-v1"
        or manifest.get("repository") != spec.repository
        or manifest.get("revision") != spec.revision
        or manifest.get("tree_sha256") != spec.tree_sha256
        or not isinstance(rows, list)
        or not rows
        or receipt.get("schema") != "shohin-dense-model-restoration-v1"
        or receipt.get("status") != "complete"
        or receipt.get("repository") != spec.repository
        or receipt.get("model_revision") != spec.revision
        or receipt.get("tree_sha256") != spec.tree_sha256
        or receipt.get("manifest_sha256") != spec.manifest_sha256
        or receipt.get("manifest_verified") is not True
        or receipt.get("model_root") != str(root.resolve())
        or receipt.get("manifest") != str(manifest_path.resolve())
        or receipt.get("symlinks") != 0
        or receipt.get("special_files") != 0
    ):
        raise ExternalSnapshotError("external snapshot identity differs")

    normalized = []
    names: set[str] = set()
    total_bytes = 0
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ExternalSnapshotError("external snapshot manifest row differs")
        name = row.get("path")
        size = row.get("bytes")
        digest = row.get("sha256")
        relative = Path(name) if isinstance(name, str) else Path(".")
        if (
            not isinstance(name, str)
            or not name
            or relative.is_absolute()
            or len(relative.parts) != 1
            or name in names
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ExternalSnapshotError("external snapshot member descriptor differs")
        member = root / name
        if (
            not member.is_file()
            or member.is_symlink()
            or member.stat().st_nlink != 1
            or stat.S_IMODE(member.stat().st_mode) & 0o222
            or member.stat().st_size != size
            or sha256_file(member) != digest
        ):
            raise ExternalSnapshotError("external snapshot member differs")
        names.add(name)
        total_bytes += size
        normalized.append({"path": name, "bytes": size, "sha256": digest})

    if (
        {path.name for path in root.iterdir()} != names
        or manifest.get("bytes") != total_bytes
        or receipt.get("bytes") != total_bytes
        or receipt.get("files") != len(normalized)
    ):
        raise ExternalSnapshotError("external snapshot tree geometry differs")
    return {
        "repository": spec.repository,
        "revision": spec.revision,
        "tree_sha256": spec.tree_sha256,
        "manifest_sha256": spec.manifest_sha256,
        "receipt_file_sha256": spec.receipt_sha256,
        "file_count": len(normalized),
        "total_bytes": total_bytes,
        "files": normalized,
    }
