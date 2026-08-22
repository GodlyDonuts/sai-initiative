"""Restore and validate the exact pinned Qwen3.5-0.8B upstream snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-pinned-hf-model-snapshot-v1"
REPO_ID = "Qwen/Qwen3.5-0.8B"
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
FILES: dict[str, dict[str, Any]] = {
    ".gitattributes": {
        "size": 1_570,
        "git_blob_sha1": "52373fe24473b1aa44333d318f578ae6bf04b49b",
    },
    "LICENSE": {
        "size": 11_544,
        "git_blob_sha1": "f938136e3adacfd92be087f6e113b5d6d97f678f",
    },
    "README.md": {
        "size": 61_705,
        "git_blob_sha1": "5824f1761b2b3a55a2141a9a1172a7f92c7c2ad9",
    },
    "chat_template.jinja": {
        "size": 7_755,
        "git_blob_sha1": "0ef09f214eaa6d9bca297988afc1454b5827b2c7",
    },
    "config.json": {
        "size": 2_907,
        "git_blob_sha1": "715f0448b9d38103211f0ad88bbb4d6e4f4be8c9",
    },
    "merges.txt": {
        "size": 3_353_259,
        "git_blob_sha1": "a494e019ca1502219fd0128658b979e5f05ae8e8",
    },
    "model.safetensors-00001-of-00001.safetensors": {
        "size": 1_746_942_600,
        "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    },
    "model.safetensors.index.json": {
        "size": 50_900,
        "git_blob_sha1": "f691cefdb79d73270895ebd6d9594ddcecfc1838",
    },
    "preprocessor_config.json": {
        "size": 390,
        "git_blob_sha1": "2ea84a437d448ff71b08df68fdd949d5cc4ebb64",
    },
    "tokenizer.json": {
        "size": 12_807_982,
        "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    },
    "tokenizer_config.json": {
        "size": 16_709,
        "git_blob_sha1": "fae3ce993e07c092ad024dde45e592379fde91bb",
    },
    "video_preprocessor_config.json": {
        "size": 385,
        "git_blob_sha1": "3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe",
    },
    "vocab.json": {
        "size": 6_722_759,
        "git_blob_sha1": "0aa0ce0658d60ac4a5d609f4eadb0e8e43514176",
    },
}


class ModelSnapshotError(RuntimeError):
    """Pinned upstream bytes, output geometry, or receipt differs."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _verify_upstream(path: Path, descriptor: dict[str, Any]) -> str:
    if not path.is_file() or path.stat().st_size != descriptor["size"]:
        raise ModelSnapshotError("upstream model member size differs")
    observed_sha256 = _sha256_file(path)
    if "sha256" in descriptor:
        if observed_sha256 != descriptor["sha256"]:
            raise ModelSnapshotError("upstream LFS member SHA256 differs")
    elif _git_blob_sha1(path) != descriptor["git_blob_sha1"]:
        raise ModelSnapshotError("upstream Git blob identity differs")
    return observed_sha256


def _default_download(filename: str, cache_root: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise ModelSnapshotError("huggingface_hub is required") from error
    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            revision=REVISION,
            cache_dir=cache_root,
        )
    )


def restore_snapshot(
    output_root: Path,
    cache_root: Path,
    *,
    download: Callable[[str, Path], Path] = _default_download,
    files: dict[str, dict[str, Any]] = FILES,
) -> dict[str, Any]:
    """Download, authenticate, seal, and atomically publish one model tree."""

    output_root = Path(output_root)
    cache_root = Path(cache_root)
    if (
        output_root.exists()
        or output_root.is_symlink()
        or not output_root.parent.is_dir()
        or output_root.parent.is_symlink()
        or not cache_root.is_dir()
        or cache_root.is_symlink()
        or not files
    ):
        raise ModelSnapshotError("snapshot output or cache boundary differs")
    stage = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.stage"
    stage.mkdir(mode=0o700)
    descriptors = []
    try:
        for filename, expected in sorted(files.items()):
            if Path(filename).is_absolute() or len(Path(filename).parts) != 1:
                raise ModelSnapshotError("snapshot member name differs")
            source = Path(download(filename, cache_root))
            observed_sha256 = _verify_upstream(source, expected)
            target = stage / filename
            shutil.copyfile(source, target)
            os.chmod(target, 0o444)
            if _verify_upstream(target, expected) != observed_sha256:
                raise ModelSnapshotError("published member bytes differ")
            descriptors.append(
                {
                    "path": filename,
                    "bytes": target.stat().st_size,
                    "sha256": observed_sha256,
                    "upstream_identity": dict(expected),
                }
            )
        tree_sha256 = canonical_sha256(descriptors)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "repo_id": REPO_ID,
            "revision": REVISION,
            "files": descriptors,
            "file_count": len(descriptors),
            "total_bytes": sum(row["bytes"] for row in descriptors),
            "tree_sha256": tree_sha256,
            "expected_parameter_count": 852_985_920,
            "training_authorized": False,
            "four_b_training_authorized": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        receipt_path = stage / "snapshot.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        os.chmod(receipt_path, 0o444)
        manifest_rows = [(row["sha256"], row["path"]) for row in descriptors] + [
            (_sha256_file(receipt_path), receipt_path.name)
        ]
        manifest = stage / "SHA256SUMS"
        manifest.write_text(
            "".join(f"{digest}  {name}\n" for digest, name in manifest_rows)
        )
        os.chmod(manifest, 0o444)
        os.chmod(stage, 0o555)
        os.rename(stage, output_root)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    validate_snapshot(output_root, files=files)
    return receipt


def validate_snapshot(
    root: Path, *, files: dict[str, dict[str, Any]] = FILES
) -> dict[str, Any]:
    """Replay every upstream identity and the exact local receipt."""

    root = Path(root)
    expected_members = set(files) | {"snapshot.json", "SHA256SUMS"}
    if (
        not root.is_dir()
        or root.is_symlink()
        or stat.S_IMODE(root.stat().st_mode) & 0o222
        or {path.name for path in root.iterdir()} != expected_members
    ):
        raise ModelSnapshotError("snapshot tree geometry differs")
    descriptors = []
    for filename, expected in sorted(files.items()):
        path = root / filename
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise ModelSnapshotError("snapshot member is unsafe or writable")
        descriptors.append(
            {
                "path": filename,
                "bytes": path.stat().st_size,
                "sha256": _verify_upstream(path, expected),
                "upstream_identity": dict(expected),
            }
        )
    receipt_path = root / "snapshot.json"
    receipt = json.loads(receipt_path.read_text())
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("repo_id") != REPO_ID
        or receipt.get("revision") != REVISION
        or receipt.get("files") != descriptors
        or receipt.get("file_count") != len(descriptors)
        or receipt.get("total_bytes") != sum(row["bytes"] for row in descriptors)
        or receipt.get("tree_sha256") != canonical_sha256(descriptors)
        or receipt.get("training_authorized") is not False
        or receipt.get("four_b_training_authorized") is not False
        or claimed != canonical_sha256(unsigned)
    ):
        raise ModelSnapshotError("snapshot receipt differs")
    expected_manifest = "".join(
        f"{digest}  {name}\n"
        for digest, name in [
            *[(row["sha256"], row["path"]) for row in descriptors],
            (_sha256_file(receipt_path), receipt_path.name),
        ]
    )
    manifest = root / "SHA256SUMS"
    if manifest.read_text() != expected_manifest:
        raise ModelSnapshotError("snapshot manifest differs")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = restore_snapshot(args.output_root, args.cache_root)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "tree_sha256": receipt["tree_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
