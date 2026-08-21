from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from sai.data.hf_model_snapshot import (
    FILES,
    REVISION,
    ModelSnapshotError,
    restore_snapshot,
    validate_snapshot,
)


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def test_pinned_qwen_snapshot_binds_exact_revision_and_weight() -> None:
    assert REVISION == "2fc06364715b967f1860aea9cf38778875588b17"
    weight = FILES["model.safetensors-00001-of-00001.safetensors"]
    assert weight == {
        "size": 1_746_942_600,
        "sha256": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    }
    assert FILES["tokenizer.json"]["sha256"] == (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    )
    assert len(FILES) == 13


def test_restores_validates_and_rejects_tamper(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    sources.mkdir()
    payloads = {"a.txt": b"alpha", "b.bin": b"beta"}
    files = {
        "a.txt": {"size": 5, "git_blob_sha1": _git_blob(b"alpha")},
        "b.bin": {"size": 4, "sha256": hashlib.sha256(b"beta").hexdigest()},
    }
    for name, data in payloads.items():
        (sources / name).write_bytes(data)
    cache = tmp_path / "cache"
    cache.mkdir()

    def download(name: str, _: Path) -> Path:
        return sources / name

    output = tmp_path / "model"
    receipt = restore_snapshot(output, cache, download=download, files=files)
    assert validate_snapshot(output, files=files) == receipt
    assert receipt["file_count"] == 2
    assert receipt["total_bytes"] == 9
    assert not os.stat(output).st_mode & 0o222
    assert all(not os.stat(path).st_mode & 0o222 for path in output.iterdir())

    os.chmod(output / "a.txt", 0o644)
    with pytest.raises(ModelSnapshotError, match="unsafe or writable"):
        validate_snapshot(output, files=files)


def test_job_is_cpu_only_exact_and_no_retry() -> None:
    job = (
        Path(__file__).resolve().parents[1]
        / "jobs"
        / "sai-restore-qwen35-0p8b-cpu.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in job
    assert "#SBATCH --no-requeue" in job
    assert "SLURM_TMPDIR:-/tmp" in job
    assert "4_000_000_000" in job
    assert 'find "$cache_root" -xdev -depth -delete' in job
    assert "retry" not in job.lower()
