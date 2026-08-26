from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.one_b_hf_publish import (
    PREFIX,
    REPOSITORY,
    SCHEMA,
    OneBHfPublishError,
    _portable_configs,
)


def test_packed_release_uses_content_addressed_immutable_prefix() -> None:
    assert REPOSITORY == "Godlydonuts/Sai"
    assert PREFIX == "training/packed/one-b/20260826-r2"
    assert SCHEMA == "sai-1b-packed-hf-publication-v2"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_portable_configs_bind_data_and_tokenizer_to_release_root(
    tmp_path: Path,
) -> None:
    data = tmp_path / "source.bin"
    data.write_bytes(b"\x01\x00" * 4_096)
    config = {
        "data": {"paths": [str(data), str(data)]},
        "tokenizer": {"identifier": "/private/tokenizer"},
        "load_path": "__REQUIRED_PREVIOUS_STAGE_BOUNDARY_CHECKPOINT__",
    }
    config_path = tmp_path / "stage-1-body.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    configs = {
        "configs": [
            {
                "stage": 1,
                "phase": "body",
                "path": config_path.name,
                "bytes": config_path.stat().st_size,
                "sha256": _sha(config_path),
            }
        ],
    }
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    packed = [
        {
            "local_path": str(data),
            "remote_path": f"{PREFIX}/data/ab/abcdef.bin",
        }
    ]
    uploads = _portable_configs(configs, tmp_path, packed, metadata)
    portable = json.loads(Path(uploads[0]["local_path"]).read_text())
    assert portable["data"]["paths"] == [
        "data/ab/abcdef.bin",
        "data/ab/abcdef.bin",
    ]
    assert portable["tokenizer"]["identifier"] == "tokenizer"
    assert portable["load_path"] == config["load_path"]


def test_portable_configs_reject_unpublished_data_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.bin"
    config_path = tmp_path / "stage-0-body.json"
    config_path.write_text(
        json.dumps(
            {
                "data": {"paths": [str(missing)]},
                "tokenizer": {"identifier": "/private/tokenizer"},
                "load_path": None,
            }
        ),
        encoding="utf-8",
    )
    configs = {
        "configs": [
            {
                "stage": 0,
                "phase": "body",
                "path": config_path.name,
                "bytes": config_path.stat().st_size,
                "sha256": _sha(config_path),
            }
        ],
    }
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    with pytest.raises(OneBHfPublishError, match="absent from publication"):
        _portable_configs(configs, tmp_path, [], metadata)
