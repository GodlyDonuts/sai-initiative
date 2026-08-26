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
    _packed_files,
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
    packed_paths = {str(data.resolve()): f"{PREFIX}/data/ab/abcdef.bin"}
    uploads = _portable_configs(configs, tmp_path, packed_paths, metadata)
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
        _portable_configs(configs, tmp_path, {}, metadata)


def test_portable_configs_preserve_content_addressed_path_aliases(
    tmp_path: Path,
) -> None:
    first = tmp_path / "stage-0-boundary.bin"
    alias = tmp_path / "stage-2-boundary.bin"
    first.write_bytes(b"\x01\x00" * 4_096)
    alias.write_bytes(first.read_bytes())
    config = {
        "data": {"paths": [str(alias)]},
        "tokenizer": {"identifier": "/private/tokenizer"},
        "load_path": "__REQUIRED_CURRENT_STAGE_BODY_CHECKPOINT__",
    }
    config_path = tmp_path / "stage-2-boundary.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    configs = {
        "configs": [
            {
                "stage": 2,
                "phase": "boundary",
                "path": config_path.name,
                "bytes": config_path.stat().st_size,
                "sha256": _sha(config_path),
            }
        ],
    }
    remote = f"{PREFIX}/data/99/shared.bin"
    metadata = tmp_path / "metadata"
    metadata.mkdir()

    uploads = _portable_configs(
        configs,
        tmp_path,
        {str(first.resolve()): remote, str(alias.resolve()): remote},
        metadata,
    )

    portable = json.loads(Path(uploads[0]["local_path"]).read_text())
    assert portable["data"]["paths"] == ["data/99/shared.bin"]


def test_packed_files_retains_all_path_aliases_for_one_content_identity(
    tmp_path: Path,
) -> None:
    first = tmp_path / "stage-0-boundary.bin"
    alias = tmp_path / "stage-2-boundary.bin"
    content = b"\x01\x00" * 4_096
    first.write_bytes(content)
    alias.write_bytes(content)
    identity = _sha(first)
    entry = {
        "sha256": identity,
        "sequences_per_repeat": 1,
        "tokens_per_repeat": 4_096,
        "repeat": 1,
        "band": "foundation",
        "source": "fixture",
    }
    schedule = {
        "stages": [
            {
                "index": 0,
                "stage": "foundation",
                "body_entries": [{**entry, "path": str(first)}],
                "boundary_entries": [{**entry, "path": str(alias)}],
            }
        ]
    }

    packed, exposures, path_remotes = _packed_files(schedule)

    assert len(packed) == 1
    assert len(exposures) == 2
    assert set(path_remotes) == {str(first.resolve()), str(alias.resolve())}
    assert len(set(path_remotes.values())) == 1
