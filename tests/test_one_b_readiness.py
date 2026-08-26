from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.token_stream import canonical_sha256
from sai.training.one_b_readiness import (
    OneBReadinessError,
    _load_signed,
    _never_trained,
    _portable_publication_ready,
)


def test_load_signed_rejects_tamper(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    value = {"schema": "example", "status": "complete"}
    value["receipt_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    assert _load_signed(path, "example") == value
    value["status"] = "tampered"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(OneBReadinessError, match="signed readiness input differs"):
        _load_signed(path, "example")


def test_never_trained_requires_explicit_false_authorization() -> None:
    _never_trained(
        {"model_training_started": False, "one_b_training_authorized": False}
    )
    with pytest.raises(OneBReadinessError, match="authorization boundary differs"):
        _never_trained({"model_training_started": False})


def test_portable_publication_requires_every_phase_config() -> None:
    config = {"configs": [{"path": "a.json"}, {"path": "b.json"}]}
    publication = {
        "portable_config_files": 2,
        "portable_configs_sha256": "a" * 64,
        "portable_paths_are_release_relative": True,
        "portable_checkpoint_placeholders_preserved": True,
    }
    assert _portable_publication_ready(publication, config)
    publication["portable_config_files"] = 1
    assert not _portable_publication_ready(publication, config)
