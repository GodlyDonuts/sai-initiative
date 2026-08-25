from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.token_stream import canonical_sha256
from sai.training.one_b_readiness import OneBReadinessError, _load_signed


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
