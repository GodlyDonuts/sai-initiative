from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import sai.data.finemath_review_workspace as workspace_module
from sai.data.finemath_review_workspace import (
    FineMathReviewWorkspaceError,
    build,
    validate,
)
from sai.data.token_stream import canonical_sha256

ROOT = Path(__file__).parents[1]


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    rows = []
    for index in range(3):
        text = f"A complete mathematical explanation for blinded example {index}."
        rows.append(
            {
                "schema": "sai-finemath-language-ladder-blind-review-v1",
                "review_identity_sha256": hashlib.sha256(
                    f"identity-{index}".encode()
                ).hexdigest(),
                "source_url": f"https://example.invalid/{index}",
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "text": text,
            }
        )
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    packet = tmp_path / "blind.jsonl"
    packet.write_bytes(encoded)
    receipt = tmp_path / "ladder.receipt.json"
    receipt.write_text("{}\n")
    ladder = {
        "receipt_sha256": "1" * 64,
        "summary": {"blind_review_rows": len(rows)},
        "blind_review_output": {
            "path": str(packet),
            "rows": len(rows),
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "ordered_rows_sha256": canonical_sha256(rows),
        },
    }
    monkeypatch.setattr(workspace_module, "validate_ladder", lambda _: ladder)
    return {
        "ladder_receipt": receipt,
        "expected_ladder_receipt_sha256": hashlib.sha256(
            receipt.read_bytes()
        ).hexdigest(),
        "workspace_output": tmp_path / "review.html",
        "receipt_output": tmp_path / "review.receipt.json",
    }


def test_builds_replayable_blinded_offline_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    payload = build(**inputs)
    assert payload["rows"] == 3
    assert payload["offline_only"] is True
    assert payload["external_requests"] is False
    assert payload["hidden_review_key_included"] is False
    assert payload["language_scores_included"] is False
    assert payload["source_urls_included"] is False
    assert payload["training_authorized"] is False
    page = inputs["workspace_output"].read_text()
    match = re.search(
        r'<script id="sai-data" type="application/json">(.*?)</script>', page, re.S
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert len(embedded["packet"]) == 3
    assert set(embedded["packet"][0]) == {"review_identity_sha256", "text"}
    assert "language_score" not in match.group(1)
    assert "stratum" not in match.group(1)
    assert "source_url" not in match.group(1)
    assert "ladder-review-key" not in page
    assert "fetch(" not in page
    assert "XMLHttpRequest" not in page
    assert "sai-finemath-human-review-progress-v1" in page
    assert "sai-finemath-human-quality-review-v1" in page
    assert validate(**inputs) == payload


def test_workspace_refuses_tamper_and_wrong_receipt_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    with pytest.raises(FineMathReviewWorkspaceError, match="expected ladder"):
        build(**{**inputs, "expected_ladder_receipt_sha256": "0" * 64})
    build(**inputs)
    inputs["workspace_output"].chmod(0o644)
    inputs["workspace_output"].write_text(inputs["workspace_output"].read_text() + " ")
    with pytest.raises(FineMathReviewWorkspaceError, match="workspace differs"):
        validate(**inputs)


def test_workspace_is_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path, monkeypatch)
    build(**inputs)
    with pytest.raises(FineMathReviewWorkspaceError, match="output boundary"):
        build(**inputs)
