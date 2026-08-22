from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from sai.data.authored_review_workspace import (
    AuthoredReviewWorkspaceError,
    build,
    validate,
)

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts" / "authored-curriculum-sources-r1"


def _inputs(tmp_path: Path) -> dict:
    packet = ARTIFACT / "authored-curriculum-blind-review.jsonl"
    receipt = ARTIFACT / "authored-curriculum-review-receipt.json"
    return {
        "review_packet": packet,
        "review_packet_receipt": receipt,
        "expected_review_packet_sha256": hashlib.sha256(
            packet.read_bytes()
        ).hexdigest(),
        "expected_review_packet_receipt_sha256": hashlib.sha256(
            receipt.read_bytes()
        ).hexdigest(),
        "concept_list": ROOT
        / "docs"
        / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json",
        "annotation_policy": ROOT / "docs" / "SAI_SEMANTIC_ANNOTATION_POLICY.json",
        "workspace_output": tmp_path / "review.html",
        "receipt_output": tmp_path / "review.receipt.json",
    }


def test_builds_replayable_offline_blind_workspace(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    payload = build(**inputs)
    assert payload["rows"] == 127
    assert payload["offline_only"] is True
    assert payload["external_requests"] is False
    assert payload["hidden_review_key_included"] is False
    assert payload["human_review_completed"] is False
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    page = inputs["workspace_output"].read_text()
    match = re.search(
        r'<script id="sai-data" type="application/json">(.*?)</script>', page, re.S
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert len(embedded["packet"]) == 127
    assert set(embedded["packet"][0]) == {"review_identity_sha256", "text"}
    assert "source_id" not in match.group(1)
    assert "fetch(" not in page
    assert "XMLHttpRequest" not in page
    assert ".join('\\n')+'\\n'" in page
    assert validate(**inputs) == payload


def test_workspace_refuses_input_and_output_tamper(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    with pytest.raises(AuthoredReviewWorkspaceError, match="inputs"):
        build(**{**inputs, "expected_review_packet_sha256": "0" * 64})
    build(**inputs)
    inputs["workspace_output"].chmod(0o644)
    inputs["workspace_output"].write_text(inputs["workspace_output"].read_text() + " ")
    with pytest.raises(AuthoredReviewWorkspaceError, match="workspace differs"):
        validate(**inputs)


def test_workspace_is_create_only(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    build(**inputs)
    with pytest.raises(AuthoredReviewWorkspaceError, match="output boundary"):
        build(**inputs)
