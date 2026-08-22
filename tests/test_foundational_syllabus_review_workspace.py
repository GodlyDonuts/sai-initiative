from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sai.data.foundational_syllabus_review_workspace import (
    FoundationalSyllabusReviewWorkspaceError,
    build,
    validate,
)

ROOT = Path(__file__).parents[1]


def _inputs(tmp_path: Path) -> dict:
    return {
        "base_concepts": ROOT
        / "docs"
        / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json",
        "additions": ROOT
        / "docs"
        / "SAI_FOUNDATIONAL_SYLLABUS_ADDITIONS_CANDIDATE.json",
        "workspace_output": tmp_path / "review.html",
        "receipt_output": tmp_path / "receipt.json",
    }


def test_builds_complete_offline_graph_review_workspace(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    payload = build(**inputs)
    assert payload["concepts"] == 125
    assert payload["hard_edges"] == 263
    assert payload["flagged_concepts"] == 67
    assert payload["offline_only"] is True
    assert payload["external_requests"] is False
    assert payload["subject_review_completed"] is False
    assert payload["training_authorized"] is False
    page = inputs["workspace_output"].read_text()
    match = re.search(
        r'<script id="sai-data" type="application/json">(.*?)</script>', page, re.S
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert len(embedded["rows"]) == 125
    assert sum(len(row["prerequisites"]) for row in embedded["rows"]) == 263
    assert "hard" in page and "supporting" in page and "remove" in page
    assert "sai-foundational-syllabus-review-progress-v1" in page
    assert "fetch(" not in page
    assert "XMLHttpRequest" not in page
    assert validate(**inputs) == payload


def test_workspace_is_create_only_and_tamper_evident(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    build(**inputs)
    with pytest.raises(
        FoundationalSyllabusReviewWorkspaceError, match="output boundary"
    ):
        build(**inputs)
    inputs["workspace_output"].chmod(0o644)
    inputs["workspace_output"].write_text(inputs["workspace_output"].read_text() + " ")
    with pytest.raises(FoundationalSyllabusReviewWorkspaceError, match="workspace"):
        validate(**inputs)
