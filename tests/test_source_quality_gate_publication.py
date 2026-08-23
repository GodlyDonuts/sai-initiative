import json

import pytest

from sai.data.source_quality_gate import build
from sai.data.source_quality_gate_publication import (
    SourceQualityGatePublicationError,
    build_publication,
)
from tests.test_source_quality_gate import candidate


def write_population(root, rows):
    root.mkdir()
    source = root / "candidates.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = root / "receipt.json"
    build(source, root / "decisions.jsonl", receipt)
    return receipt


def test_publication_replays_counts_and_cross_population_overlap(tmp_path) -> None:
    shared = candidate("Shared coherent educational source. " * 12, 1)
    first = write_population(
        tmp_path / "first",
        [shared, candidate("First distinct coherent source. " * 12, 2)],
    )
    second = write_population(
        tmp_path / "second",
        [shared, candidate("Second distinct coherent source. " * 12, 3)],
    )
    output = tmp_path / "publication.json"
    result = build_publication([first, second], output)
    assert result["population_assignment_rows"] == 4
    assert result["unique_candidate_rows"] == 3
    assert result["cross_population_duplicate_identity_rows"] == 1
    assert result["cross_population_duplicate_assignments"] == 1
    assert result["unique_source_content_rows"] == 3
    assert result["cross_population_duplicate_content_rows"] == 1
    assert result["cross_population_duplicate_content_assignments"] == 1
    assert result["decision_counts"] == {"pass_mechanical_gate": 4}
    assert result["publication_contains_source_text"] is False
    assert result["mechanical_pass_is_semantic_admission"] is False
    assert result["training_ready"] is False


def test_publication_detects_same_content_under_distinct_identities(tmp_path) -> None:
    text = "The same exact source content under two source locators. " * 12
    first = write_population(tmp_path / "first", [candidate(text, 1)])
    second = write_population(tmp_path / "second", [candidate(text, 2)])
    result = build_publication([first, second], tmp_path / "content-publication.json")
    assert result["unique_candidate_rows"] == 2
    assert result["cross_population_duplicate_identity_rows"] == 0
    assert result["unique_source_content_rows"] == 1
    assert result["cross_population_duplicate_content_rows"] == 1
    assert result["cross_population_duplicate_content_assignments"] == 1


def test_publication_rejects_tampered_gate(tmp_path) -> None:
    receipt = write_population(
        tmp_path / "population",
        [candidate("A coherent educational source. " * 12, 1)],
    )
    payload = json.loads(receipt.read_text())
    payload["decision_counts"] = {"hard_reject": 1}
    receipt.write_text(json.dumps(payload))
    with pytest.raises(SourceQualityGatePublicationError, match="replay"):
        build_publication([receipt], tmp_path / "publication.json")
