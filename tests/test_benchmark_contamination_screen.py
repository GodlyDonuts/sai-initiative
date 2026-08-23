from __future__ import annotations

import hashlib

from sai.data.agent_labeling import CANDIDATE_SCHEMA
from sai.data.benchmark_contamination_screen import summarize
from sai.data.decontamination import _text_shingles
from sai.data.token_stream import canonical_sha256


def _candidate(text: str, row_id: str) -> dict:
    row = {
        "schema": CANDIDATE_SCHEMA,
        "text": text,
        "source": {
            "dataset": "example/data",
            "revision": "v1",
            "row_id": row_id,
            "license": "CC-BY-4.0",
            "source_type": "reference",
        },
        "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "provenance_sha256": hashlib.sha256(row_id.encode()).hexdigest(),
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def test_screen_aggregates_overlap_without_persisting_source_text() -> None:
    boundary_text = (
        "one two three four five six seven eight nine ten eleven twelve thirteen"
    )
    word_boundary, code_boundary = _text_shingles(boundary_text)
    candidates = [
        _candidate((boundary_text + " ") * 4, "contaminated"),
        _candidate(
            "Unique archival astronomy and engineering explanation. " * 8, "clean"
        ),
    ]
    result = summarize(
        candidates,
        [
            {"source_id": "source_a", "stratum": "one"},
            {"source_id": "source_b", "stratum": "two"},
        ],
        word_boundary,
        code_boundary,
    )
    assert result["rows"] == 2
    assert result["contaminated_rows"] == 1
    assert result["clean_rows"] == 1
    assert result["by_source"]["source_a"]["contaminated_rows"] == 1
    assert result["by_source"]["source_b"].get("contaminated_rows", 0) == 0
    assert result["individual_decisions_persisted"] is False
    assert result["source_text_persisted"] is False
