import hashlib

from sai.data.agent_labeling import CANDIDATE_SCHEMA
from sai.data.reservoir_audit_duplicates import find_duplicate_pairs
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


def test_duplicate_audit_finds_cross_source_copy_and_ignores_unrelated_text() -> None:
    base = " ".join(f"concept{index}" for index in range(160))
    candidates = [
        _candidate(base, "a"),
        _candidate(base + " one additional ending", "b"),
        _candidate("unrelated evidence " * 100, "c"),
    ]
    lineage = [
        {"source_id": "one"},
        {"source_id": "two"},
        {"source_id": "three"},
    ]
    pairs = find_duplicate_pairs(candidates, lineage)
    assert len(pairs) == 1
    assert pairs[0]["cross_source"] is True
    assert "five_word_shingle_containment" in pairs[0]["reasons"]


def test_duplicate_audit_detects_normalized_token_identity() -> None:
    text = "Grounded science, mathematics, history, and art. " * 30
    candidates = [_candidate(text, "a"), _candidate(text.upper(), "b")]
    pairs = find_duplicate_pairs(
        candidates, [{"source_id": "one"}, {"source_id": "one"}]
    )
    assert len(pairs) == 1
    assert "normalized_token_exact" in pairs[0]["reasons"]


def test_duplicate_audit_does_not_merge_distinct_tokenless_content() -> None:
    candidates = [_candidate("😀" * 250, "a"), _candidate("💾" * 250, "b")]
    pairs = find_duplicate_pairs(
        candidates, [{"source_id": "one"}, {"source_id": "two"}]
    )
    assert pairs == []
