from __future__ import annotations

from sai.data.decontamination import POLICY, _normalize, _shingles
from sai.data.public_domain_review_decontamination import screen_candidate
from sai.data.public_domain_review_scoped_candidates import CANDIDATE_SCHEMA
from sai.data.token_stream import canonical_sha256


def _candidate(text: str) -> dict:
    row = {
        "schema": CANDIDATE_SCHEMA,
        "text": text,
        "original_candidate_identity_sha256": "1" * 64,
        "record_sha256": "2" * 64,
    }
    return row


def test_post_scope_screen_detects_exact_word_shingle() -> None:
    text = "one two three four five six seven eight nine ten eleven twelve thirteen"
    boundary = _shingles(_normalize(text).split(), POLICY["word_shingle_tokens"])
    decision = screen_candidate(_candidate(text), boundary, set())
    assert decision["word_overlap_count"] == 1
    assert decision["code_overlap_count"] == 0
    assert decision["contaminated"] is True
    assert decision["source_text_persisted"] is False
    assert decision["training_ready"] is False
    assert decision["record_sha256"] == canonical_sha256(
        {key: value for key, value in decision.items() if key != "record_sha256"}
    )


def test_post_scope_screen_keeps_disjoint_short_prose() -> None:
    decision = screen_candidate(_candidate("A short cultural note."), set(), set())
    assert decision["word_overlap_count"] == 0
    assert decision["code_overlap_count"] == 0
    assert decision["contaminated"] is False
