from __future__ import annotations

import hashlib

from sai.data.decontamination import _WORD, POLICY, _normalize
from sai.data.institutional_books_full_decontamination import screen_book
from sai.data.institutional_books_independent_agreement import RECORD_SCHEMA
from sai.data.token_stream import canonical_sha256


def _agreement() -> dict:
    row = {
        "schema": RECORD_SCHEMA,
        "candidate_identity_sha256": "a" * 64,
        "source_book_id": "book-1",
        "disposition": "consensus_candidate",
        "training_ready": False,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def test_full_book_screen_detects_exact_official_word_shingle() -> None:
    text = (
        "one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen"
    )
    tokens = _WORD.findall(_normalize(text))
    digest = bytes.fromhex(canonical_sha256(tokens[: POLICY["word_shingle_tokens"]]))
    decision = screen_book(
        _agreement(), text, hashlib.sha256(text.encode()).hexdigest(), {digest}, set()
    )
    assert decision["contaminated"] is True
    assert decision["word_overlap_count"] == 1
    assert decision["full_source_text_persisted"] is False


def test_full_book_screen_keeps_clean_text_without_persisting_it() -> None:
    text = "A distinct history of orchards, irrigation, grafting, and seasonal care."
    decision = screen_book(
        _agreement(), text, hashlib.sha256(text.encode()).hexdigest(), set(), set()
    )
    assert decision["contaminated"] is False
    assert text not in str(decision)
