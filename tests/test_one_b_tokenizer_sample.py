from __future__ import annotations

import hashlib

import pytest

from sai.data.one_b_tokenizer_sample import _bounded_by_stratum
from sai.data.token_stream import TOKENIZER_ROW_SCHEMA, normalize_tokenizer_document


def _document(text: str, row_id: str) -> dict:
    identity = hashlib.sha256(row_id.encode()).hexdigest()
    return {
        "schema": TOKENIZER_ROW_SCHEMA,
        "text": text,
        "selection_identity_sha256": identity,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "tokenizer_training_only": True,
        "source": {
            "dataset": "test",
            "row_id": row_id,
            "license": "CC0",
            "domain": "english",
        },
    }


def test_bounded_selection_is_deterministic_and_stratified() -> None:
    rows = []
    for index in range(20):
        document = normalize_tokenizer_document(
            _document("x" * 100_000, str(index))
        )
        rows.append(
            ("easy" if index % 2 else "hard", document["identity_sha256"], document)
        )
    first, counts = _bounded_by_stratum(iter(rows), 1_000_000)
    second, _ = _bounded_by_stratum(iter(rows), 1_000_000)
    assert first == second
    assert sum(map(len, first)) <= 1_000_000
    assert counts["stratum::easy::documents"] == 10
    assert counts["stratum::hard::documents"] == 10
    assert hashlib.sha256(b"".join(first)).hexdigest()


def test_bounded_selection_rejects_tiny_cap() -> None:
    with pytest.raises(Exception, match="byte cap"):
        _bounded_by_stratum(iter(()), 999_999)
