from sai.data.pleias_bounded_mechanical_candidates import CANDIDATE_SCHEMA
from sai.data.pleias_subdocument_signature import signature_rows


def test_signature_rows_are_lossless_locators_without_source_text():
    import hashlib

    text = (
        "A unique discussion of music and geometry.\n\n"
        "A separate explanation of biology and information theory."
    )
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "source_row_identity_sha256": "a" * 64,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "collection": "Books",
        "text": text,
        "training_ready": False,
    }
    rows = signature_rows(candidate, 3, 7)
    assert len(rows) == 2
    assert all("text" not in row for row in rows)
    assert rows[0]["source_shard"] == 3
    assert rows[0]["source_row_index"] == 7
    assert rows[0]["character_start"] == 0
    assert rows[-1]["character_end"] == len(text)
    assert all(row["training_ready"] is False for row in rows)
