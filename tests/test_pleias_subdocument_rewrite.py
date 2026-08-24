import hashlib

import pytest

from sai.data.frequency_length_subdocument_deduplication import (
    _normalized_chunk,
    segment_subdocuments,
)
from sai.data.pleias_bounded_mechanical_candidates import CANDIDATE_SCHEMA
from sai.data.pleias_subdocument_rewrite import (
    PleiasSubdocumentRewriteError,
    rewrite_candidate,
)


def _candidate(text):
    return {
        "schema": CANDIDATE_SCHEMA,
        "source_row_identity_sha256": "a" * 64,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "collection": "Books",
        "word_count": len(text.split()),
        "token_count": len(text.split()) * 2,
        "semantic_stratum": "Books::Open Culture::medium",
        "semantic_quality_floor_milli": 7_500,
        "semantic_quality_mean_milli": 8_000,
        "text": text,
        "training_ready": False,
    }


def test_rewrite_removes_verified_large_duplicate_chunk():
    repeated = (
        "A repeated copyright navigation statement with extended boilerplate "
        "and legal wording " * 3
    )
    text = f"A unique discussion of astronomy.\n\n{repeated}"
    candidate = _candidate(text)
    chunks = segment_subdocuments(text)
    chunk = chunks[-1]
    normalized = _normalized_chunk(chunk["text"], code=chunk["code"])
    decision = (
        candidate["source_row_identity_sha256"],
        len(chunks) - 1,
        chunk["character_start"],
        chunk["character_end"],
        hashlib.sha256(normalized.encode()).hexdigest(),
        20,
        1,
    )
    result, counts = rewrite_candidate(candidate, 0, [decision])
    assert repeated not in result["text"]
    assert counts["deleted_chunks"] == 1
    assert result["pre_dedup_content_sha256"] == candidate["content_sha256"]
    assert result["content_sha256"] != candidate["content_sha256"]
    assert result["training_ready"] is False


def test_rewrite_restores_small_chunk_and_rejects_hash_mutation():
    text = "A unique opening.\nA small repeated footer."
    candidate = _candidate(text)
    chunks = segment_subdocuments(text, minimum_characters=32)
    chunk = chunks[-1]
    normalized = _normalized_chunk(chunk["text"], code=chunk["code"])
    decision = (
        candidate["source_row_identity_sha256"],
        len(chunks) - 1,
        chunk["character_start"],
        chunk["character_end"],
        hashlib.sha256(normalized.encode()).hexdigest(),
        3,
        1,
    )
    result, counts = rewrite_candidate(candidate, 0, [decision])
    assert result["text"] == text
    assert counts["coherence_restored_chunks"] == 1
    changed = list(decision)
    changed[4] = "f" * 64
    with pytest.raises(PleiasSubdocumentRewriteError, match="chunk replay"):
        rewrite_candidate(candidate, 0, [tuple(changed)])
