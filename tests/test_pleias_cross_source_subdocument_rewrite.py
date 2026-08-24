import hashlib

from sai.data.frequency_length_subdocument_deduplication import (
    _normalized_chunk,
    segment_subdocuments,
)
from sai.data.pleias_cross_source_subdocument_rewrite import (
    OUTPUT_SCHEMA,
    rewrite_row,
)
from sai.data.pleias_subdocument_rewrite import OUTPUT_SCHEMA as SOURCE_SCHEMA


def test_rewrite_row_preserves_internal_lineage_and_rewrites_exact_chunks():
    unique = "A specific optics experiment and its measurements. " * 8
    duplicate = "A duplicated archive footer and navigation statement. " * 8
    text = f"{unique}\n\n{duplicate}"
    identity = "a" * 64
    row = {
        "schema": SOURCE_SCHEMA,
        "source_row_identity_sha256": identity,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "pre_dedup_content_sha256": "b" * 64,
        "subdocument_transform_sha256": "c" * 64,
        "collection": "Books",
        "semantic_stratum": "Books::Open Culture::medium",
        "semantic_quality_floor_milli": 7_500,
        "semantic_quality_mean_milli": 8_000,
        "word_count": len(text.split()),
        "token_count_requires_recomputation": True,
        "text": text,
        "training_ready": False,
    }
    chunks = segment_subdocuments(text)
    decisions = []
    for index in (len(chunks) - 2, len(chunks) - 1):
        chunk = chunks[index]
        normalized = _normalized_chunk(chunk["text"], code=chunk["code"])
        decisions.append(
            (
                identity,
                index,
                chunk["character_start"],
                chunk["character_end"],
                hashlib.sha256(normalized.encode()).hexdigest(),
                20,
                1,
            )
        )
    result, counts = rewrite_row(row, 0, decisions)
    assert result["schema"] == OUTPUT_SCHEMA
    assert result["pre_cross_source_content_sha256"] == row["content_sha256"]
    assert result["pre_dedup_content_sha256"] == "b" * 64
    assert result["subdocument_transform_sha256"] == "c" * 64
    assert result["semantic_quality_floor_milli"] == 7_500
    assert result["content_sha256"] != row["content_sha256"]
    assert counts["deleted_chunks"] == 2
    assert result["training_ready"] is False
