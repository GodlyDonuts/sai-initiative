import hashlib
import json

from sai.data.frequency_length_subdocument_deduplication import (
    _normalized_chunk,
    segment_subdocuments,
)
from sai.data.institutional_books_cross_source_subdocument_rewrite import (
    OUTPUT_SCHEMA,
    rewrite_row,
)
from sai.data.institutional_books_full_decontamination import CLEAN_SCHEMA
from sai.data.institutional_books_materializer import OUTPUT_SCHEMA as SOURCE_SCHEMA


def test_private_book_rewrite_preserves_source_lineage_and_hashes_output():
    unique = "An original discussion of artistic perspective and geometry. " * 8
    duplicate = "A duplicated archive footer and catalog statement. " * 8
    text = f"{unique}\n\n{duplicate}"
    identity = "a" * 64
    source_sha = hashlib.sha256(text.encode()).hexdigest()
    row = {
        "schema": SOURCE_SCHEMA,
        "barcode_src": "book-1",
        "text": text,
        "source_content_sha256": source_sha,
        "source_path": "parent.parquet",
        "training_ready": False,
    }
    clean = {
        "schema": CLEAN_SCHEMA,
        "source_book_id": "book-1",
        "candidate_identity_sha256": identity,
        "full_source_content_sha256": source_sha,
        "agreement_record_sha256": "b" * 64,
        "decontamination_record_sha256": "c" * 64,
        "agreed_genre": "technical_nonfiction",
        "shared_domains": ["physics", "art_history"],
        "benchmark_decontamination_complete": True,
        "training_ready": False,
    }
    curriculum = {
        "work_id_candidates": ["work-a"],
        "shared_prerequisites": ["geometry"],
        "shared_concepts": ["perspective"],
        "source_text_persisted": False,
    }
    from sai.data.token_stream import canonical_sha256

    curriculum["metadata_sha256"] = canonical_sha256(curriculum)
    clean["consensus_curriculum"] = curriculum
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
                30,
                1,
            )
        )
    result, counts = rewrite_row(row, clean, 0, decisions)
    assert result["schema"] == OUTPUT_SCHEMA
    assert result["source_content_sha256"] == source_sha
    assert result["pre_cross_source_content_sha256"] == source_sha
    assert result["content_sha256"] != source_sha
    assert result["source_path"] == "parent.parquet"
    assert result["semantic_genre"] == "technical_nonfiction"
    assert result["semantic_domains"] == ["physics", "art_history"]
    assert json.loads(result["curriculum_metadata_json"])["shared_concepts"] == [
        "perspective"
    ]
    assert result["corpus_split"] in {"train", "development"}
    assert result["token_count_requires_recomputation"] is True
    assert counts["deleted_chunks"] == 2
    assert result["training_ready"] is False
