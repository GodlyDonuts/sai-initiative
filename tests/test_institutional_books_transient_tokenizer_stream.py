import hashlib
import io
import json
from pathlib import Path

import pytest

from sai.data.institutional_books_cross_source_subdocument_rewrite import OUTPUT_SCHEMA
from sai.data.institutional_books_quality_selection import (
    ROW_SCHEMA as SELECTION_ROW_SCHEMA,
)
from sai.data.institutional_books_transient_tokenizer_stream import (
    InstitutionalBooksTransientTokenizerStreamError,
    _envelope,
)
from sai.data.token_stream import canonical_sha256
from sai.data.transient_tokenizer_sample import (
    GENERIC_SOURCE_RECEIPT_SCHEMA,
    GENERIC_SOURCE_STATUS,
    SAMPLE_NAME,
    build_sample,
)


def _fixture() -> tuple[dict, dict]:
    text = "A rigorous public-domain book explains geometry and proof. " * 30
    curriculum = {
        "quality_floor": {"literary_value": 4, "knowledge_density": 5},
        "complexity_range": {
            "conceptual": {"minimum": 2, "maximum": 4},
            "linguistic": {"minimum": 2, "maximum": 3},
        },
        "work_id_candidates": ["work-1"],
        "edition_id_candidates": ["edition-1"],
        "shared_subdomains": ["geometry"],
        "styles": ["expository"],
        "curriculum_band_votes": ["foundation", "growth"],
        "shared_prerequisites": ["arithmetic"],
        "shared_concepts": ["proof"],
        "shared_period": ["nineteenth-century"],
        "shared_culture_geography": ["global"],
        "shared_recommended_representations": ["original"],
        "translation_type_votes": ["original-english"],
        "shared_concept_edges": [],
        "confidence_floor_ppm": 900_000,
        "source_text_persisted": False,
    }
    curriculum["metadata_sha256"] = canonical_sha256(curriculum)
    row = {
        "schema": OUTPUT_SCHEMA,
        "barcode_src": "book-1",
        "selection_row_sha256": "1" * 64,
        "quality_agreement_record_sha256": "2" * 64,
        "benchmark_decontamination_record_sha256": "3" * 64,
        "semantic_domains": ["mathematics_geometry"],
        "curriculum_metadata_json": json.dumps(
            curriculum, sort_keys=True, separators=(",", ":")
        ),
        "curriculum_metadata_sha256": curriculum["metadata_sha256"],
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "cross_source_subdocument_transform_sha256": "4" * 64,
        "corpus_split": "train",
        "text": text,
        "training_ready": False,
    }
    selection = {
        "schema": SELECTION_ROW_SCHEMA,
        "barcode_src": "book-1",
        "rights_code": "pdus",
        "row_sha256": "1" * 64,
        "training_ready": False,
    }
    return row, selection


def test_book_envelope_binds_rights_and_semantic_domain() -> None:
    row, selection = _fixture()
    envelope = _envelope(row, selection, "5" * 64)
    source = envelope["document"]["source"]
    assert source["domain"] == "math"
    assert source["license"] == (
        "Public Domain in the United States (HathiTrust rights code: pdus)"
    )
    assert envelope["semantic_quality_floor_milli"] == 8_000
    assert envelope["semantic_difficulty_mean_milli"] == 3_000
    assert envelope["tokenization_ready"] is True


def test_book_envelope_rejects_curriculum_hash_mutation() -> None:
    row, selection = _fixture()
    row["curriculum_metadata_sha256"] = "f" * 64
    with pytest.raises(
        InstitutionalBooksTransientTokenizerStreamError,
        match="curriculum metadata differs",
    ):
        _envelope(row, selection, "5" * 64)


def test_generic_sampler_accepts_rights_bound_book_stream(tmp_path: Path) -> None:
    row, selection = _fixture()
    envelope = _envelope(row, selection, "5" * 64)
    payload = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    source_receipt = {
        "schema": GENERIC_SOURCE_RECEIPT_SCHEMA,
        "status": GENERIC_SOURCE_STATUS,
        "counts": {"documents": 1},
        "ordered_jsonl_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "source_text_persisted_by_compiler": False,
        "training_ready": False,
    }
    source_receipt["receipt_sha256"] = canonical_sha256(source_receipt)
    receipt = tmp_path / "source.json"
    receipt.write_text(json.dumps(source_receipt))
    output = tmp_path / "sample"
    result = build_sample(
        io.StringIO(payload),
        receipt,
        output,
        maximum_utf8_bytes=1_000_000,
    )
    assert result["sample"]["documents"] == 1
    sampled = json.loads((output / SAMPLE_NAME).read_text())
    assert sampled["source"]["license"].endswith("rights code: pdus)")
