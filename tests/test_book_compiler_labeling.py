import copy
import hashlib
import json
from pathlib import Path

import pytest

from sai.data.book_compiler_labeling import (
    CANDIDATE_SCHEMA,
    RUBRIC_SHA256,
    BookCompilerError,
    build_messages,
    normalize_book_candidate,
    normalize_model_judgment,
)
from sai.data.nous_book_compiler_worker import (
    RECEIPT_SCHEMA,
    _load_book_jsonl,
    execute_one,
)
from sai.data.token_stream import canonical_sha256


def _candidate(language: str = "english") -> dict:
    text = (
        "Before studying atmospheric thermodynamics, the reader should understand "
        "matter, temperature, pressure, evaporation, condensation, and latent heat. "
        "These foundations explain how energy and water move through the atmosphere. "
    ) * 4
    row = {
        "schema": CANDIDATE_SCHEMA,
        "text_excerpt": text,
        "source": {
            "dataset": "institutional/institutional-books-hl-enriched-text",
            "revision": "9" * 40,
            "barcode_src": "32044000000018",
            "metadata_row_sha256": "1" * 64,
            "dataset_terms_sha256": "2" * 64,
            "source_archive": "harvard_library_google_books",
            "text_field": "processed_middlematter_gen",
        },
        "bibliographic": {
            "title_src": "Atmospheric Thermodynamics",
            "author_src": "Example, Ada",
            "date1_src": "1917",
            "date2_src": None,
            "language_src": "eng" if language == "english" else "rus",
            "language_gen": "eng" if language == "english" else "rus",
            "topic_or_subject_src": "Meteorology",
            "topic_or_subject_gen": "SCIENCE",
            "genre_or_form_src": "Textbook",
            "general_note_src": None,
            "likely_duplicates_barcodes_gen": [],
            "identifiers_src": {"lccn": [], "isbn": [], "ocolc": ["123"]},
            "rights_evidence": {
                "provider": "hathitrust",
                "status_code": "pd",
                "reason_code": "bib",
                "last_checked": "2025-05-12",
                "source_url": "https://hdl.handle.net/2027/example",
            },
        },
        "measurements": {
            "page_count_src": 314,
            "token_count_o200k_base_gen": 201834,
            "ocr_score_src": 94,
            "ocr_score_gen": 98,
        },
        "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "provenance_sha256": "3" * 64,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def _judgment(language: str = "english", genre: str = "textbook_manual") -> dict:
    quote = "matter, temperature, pressure, evaporation, condensation, and latent heat"
    english = language == "english"
    return {
        "verdict": "retain",
        "work_id_candidate": "atmospheric-thermodynamics-example",
        "edition_id_candidate": "atmospheric-thermodynamics-example-1917",
        "author_normalized_candidate": "Ada Example",
        "author_death_candidate": None,
        "publication_date_normalized_candidate": "1917",
        "original_language": language,
        "current_language": language,
        "translator_candidate": None,
        "translation_date_candidate": None,
        "domains": ["earth_environment", "physics_astronomy"],
        "subdomains": ["atmospheric thermodynamics"],
        "genre": genre,
        "style": "exposition" if genre == "textbook_manual" else "narrative",
        "quality": {
            "overall_quality": 4,
            "ocr_quality": 4,
            "literary_value": 1 if genre == "textbook_manual" else 4,
            "knowledge_density": 4,
            "factual_reliability": 4,
            "historical_value": 3,
        },
        "complexity": {
            "linguistic_complexity": 1,
            "conceptual_complexity": 3,
            "reasoning_complexity": 2,
        },
        "curriculum_band": "advanced",
        "prerequisites": ["temperature", "pressure", "latent heat"],
        "concepts": ["atmospheric thermodynamics"],
        "concept_edges": [
            {
                "prerequisite": "latent heat",
                "dependent": "atmospheric thermodynamics",
                "relation": "requires",
                "confidence_ppm": 950_000,
                "evidence_quote": quote,
            }
        ],
        "period": ["early twentieth century"],
        "culture_geography": ["global science"],
        "translation_type": ("none_english" if english else "create_technical_english"),
        "translation_confidence_ppm": 1_000_000 if english else 900_000,
        "human_translation_search_required": False,
        "preserve_original_language_anchor": not english,
        "recommended_representations": (
            ["preserved_english_source", "prerequisite_map"]
            if english
            else ["clean_ocr_english", "prerequisite_map"]
        ),
        "duplicate_work_ids": [],
        "risks": {
            "ocr_damage": False,
            "outdated_or_harmful_claims": False,
            "factual_unreliability": False,
            "bibliographic_ambiguity": False,
            "duplicate_or_near_duplicate_edition": False,
            "translation_loss": not english,
            "cultural_flattening": False,
            "generic_model_voice": False,
            "rights_evidence_incomplete": False,
        },
        "confidence_ppm": 940_000,
        "evidence_quotes": [quote],
        "rationale": "The book teaches a prerequisite-rich scientific progression.",
    }


def test_book_candidate_and_prompt_bind_archive_fields() -> None:
    candidate = normalize_book_candidate(_candidate())
    messages = build_messages(candidate)
    assert "Difficulty is not one scalar" in messages[0]["content"]
    envelope = json.loads(messages[1]["content"])
    assert envelope["source"]["barcode_src"] == "32044000000018"
    assert envelope["rubric_sha256"] == RUBRIC_SHA256
    assert set(envelope["output_schema"]["complexity"]) == {
        "linguistic_complexity",
        "conceptual_complexity",
        "reasoning_complexity",
    }


def test_book_judgment_preserves_rights_and_explicit_graph() -> None:
    candidate = _candidate()
    result = normalize_model_judgment(_judgment(), candidate)
    assert result["source_id"] == "32044000000018"
    assert result["rights_are_model_inferred"] is False
    assert result["rights_evidence"]["status_code"] == "pd"
    assert result["complexity"]["conceptual_complexity"] == 3
    assert result["concept_edges"][0]["prerequisite"] == "latent heat"
    assert result["raw_archive_source_is_training_ready"] is False


def test_non_english_technical_book_is_routed_to_english() -> None:
    result = normalize_model_judgment(_judgment("russian"), _candidate("russian"))
    assert result["translation_type"] == "create_technical_english"
    assert result["translation_is_synthetic"] is True
    assert result["preserve_original_language_anchor"] is True


def test_non_english_literature_requires_human_search_or_dual_translation() -> None:
    candidate = _candidate("russian")
    raw = _judgment("russian", "literature")
    raw["translation_type"] = "create_technical_english"
    raw["recommended_representations"] = ["clean_ocr_english"]
    with pytest.raises(BookCompilerError, match="literary translation policy"):
        normalize_model_judgment(raw, candidate)

    raw["translation_type"] = "create_literal_and_literary_english"
    raw["human_translation_search_required"] = True
    raw["recommended_representations"] = [
        "synthetic_literal_english_translation",
        "synthetic_literary_english_translation",
    ]
    result = normalize_model_judgment(raw, candidate)
    assert result["translation_is_synthetic"] is True


def test_book_compiler_rejects_fabricated_graph_evidence_and_rights_tamper() -> None:
    candidate = _candidate()
    raw = _judgment()
    raw["concept_edges"][0]["evidence_quote"] = "not in the excerpt"
    with pytest.raises(BookCompilerError, match="edge evidence"):
        normalize_model_judgment(raw, candidate)
    tampered = copy.deepcopy(candidate)
    tampered["bibliographic"]["rights_evidence"]["status_code"] = "ic"
    with pytest.raises(BookCompilerError, match="identity"):
        normalize_book_candidate(tampered)


def test_book_worker_loader_accepts_only_book_candidates(tmp_path: Path) -> None:
    path = tmp_path / "books.jsonl"
    path.write_text(json.dumps(_candidate()) + "\n")
    assert _load_book_jsonl(path)[0]["source"]["barcode_src"] == "32044000000018"
    path.write_text(json.dumps(_candidate()) + "\n" + json.dumps(_candidate()) + "\n")
    with pytest.raises(Exception, match="duplicated"):
        _load_book_jsonl(path)


def test_book_worker_emits_source_bound_non_training_receipt() -> None:
    candidate = _candidate()
    judgment = _judgment()

    def request_function(**kwargs):
        assert kwargs["body"]["model"] == "stealth/ox-alpha"
        assert kwargs["body"]["max_tokens"] == 4000
        return (
            {
                "id": "response-1",
                "model": "stealth/ox-alpha",
                "provider": "nous",
                "created": 1,
                "choices": [
                    {
                        "message": {"content": json.dumps(judgment)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                    "total_tokens": 300,
                },
            },
            200,
        )

    receipt = execute_one(
        candidate,
        model="stealth/ox-alpha",
        base_url="http://127.0.0.1:8645/v1",
        api_key="loopback-only",
        timeout_seconds=1,
        maximum_attempts=1,
        request_function=request_function,
    )
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert (
        receipt["candidate_identity_sha256"] == candidate["candidate_identity_sha256"]
    )
    assert receipt["training_ready"] is False
    assert receipt["judgment"]["raw_archive_source_is_training_ready"] is False
