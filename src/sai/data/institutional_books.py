"""Build exact Institutional Books candidates without admitting raw books."""

from __future__ import annotations

import hashlib
from typing import Any

from sai.data.book_compiler_labeling import CANDIDATE_SCHEMA, normalize_book_candidate
from sai.data.token_stream import canonical_sha256

BOOKS_REPOSITORY = "institutional/institutional-books-hl"
BOOKS_REVISION = "1f12e87e317077474679899a8f78feaeb8a995ff"
BOOKS_CARD_SHA256 = "7dafaade54d9f8ec06730016d20cad0b97b9fa894911aa40c1d1279d356044c2"
ENRICHED_REPOSITORY = "institutional/institutional-books-hl-enriched-text"
ENRICHED_REVISION = "92fcdf938eb87edfe0fbf09d4f692fa3d8bc9bcd"
ENRICHED_CARD_SHA256 = (
    "8b8d48599d470ad4c6045b901a644b915cc00ddd0dac05c48954898bc8f24a7f"
)
METADATA_REPOSITORY = "institutional/institutional-books-hl-metadata"
METADATA_REVISION = "e0c0b860482e49cced246c5cd7c0d22ac8a86040"
METADATA_CARD_SHA256 = (
    "96ed6214515223d12695b558321e100dd372765fb75411a6ce51a35c38da05a2"
)
METADATA_PARQUET_SHA256 = (
    "55861c3e735c71bdcd78ee3d4eeec59a0a547aa37a4f62e9df712869d44c8dfb"
)
METADATA_PARQUET_BYTES = 306_251_508
TEXT_FIELD = "processed_middlematter_gen"
MAX_EXCERPT_BYTES = 32_768


class InstitutionalBooksError(RuntimeError):
    """Institutional Books metadata, text, or source pin differs."""


def _nullable(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _bounded_metadata_strings(
    value: Any, *, maximum: int, label: str
) -> list[str]:
    """Deduplicate archive metadata and retain a deterministic bounded prefix."""

    if value is None:
        return []
    if (
        not isinstance(value, list)
        or isinstance(maximum, bool)
        or maximum < 1
        or any(
            not isinstance(item, str) or not item or len(item) > 512
            for item in value
        )
    ):
        raise InstitutionalBooksError(f"{label} differs")
    # The unmodified metadata row remains bound by metadata_row_sha256.  The
    # prompt-facing view removes repeated identifiers and bounds pathological
    # edition clusters so one archive record cannot break or dominate a book
    # review request.
    return list(dict.fromkeys(value))[:maximum]


def _utf8_prefix(value: str, maximum: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum:
        return value
    return encoded[:maximum].decode("utf-8", errors="ignore")


def representative_excerpt(text: str, maximum_bytes: int = MAX_EXCERPT_BYTES) -> str:
    """Sample beginning, middle, and end without pretending it is the full book."""

    if (
        not isinstance(text, str)
        or len(text.encode("utf-8")) < 200
        or isinstance(maximum_bytes, bool)
        or not 12_288 <= maximum_bytes <= 262_144
    ):
        raise InstitutionalBooksError("book text or excerpt geometry differs")
    if len(text.encode("utf-8")) <= maximum_bytes:
        return text
    marker = "\n\n[SAI REPRESENTATIVE EXCERPT BOUNDARY]\n\n"
    marker_bytes = len(marker.encode()) * 2
    segment_bytes = (maximum_bytes - marker_bytes) // 3
    span = max(1, len(text) // 3)
    start = _utf8_prefix(text, segment_bytes)
    middle_start = max(0, len(text) // 2 - span // 2)
    middle = _utf8_prefix(text[middle_start:], segment_bytes)
    end_candidate = text[max(0, len(text) - span) :]
    encoded_end = end_candidate.encode("utf-8")
    end = encoded_end[-segment_bytes:].decode("utf-8", errors="ignore")
    excerpt = marker.join((start, middle, end))
    if not 200 <= len(excerpt.encode("utf-8")) <= maximum_bytes:
        raise InstitutionalBooksError("representative excerpt size differs")
    return excerpt


def build_book_candidate(
    metadata_row: dict[str, Any], enriched_row: dict[str, Any]
) -> dict[str, Any]:
    """Join exact metadata and enriched text into a non-training candidate."""

    if not isinstance(metadata_row, dict) or not isinstance(enriched_row, dict):
        raise InstitutionalBooksError("Institutional Books row differs")
    required_metadata = {
        "barcode_src",
        "title_src",
        "author_src",
        "date1_src",
        "date2_src",
        "page_count_src",
        "token_count_o200k_base_gen",
        "language_src",
        "language_gen",
        "topic_or_subject_src",
        "topic_or_subject_gen",
        "genre_or_form_src",
        "general_note_src",
        "ocr_score_src",
        "ocr_score_gen",
        "likely_duplicates_barcodes_gen",
        "identifiers_src",
        "hathitrust_data_ext",
    }
    if not required_metadata.issubset(metadata_row):
        raise InstitutionalBooksError("Institutional Books metadata fields differ")
    barcode = metadata_row["barcode_src"]
    if (
        not isinstance(barcode, str)
        or not barcode
        or enriched_row.get("barcode_src") != barcode
        or not isinstance(enriched_row.get(TEXT_FIELD), str)
    ):
        raise InstitutionalBooksError("Institutional Books join differs")
    duplicate_barcodes = _bounded_metadata_strings(
        metadata_row["likely_duplicates_barcodes_gen"],
        maximum=256,
        label="duplicate barcodes",
    )
    raw_identifiers = metadata_row["identifiers_src"] or {
        "lccn": [],
        "isbn": [],
        "ocolc": [],
    }
    if not isinstance(raw_identifiers, dict) or set(raw_identifiers) != {
        "lccn",
        "isbn",
        "ocolc",
    }:
        raise InstitutionalBooksError("identifiers differ")
    identifiers = {
        field: _bounded_metadata_strings(
            raw_identifiers[field], maximum=64, label=field
        )
        for field in ("lccn", "isbn", "ocolc")
    }
    rights = metadata_row["hathitrust_data_ext"] or {}
    metadata_sha256 = canonical_sha256(metadata_row)
    enriched_sha256 = canonical_sha256(enriched_row)
    excerpt = representative_excerpt(enriched_row[TEXT_FIELD])
    provenance = {
        "metadata_repository": METADATA_REPOSITORY,
        "metadata_revision": METADATA_REVISION,
        "metadata_parquet_sha256": METADATA_PARQUET_SHA256,
        "metadata_row_sha256": metadata_sha256,
        "enriched_repository": ENRICHED_REPOSITORY,
        "enriched_revision": ENRICHED_REVISION,
        "enriched_row_sha256": enriched_sha256,
        "join_key": barcode,
        "excerpt_policy": "beginning_middle_end_utf8_32768_v2",
        "bibliographic_normalization_policy": (
            "stable_first_unique_identifiers_64_duplicate_barcodes_256_v1"
        ),
    }
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "text_excerpt": excerpt,
        "source": {
            "dataset": ENRICHED_REPOSITORY,
            "revision": ENRICHED_REVISION,
            "barcode_src": barcode,
            "metadata_row_sha256": metadata_sha256,
            "dataset_terms_sha256": ENRICHED_CARD_SHA256,
            "source_archive": "harvard_library_google_books",
            "text_field": TEXT_FIELD,
        },
        "bibliographic": {
            "title_src": _nullable(metadata_row["title_src"]),
            "author_src": _nullable(metadata_row["author_src"]),
            "date1_src": _nullable(metadata_row["date1_src"]),
            "date2_src": _nullable(metadata_row["date2_src"]),
            "language_src": _nullable(metadata_row["language_src"]),
            "language_gen": _nullable(metadata_row["language_gen"]),
            "topic_or_subject_src": _nullable(metadata_row["topic_or_subject_src"]),
            "topic_or_subject_gen": _nullable(metadata_row["topic_or_subject_gen"]),
            "genre_or_form_src": _nullable(metadata_row["genre_or_form_src"]),
            "general_note_src": _nullable(metadata_row["general_note_src"]),
            "likely_duplicates_barcodes_gen": duplicate_barcodes,
            "identifiers_src": identifiers,
            "rights_evidence": {
                "provider": "hathitrust" if rights else None,
                "status_code": _nullable(rights.get("rights_code")),
                "reason_code": _nullable(rights.get("reason_code")),
                "last_checked": _nullable(rights.get("last_check")),
                "source_url": _nullable(rights.get("url")),
            },
        },
        "measurements": {
            "page_count_src": metadata_row["page_count_src"],
            "token_count_o200k_base_gen": metadata_row["token_count_o200k_base_gen"],
            "ocr_score_src": metadata_row["ocr_score_src"],
            "ocr_score_gen": metadata_row["ocr_score_gen"],
        },
        "source_content_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        "provenance_sha256": canonical_sha256(provenance),
    }
    candidate["candidate_identity_sha256"] = canonical_sha256(candidate)
    return normalize_book_candidate(candidate)
