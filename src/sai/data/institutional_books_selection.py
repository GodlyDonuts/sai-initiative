"""Select a coverage-first Institutional Books metadata review population."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.institutional_books import (
    METADATA_PARQUET_BYTES,
    METADATA_PARQUET_SHA256,
    METADATA_REPOSITORY,
    METADATA_REVISION,
)
from sai.data.token_stream import canonical_sha256, sha256_file

ROW_SCHEMA = "sai-institutional-books-metadata-review-row-v1"
RECEIPT_SCHEMA = "sai-institutional-books-metadata-review-selection-v1"
ALLOWED_RIGHTS_CODES = {"pd", "pdus", "cc-zero"}
MINIMUM_OCR_SCORE = 80
MINIMUM_TOKENS = 2_000
MAXIMUM_TOKENS = 2_000_000
EXPECTED_COLUMNS = {
    "barcode_src",
    "title_src",
    "author_src",
    "date1_src",
    "date2_src",
    "date_types_src",
    "page_count_src",
    "token_count_o200k_base_gen",
    "language_src",
    "language_gen",
    "language_distribution_gen",
    "topic_or_subject_src",
    "topic_or_subject_gen",
    "topic_or_subject_score_gen",
    "genre_or_form_src",
    "general_note_src",
    "ocr_score_src",
    "ocr_score_gen",
    "likely_duplicates_barcodes_gen",
    "text_analysis_gen",
    "identifiers_src",
    "hathitrust_data_ext",
}


class InstitutionalBooksSelectionError(RuntimeError):
    """The metadata source, duplicate graph, or selection differs."""


def _score(row: dict[str, Any]) -> tuple[Any, ...]:
    ocr = row.get("ocr_score_gen")
    topic = row.get("topic_or_subject_score_gen")
    analysis = row.get("text_analysis_gen") or {}
    generated = analysis.get("text_by_page_gen") or {}
    tokenizability = generated.get("tokenizability_score")
    completeness = sum(
        bool(row.get(field))
        for field in (
            "title_src",
            "author_src",
            "date1_src",
            "topic_or_subject_src",
            "genre_or_form_src",
            "general_note_src",
        )
    )
    return (
        -(ocr if isinstance(ocr, int) else -1),
        -(topic if isinstance(topic, (int, float)) else -1.0),
        -(tokenizability if isinstance(tokenizability, (int, float)) else -1.0),
        -completeness,
        row["barcode_src"],
    )


def _representatives(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    barcodes = [row.get("barcode_src") for row in rows]
    if (
        not rows
        or any(not isinstance(item, str) or not item for item in barcodes)
        or len(barcodes) != len(set(barcodes))
    ):
        raise InstitutionalBooksSelectionError("metadata barcode population differs")
    index = {barcode: position for position, barcode in enumerate(barcodes)}
    parent = list(range(len(rows)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left = find(left)
        right = find(right)
        if left != right:
            if left > right:
                left, right = right, left
            parent[right] = left

    links = 0
    for position, row in enumerate(rows):
        duplicates = row.get("likely_duplicates_barcodes_gen") or []
        if not isinstance(duplicates, list):
            raise InstitutionalBooksSelectionError("duplicate barcode evidence differs")
        for duplicate in duplicates:
            other = index.get(duplicate)
            if other is not None and other != position:
                union(position, other)
                links += 1
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for position, row in enumerate(rows):
        groups[find(position)].append(row)
    representatives = [min(group, key=_score) for group in groups.values()]
    return representatives, links


def _eligible(row: dict[str, Any]) -> bool:
    rights = row.get("hathitrust_data_ext") or {}
    tokens = row.get("token_count_o200k_base_gen")
    return bool(
        isinstance(row.get("language_gen"), str)
        and row["language_gen"]
        and isinstance(row.get("topic_or_subject_gen"), str)
        and row["topic_or_subject_gen"]
        and isinstance(row.get("ocr_score_gen"), int)
        and row["ocr_score_gen"] >= MINIMUM_OCR_SCORE
        and isinstance(tokens, int)
        and MINIMUM_TOKENS <= tokens <= MAXIMUM_TOKENS
        and isinstance(rights, dict)
        and rights.get("rights_code") in ALLOWED_RIGHTS_CODES
    )


def select_metadata_rows(
    rows: list[dict[str, Any]], target_count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one duplicate representative per coverage-first review slot."""

    if (
        isinstance(target_count, bool)
        or not isinstance(target_count, int)
        or target_count <= 0
    ):
        raise InstitutionalBooksSelectionError("selection target differs")
    representatives, duplicate_links = _representatives(rows)
    eligible = [row for row in representatives if _eligible(row)]
    if target_count > len(eligible):
        raise InstitutionalBooksSelectionError("selection target exceeds population")
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        cells[(row["language_gen"], row["topic_or_subject_gen"])].append(row)
    for values in cells.values():
        values.sort(key=_score)
    cell_order = sorted(
        cells,
        key=lambda cell: hashlib.sha256(
            f"sai-institutional-books-review-v1:{cell[0]}:{cell[1]}".encode()
        ).digest(),
    )
    selected = []
    depth = 0
    while len(selected) < target_count:
        progressed = False
        for cell in cell_order:
            if depth < len(cells[cell]):
                selected.append(cells[cell][depth])
                progressed = True
                if len(selected) == target_count:
                    break
        if not progressed:
            raise InstitutionalBooksSelectionError("coverage round robin exhausted")
        depth += 1
    records = []
    for rank, row in enumerate(selected, start=1):
        metadata_row_sha256 = canonical_sha256(row)
        record = {
            "schema": ROW_SCHEMA,
            "rank": rank,
            "barcode_src": row["barcode_src"],
            "metadata_row_sha256": metadata_row_sha256,
            "language_src": row.get("language_src"),
            "language_gen": row["language_gen"],
            "english_translation_required": row["language_gen"] != "eng",
            "topic_or_subject_gen": row["topic_or_subject_gen"],
            "topic_or_subject_score_gen": row.get("topic_or_subject_score_gen"),
            "ocr_score_gen": row["ocr_score_gen"],
            "token_count_o200k_base_gen": row["token_count_o200k_base_gen"],
            "title_src": row.get("title_src"),
            "author_src": row.get("author_src"),
            "date1_src": row.get("date1_src"),
            "genre_or_form_src": row.get("genre_or_form_src"),
            "duplicate_barcodes": row.get("likely_duplicates_barcodes_gen") or [],
            "rights_evidence": row.get("hathitrust_data_ext"),
            "review_queue_only": True,
            "training_ready": False,
        }
        record["row_identity_sha256"] = canonical_sha256(record)
        records.append(record)
    languages = Counter(row["language_gen"] for row in selected)
    topics = Counter(row["topic_or_subject_gen"] for row in selected)
    statistics = {
        "source_rows": len(rows),
        "duplicate_graph_links": duplicate_links,
        "duplicate_components": len(representatives),
        "eligible_representatives": len(eligible),
        "coverage_cells": len(cells),
        "selected_rows": len(records),
        "selected_english_rows": languages.get("eng", 0),
        "selected_non_english_rows": len(records) - languages.get("eng", 0),
        "selected_languages": dict(sorted(languages.items())),
        "selected_topics": dict(sorted(topics.items())),
    }
    return records, statistics


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise InstitutionalBooksSelectionError("selection output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial.{uuid.uuid4().hex}"
    try:
        with temporary.open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != METADATA_PARQUET_BYTES
        or sha256_file(path) != METADATA_PARQUET_SHA256
    ):
        raise InstitutionalBooksSelectionError("metadata parquet identity differs")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksSelectionError("pyarrow is required") from error
    parquet = pq.ParquetFile(path)
    if set(parquet.schema_arrow.names) != EXPECTED_COLUMNS:
        raise InstitutionalBooksSelectionError("metadata parquet columns differ")
    return parquet.read().to_pylist()


def build_selection(
    source: Path, output: Path, receipt_path: Path, *, target_count: int
) -> dict[str, Any]:
    """Create a metadata-only review queue and exact receipt."""

    if receipt_path.exists() or receipt_path.is_symlink():
        raise InstitutionalBooksSelectionError("selection receipt already exists")
    records, statistics = select_metadata_rows(_read_parquet(source), target_count)
    _atomic_write_jsonl(output, records)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "source": {
            "repository": METADATA_REPOSITORY,
            "revision": METADATA_REVISION,
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": METADATA_PARQUET_SHA256,
        },
        "selection_policy": {
            "purpose": "coverage_first_review_queue_not_training_mixture",
            "minimum_ocr_score": MINIMUM_OCR_SCORE,
            "minimum_tokens": MINIMUM_TOKENS,
            "maximum_tokens": MAXIMUM_TOKENS,
            "allowed_rights_codes": sorted(ALLOWED_RIGHTS_CODES),
            "duplicate_policy": (
                "one_best_metadata_representative_per_connected_component"
            ),
            "coverage_policy": "round_robin_detected_language_by_generated_subject",
        },
        "statistics": statistics,
        "output": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "rows": len(records),
            "row_identities_sha256": canonical_sha256(
                [row["row_identity_sha256"] for row in records]
            ),
        },
        "text_downloaded": False,
        "hermes_calls_completed": 0,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_write_jsonl(receipt_path, [receipt])
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=10_000)
    args = parser.parse_args()
    receipt = build_selection(
        args.source, args.output, args.receipt, target_count=args.target_count
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
