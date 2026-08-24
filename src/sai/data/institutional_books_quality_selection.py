"""Freeze the strict English OCR>=95 Institutional Books materialization set."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books import (
    METADATA_PARQUET_SHA256,
    METADATA_REPOSITORY,
    METADATA_REVISION,
)
from sai.data.institutional_books_quality_census import (
    SCHEMA as CENSUS_SCHEMA,
)
from sai.data.institutional_books_quality_census import _base_eligible
from sai.data.institutional_books_selection import _read_parquet, _representatives
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-institutional-books-strict-english-selection-v1"
ROW_SCHEMA = "sai-institutional-books-strict-english-selection-row-v1"
MINIMUM_OCR_SCORE = 95


class InstitutionalBooksQualitySelectionError(RuntimeError):
    """The strict book selection or its census binding differs."""


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise InstitutionalBooksQualitySelectionError("selection output exists")
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


def select_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select duplicate representatives in the strict English quality tier."""

    representatives, _ = _representatives(rows)
    selected = [
        row
        for row in representatives
        if _base_eligible(row)
        and row["language_gen"] == "eng"
        and row["ocr_score_gen"] >= MINIMUM_OCR_SCORE
    ]
    selected.sort(key=lambda row: row["barcode_src"])
    output = []
    for row in selected:
        rights = row["hathitrust_data_ext"]
        record = {
            "schema": ROW_SCHEMA,
            "barcode_src": row["barcode_src"],
            "metadata_row_sha256": canonical_sha256(row),
            "token_count_o200k_base_gen": row[
                "token_count_o200k_base_gen"
            ],
            "ocr_score_gen": row["ocr_score_gen"],
            "language_gen": "eng",
            "topic_or_subject_gen": row["topic_or_subject_gen"],
            "genre_or_form_src": row.get("genre_or_form_src"),
            "rights_code": rights["rights_code"],
            "rights_reason_code": rights.get("reason_code"),
            "rights_last_check": rights.get("last_check"),
            "source_text_persisted": False,
            "training_ready": False,
        }
        record["row_sha256"] = canonical_sha256(record)
        output.append(record)
    return output


def _load_census(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise InstitutionalBooksQualitySelectionError("quality census is unsafe")
    try:
        census = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise InstitutionalBooksQualitySelectionError(
            "quality census is invalid"
        ) from error
    unsigned = {key: value for key, value in census.items() if key != "receipt_sha256"}
    if (
        not isinstance(census, dict)
        or census.get("schema") != CENSUS_SCHEMA
        or census.get("receipt_sha256") != canonical_sha256(unsigned)
        or census.get("training_ready") is not False
    ):
        raise InstitutionalBooksQualitySelectionError("quality census differs")
    return census


def build_selection(
    source: Path, census_path: Path, output_root: Path
) -> dict[str, Any]:
    """Write the exact strict-tier barcode set and a source-safe receipt."""

    if output_root.exists() or output_root.is_symlink():
        raise InstitutionalBooksQualitySelectionError("selection root exists")
    census = _load_census(census_path)
    rows = select_rows(_read_parquet(source))
    expected_rows = census.get("counts", {}).get("english_ocr_95_rows")
    expected_tokens = census.get("counts", {}).get("english_ocr_95_tokens")
    selected_tokens = sum(row["token_count_o200k_base_gen"] for row in rows)
    if len(rows) != expected_rows or selected_tokens != expected_tokens:
        raise InstitutionalBooksQualitySelectionError(
            "selection-to-census accounting differs"
        )
    output_root.mkdir(parents=True)
    try:
        manifest = output_root / "selection.jsonl"
        _atomic_jsonl(manifest, rows)
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_strict_english_book_selection",
            "source": {
                "repository": METADATA_REPOSITORY,
                "revision": METADATA_REVISION,
                "sha256": METADATA_PARQUET_SHA256,
            },
            "quality_census": {
                "receipt_sha256": census["receipt_sha256"],
                "file_sha256": sha256_file(census_path),
            },
            "policy": {
                "language_gen": "eng",
                "minimum_ocr_score_gen": MINIMUM_OCR_SCORE,
                "duplicate_policy": (
                    "one_best_metadata_representative_per_connected_component"
                ),
                "rights_and_token_policy": "quality_census_v1",
            },
            "selection": {
                "path": manifest.name,
                "rows": len(rows),
                "tokens": selected_tokens,
                "bytes": manifest.stat().st_size,
                "sha256": sha256_file(manifest),
                "ordered_rows_sha256": canonical_sha256(
                    [row["row_sha256"] for row in rows]
                ),
            },
            "selection_contains_source_text": False,
            "selection_is_training_admission": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        for child in output_root.iterdir():
            child.unlink()
        output_root.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quality-census", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_selection(args.source, args.quality_census, args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
