"""Build the first local Institutional Books Hermes candidate population."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.institutional_books import (
    ENRICHED_REPOSITORY,
    ENRICHED_REVISION,
    METADATA_PARQUET_BYTES,
    METADATA_PARQUET_SHA256,
    build_book_candidate,
)
from sai.data.institutional_books_selection import (
    ALLOWED_RIGHTS_CODES,
    MAXIMUM_TOKENS,
    MINIMUM_OCR_SCORE,
    MINIMUM_TOKENS,
)
from sai.data.institutional_books_selection import (
    EXPECTED_COLUMNS as METADATA_COLUMNS,
)
from sai.data.token_stream import canonical_sha256, sha256_file

PILOT_SHARD_REPOSITORY_PATH = "train/data/train-00000-of-04916.parquet"
PILOT_SHARD_SHA256 = "8c86db250ede35dfb5039760dca4f5699f9c953041e9faa75151c418aa1c778b"
PILOT_SHARD_BYTES = 127_727_860
RECEIPT_SCHEMA = "sai-institutional-books-pilot-candidate-population-v1"
ENRICHED_COLUMNS = {
    "barcode_src",
    "primary_language_gen",
    "language_distribution_gen",
    "token_count_gen",
    "char_count_gen",
    "word_count_gen",
    "sentence_count_gen",
    "paragraph_count_gen",
    "section_count_gen",
    "bigram_count_gen",
    "bigram_count_unique_gen",
    "trigram_count_gen",
    "trigram_count_unique_gen",
    "tokenizability_ratio_gen",
    "bpb_min_gen",
    "bpb_max_gen",
    "bpb_median_gen",
    "bpb_avg_gen",
    "bpb_p10_gen",
    "bpb_p30_gen",
    "bpb_p70_gen",
    "bpb_p90_gen",
    "processed_middlematter_gen",
    "frontmatter_gen",
    "middlematter_gen",
    "backmatter_gen",
}


class InstitutionalBooksPilotError(RuntimeError):
    """The pilot shard, metadata join, or candidate population differs."""


def _eligible(metadata: dict[str, Any], enriched: dict[str, Any]) -> str | None:
    rights = metadata.get("hathitrust_data_ext") or {}
    tokens = metadata.get("token_count_o200k_base_gen")
    text = enriched.get("processed_middlematter_gen")
    if rights.get("rights_code") not in ALLOWED_RIGHTS_CODES:
        return "rights_code"
    if not isinstance(metadata.get("ocr_score_gen"), int) or (
        metadata["ocr_score_gen"] < MINIMUM_OCR_SCORE
    ):
        return "ocr_below_minimum"
    if (
        not isinstance(tokens, int)
        or tokens < MINIMUM_TOKENS
        or tokens > MAXIMUM_TOKENS
    ):
        return "token_count_out_of_range"
    if not isinstance(text, str) or len(text.encode("utf-8")) < 200:
        return "processed_middlematter_missing_or_short"
    return None


def build_candidate_population(
    metadata_rows: list[dict[str, Any]], enriched_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join, filter, and compile one shard into local non-training candidates."""

    if not metadata_rows or not enriched_rows:
        raise InstitutionalBooksPilotError("pilot row population is empty")
    metadata = {}
    for row in metadata_rows:
        barcode = row.get("barcode_src")
        if not isinstance(barcode, str) or not barcode or barcode in metadata:
            raise InstitutionalBooksPilotError("pilot metadata barcode differs")
        metadata[barcode] = row
    candidates = []
    rejected: Counter[str] = Counter()
    seen = set()
    languages: Counter[str] = Counter()
    topics: Counter[str] = Counter()
    for row in enriched_rows:
        barcode = row.get("barcode_src")
        if not isinstance(barcode, str) or not barcode or barcode in seen:
            raise InstitutionalBooksPilotError("pilot enriched barcode differs")
        seen.add(barcode)
        metadata_row = metadata.get(barcode)
        if metadata_row is None:
            raise InstitutionalBooksPilotError("pilot metadata join is incomplete")
        reason = _eligible(metadata_row, row)
        if reason is not None:
            rejected[reason] += 1
            continue
        candidate = build_book_candidate(metadata_row, row)
        candidates.append(candidate)
        languages[metadata_row["language_gen"]] += 1
        topics[metadata_row["topic_or_subject_gen"]] += 1
    if not candidates:
        raise InstitutionalBooksPilotError("pilot admitted no book candidates")
    statistics = {
        "enriched_rows": len(enriched_rows),
        "matched_metadata_rows": len(seen),
        "candidate_rows": len(candidates),
        "rejected_rows": sum(rejected.values()),
        "rejection_reasons": dict(sorted(rejected.items())),
        "candidate_english_rows": languages.get("eng", 0),
        "candidate_non_english_rows": len(candidates) - languages.get("eng", 0),
        "candidate_languages": dict(sorted(languages.items())),
        "candidate_topics": dict(sorted(topics.items())),
    }
    return candidates, statistics


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise InstitutionalBooksPilotError("pilot output already exists")
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


def _read_inputs(
    metadata_path: Path, enriched_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        not metadata_path.is_file()
        or metadata_path.is_symlink()
        or metadata_path.stat().st_nlink != 1
        or metadata_path.stat().st_size != METADATA_PARQUET_BYTES
        or sha256_file(metadata_path) != METADATA_PARQUET_SHA256
    ):
        raise InstitutionalBooksPilotError("pilot metadata identity differs")
    if (
        not enriched_path.is_file()
        or enriched_path.is_symlink()
        or enriched_path.stat().st_nlink != 1
        or enriched_path.stat().st_size != PILOT_SHARD_BYTES
        or sha256_file(enriched_path) != PILOT_SHARD_SHA256
    ):
        raise InstitutionalBooksPilotError("pilot enriched shard identity differs")
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstitutionalBooksPilotError("pyarrow is required") from error
    enriched_file = pq.ParquetFile(enriched_path)
    metadata_file = pq.ParquetFile(metadata_path)
    if set(enriched_file.schema_arrow.names) != ENRICHED_COLUMNS:
        raise InstitutionalBooksPilotError("pilot enriched columns differ")
    if set(metadata_file.schema_arrow.names) != METADATA_COLUMNS:
        raise InstitutionalBooksPilotError("pilot metadata columns differ")
    enriched = enriched_file.read().to_pylist()
    barcodes = pa.array([row["barcode_src"] for row in enriched])
    metadata_table = metadata_file.read(
        columns=sorted(
            METADATA_COLUMNS - {"date_types_src", "language_distribution_gen"}
        )
    )
    metadata_table = metadata_table.filter(
        pc.is_in(metadata_table["barcode_src"], value_set=barcodes)
    )
    return metadata_table.to_pylist(), enriched


def build_pilot(
    metadata_path: Path,
    enriched_path: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Build and seal the first candidate population from exact local shards."""

    if receipt_path.exists() or receipt_path.is_symlink():
        raise InstitutionalBooksPilotError("pilot receipt already exists")
    metadata, enriched = _read_inputs(metadata_path, enriched_path)
    candidates, statistics = build_candidate_population(metadata, enriched)
    _atomic_jsonl(output_path, candidates)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "source": {
            "repository": ENRICHED_REPOSITORY,
            "revision": ENRICHED_REVISION,
            "repository_path": PILOT_SHARD_REPOSITORY_PATH,
            "local_path": str(enriched_path.resolve()),
            "bytes": PILOT_SHARD_BYTES,
            "sha256": PILOT_SHARD_SHA256,
        },
        "metadata": {
            "local_path": str(metadata_path.resolve()),
            "bytes": METADATA_PARQUET_BYTES,
            "sha256": METADATA_PARQUET_SHA256,
        },
        "statistics": statistics,
        "output": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "candidate_identities_sha256": canonical_sha256(
                [row["candidate_identity_sha256"] for row in candidates]
            ),
        },
        "book_text_downloaded": True,
        "hermes_calls_completed": 0,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_jsonl(receipt_path, [receipt])
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--enriched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_pilot(args.metadata, args.enriched, args.output, args.receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
