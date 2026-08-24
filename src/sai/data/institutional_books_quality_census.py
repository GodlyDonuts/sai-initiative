"""Measure deduplicated, rights-bounded Institutional Books quality tiers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books import (
    METADATA_PARQUET_BYTES,
    METADATA_PARQUET_SHA256,
    METADATA_REPOSITORY,
    METADATA_REVISION,
)
from sai.data.institutional_books_selection import (
    ALLOWED_RIGHTS_CODES,
    MAXIMUM_TOKENS,
    MINIMUM_TOKENS,
    _read_parquet,
    _representatives,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-institutional-books-quality-census-v1"
OCR_THRESHOLDS = (80, 90, 95)


class InstitutionalBooksQualityCensusError(RuntimeError):
    """The metadata identity or quality-tier accounting differs."""


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _base_eligible(row: dict[str, Any]) -> bool:
    rights = row.get("hathitrust_data_ext") or {}
    tokens = _integer(row.get("token_count_o200k_base_gen"))
    return bool(
        isinstance(row.get("language_gen"), str)
        and row["language_gen"]
        and isinstance(row.get("topic_or_subject_gen"), str)
        and row["topic_or_subject_gen"]
        and isinstance(row.get("ocr_score_gen"), int)
        and tokens is not None
        and MINIMUM_TOKENS <= tokens <= MAXIMUM_TOKENS
        and isinstance(rights, dict)
        and rights.get("rights_code") in ALLOWED_RIGHTS_CODES
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    languages: dict[str, Counter[str]] = {
        f"ocr_{threshold}": Counter() for threshold in OCR_THRESHOLDS
    }
    topics: dict[str, Counter[str]] = {
        f"ocr_{threshold}": Counter() for threshold in OCR_THRESHOLDS
    }
    for row in rows:
        if not _base_eligible(row):
            continue
        language = row["language_gen"]
        topic = row["topic_or_subject_gen"]
        tokens = row["token_count_o200k_base_gen"]
        ocr = row["ocr_score_gen"]
        counts["base_eligible_rows"] += 1
        counts["base_eligible_tokens"] += tokens
        for threshold in OCR_THRESHOLDS:
            if ocr < threshold:
                continue
            tier = f"ocr_{threshold}"
            counts[f"{tier}_rows"] += 1
            counts[f"{tier}_tokens"] += tokens
            if language == "eng":
                counts[f"english_{tier}_rows"] += 1
                counts[f"english_{tier}_tokens"] += tokens
            else:
                counts[f"translation_{tier}_rows"] += 1
                counts[f"translation_{tier}_tokens"] += tokens
            languages[tier][language] += 1
            topics[tier][topic] += 1
    return {
        "counts": dict(sorted(counts.items())),
        "rows_by_language": {
            tier: dict(sorted(values.items()))
            for tier, values in languages.items()
        },
        "rows_by_topic": {
            tier: dict(sorted(values.items())) for tier, values in topics.items()
        },
    }


def build_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Deduplicate all metadata rows and measure nested quality tiers."""

    representatives, links = _representatives(rows)
    summary = _summarize(representatives)
    counts = summary["counts"]
    if (
        not rows
        or len(representatives) > len(rows)
        or counts.get("english_ocr_95_rows", 0)
        > counts.get("english_ocr_90_rows", 0)
        or counts.get("english_ocr_90_rows", 0)
        > counts.get("english_ocr_80_rows", 0)
        or counts.get("translation_ocr_95_rows", 0)
        > counts.get("translation_ocr_90_rows", 0)
        or counts.get("translation_ocr_90_rows", 0)
        > counts.get("translation_ocr_80_rows", 0)
    ):
        raise InstitutionalBooksQualityCensusError(
            "Institutional Books quality tiers differ"
        )
    return {
        "schema": SCHEMA,
        "status": "complete_nontraining_institutional_books_quality_census",
        "method": {
            "duplicate_policy": (
                "one_best_metadata_representative_per_connected_component"
            ),
            "allowed_rights_codes": sorted(ALLOWED_RIGHTS_CODES),
            "minimum_tokens": MINIMUM_TOKENS,
            "maximum_tokens": MAXIMUM_TOKENS,
            "ocr_thresholds": list(OCR_THRESHOLDS),
            "english_language_code": "eng",
            "nonenglish_tiers_are_translation_candidates": True,
        },
        "source_rows": len(rows),
        "duplicate_graph_links": links,
        "duplicate_components": len(representatives),
        **summary,
        "source_text_read": False,
        "source_text_persisted": False,
        "quality_tier_is_training_admission": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }


def build_census(source: Path, output: Path) -> dict[str, Any]:
    """Verify the pinned metadata Parquet and atomically write the census."""

    if output.exists() or output.is_symlink():
        raise InstitutionalBooksQualityCensusError("quality census output exists")
    payload = build_payload(_read_parquet(source))
    if (
        source.stat().st_size != METADATA_PARQUET_BYTES
        or sha256_file(source) != METADATA_PARQUET_SHA256
    ):
        raise InstitutionalBooksQualityCensusError("metadata source differs")
    payload["source"] = {
        "repository": METADATA_REPOSITORY,
        "revision": METADATA_REVISION,
        "bytes": METADATA_PARQUET_BYTES,
        "sha256": METADATA_PARQUET_SHA256,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_census(args.source, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_rows": result["source_rows"],
                "duplicate_components": result["duplicate_components"],
                "counts": result["counts"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
