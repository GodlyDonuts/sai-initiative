"""Build a source-safe publication receipt for the UltraData Math tier audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.audit_population_decontamination import DECISION_SCHEMA
from sai.data.benchmark_contamination_screen import SCHEMA as SCREEN_SCHEMA
from sai.data.reservoir_audit_population import SCHEMA as POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file
from sai.data.ultradata_math_tier_audit_population import (
    EXPECTED_BATCHES,
    EXPECTED_ROWS,
    REPOSITORY,
    REVISION,
)

SCHEMA = "sai-ultradata-math-tier-audit-publication-v1"


class UltraDataMathTierPublicationError(RuntimeError):
    """The audit evidence or source-safe publication boundary differs."""


def _load_single(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise UltraDataMathTierPublicationError("publication input is unsafe")
    try:
        rows = [json.loads(line) for line in path.open()]
    except (OSError, json.JSONDecodeError) as error:
        raise UltraDataMathTierPublicationError(
            "publication input cannot be decoded"
        ) from error
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise UltraDataMathTierPublicationError("publication input differs")
    return rows[0]


def _valid_receipt(payload: dict[str, Any], schema: str) -> bool:
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return (
        payload.get("schema") == schema
        and payload.get("receipt_sha256") == canonical_sha256(unsigned)
    )


def summarize_publication(
    source: dict[str, Any], clean: dict[str, Any], screen: dict[str, Any]
) -> dict[str, Any]:
    """Validate exact coverage and return the text-free public summary."""

    source_population = source.get("population", {})
    clean_population = clean.get("population", {})
    summary = screen.get("summary", {})
    if (
        not _valid_receipt(source, POPULATION_SCHEMA)
        or not _valid_receipt(clean, POPULATION_SCHEMA)
        or not _valid_receipt(screen, SCREEN_SCHEMA)
        or source.get("status") != "complete"
        or clean.get("status") != "complete"
        or screen.get("status") != "complete"
        or source.get("source_snapshot", {}).get("repository") != REPOSITORY
        or source.get("source_snapshot", {}).get("revision") != REVISION
        or source_population.get("rows") != EXPECTED_ROWS
        or source.get("batch_receipts", {}).get("rows") != EXPECTED_BATCHES
        or source.get("hermes_judgments_complete") is not False
        or source.get("training_ready") is not False
        or clean.get("source_population", {}).get("receipt_sha256")
        != source.get("receipt_sha256")
        or clean.get("input_rows") != EXPECTED_ROWS
        or clean.get("clean_rows") != clean_population.get("rows")
        or clean.get("contaminated_rows") + clean.get("clean_rows")
        != EXPECTED_ROWS
        or clean.get("benchmark_contamination_screen_complete") is not True
        or clean.get("hermes_judgments_complete") is not False
        or clean.get("training_ready") is not False
        or screen.get("population", {}).get("receipt_sha256")
        != source.get("receipt_sha256")
        or summary.get("rows") != EXPECTED_ROWS
        or summary.get("clean_rows") != clean.get("clean_rows")
        or summary.get("contaminated_rows") != clean.get("contaminated_rows")
        or summary.get("word_overlap_shingles") != clean.get("word_overlap_shingles")
        or summary.get("code_overlap_shingles") != clean.get("code_overlap_shingles")
    ):
        raise UltraDataMathTierPublicationError("publication evidence differs")
    return {
        "source_snapshot": source["source_snapshot"],
        "input_rows": EXPECTED_ROWS,
        "clean_rows": clean["clean_rows"],
        "contaminated_rows": clean["contaminated_rows"],
        "word_overlap_shingles": clean["word_overlap_shingles"],
        "code_overlap_shingles": clean["code_overlap_shingles"],
        "input_by_stratum": source["by_stratum"],
        "clean_by_stratum": clean["by_stratum"],
        "population_receipt_sha256": source["receipt_sha256"],
        "clean_population_receipt_sha256": clean["receipt_sha256"],
        "screen_receipt_sha256": screen["receipt_sha256"],
    }


def build_publication(
    population_root: Path,
    clean_root: Path,
    screen_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create one sanitized evidence envelope without source or benchmark text."""

    if output_path.exists() or output_path.is_symlink():
        raise UltraDataMathTierPublicationError("publication output already exists")
    source_receipt_path = population_root / "receipt.json"
    clean_receipt_path = clean_root / "receipt.json"
    batch_receipts_path = population_root / "batch_receipts.jsonl"
    decisions_path = clean_root / "decisions.jsonl"
    source = _load_single(source_receipt_path)
    clean = _load_single(clean_receipt_path)
    screen = _load_single(screen_path)
    summary = summarize_publication(source, clean, screen)
    decision_descriptor = clean.get("decisions", {})
    if (
        not batch_receipts_path.is_file()
        or sha256_file(batch_receipts_path)
        != source.get("batch_receipts", {}).get("sha256")
        or not decisions_path.is_file()
        or sha256_file(decisions_path) != decision_descriptor.get("sha256")
        or decision_descriptor.get("rows") != EXPECTED_ROWS
    ):
        raise UltraDataMathTierPublicationError("publication custody differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete_pre_hermes_source_safe_evidence",
        **summary,
        "artifacts": {
            "source_population_receipt_file_sha256": sha256_file(
                source_receipt_path
            ),
            "dataset_server_batch_receipts_file_sha256": sha256_file(
                batch_receipts_path
            ),
            "benchmark_screen_file_sha256": sha256_file(screen_path),
            "clean_population_receipt_file_sha256": sha256_file(
                clean_receipt_path
            ),
            "individual_decisions_file_sha256": sha256_file(decisions_path),
        },
        "individual_decisions_schema": DECISION_SCHEMA,
        "source_text_persisted": False,
        "benchmark_text_persisted": False,
        "individual_decisions_published": False,
        "absolute_local_paths_persisted": False,
        "hermes_judgments_complete": False,
        "quality_compilation_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_publication(
        args.population_root, args.clean_root, args.screen, args.output
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
