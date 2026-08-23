"""Build a source-safe publication receipt for the PubMed full-text screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.common_pile_rights_audit import SCHEMA as RIGHTS_SCHEMA
from sai.data.pubmed_fulltext_audit_population import (
    EXPECTED_ROWS,
    INDEX_STRATA,
    REPOSITORY,
    REVISION,
    WINDOWS_PER_STRATUM,
)
from sai.data.reservoir_audit_duplicates import SCHEMA as DUPLICATE_SCHEMA
from sai.data.reservoir_audit_population import SCHEMA as POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pubmed-fulltext-audit-publication-v1"


class PubmedFulltextPublicationError(RuntimeError):
    """The PubMed evidence or source-safe publication boundary differs."""


def _load_single(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PubmedFulltextPublicationError("publication input is unsafe")
    try:
        rows = [json.loads(line) for line in path.open()]
    except (OSError, json.JSONDecodeError) as error:
        raise PubmedFulltextPublicationError(
            "publication input cannot be decoded"
        ) from error
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise PubmedFulltextPublicationError("publication input differs")
    return rows[0]


def _valid_receipt(payload: dict[str, Any], schema: str) -> bool:
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return payload.get("schema") == schema and payload.get(
        "receipt_sha256"
    ) == canonical_sha256(unsigned)


def summarize_publication(
    source: dict[str, Any],
    clean: dict[str, Any],
    rights: dict[str, Any],
    duplicates: dict[str, Any],
) -> dict[str, Any]:
    """Validate exact gate coverage and return a text-free public summary."""

    source_population = source.get("population", {})
    clean_population = clean.get("population", {})
    rights_summary = rights.get("summary", {})
    expected_batches = INDEX_STRATA * WINDOWS_PER_STRATUM
    if (
        not _valid_receipt(source, POPULATION_SCHEMA)
        or not _valid_receipt(clean, POPULATION_SCHEMA)
        or not _valid_receipt(rights, RIGHTS_SCHEMA)
        or not _valid_receipt(duplicates, DUPLICATE_SCHEMA)
        or source.get("status") != "complete"
        or source.get("source_snapshot", {}).get("repository") != REPOSITORY
        or source.get("source_snapshot", {}).get("revision") != REVISION
        or source_population.get("rows") != EXPECTED_ROWS
        or source.get("batch_receipts", {}).get("rows") != expected_batches
        or source.get("training_ready") is not False
        or clean.get("source_population", {}).get("receipt_sha256")
        != source.get("receipt_sha256")
        or clean.get("input_rows") != EXPECTED_ROWS
        or clean.get("clean_rows") != clean_population.get("rows")
        or clean.get("clean_rows", 0) + clean.get("contaminated_rows", 0)
        != EXPECTED_ROWS
        or clean.get("benchmark_contamination_screen_complete") is not True
        or clean.get("training_ready") is not False
        or rights.get("population", {}).get("receipt_sha256")
        != source.get("receipt_sha256")
        or rights_summary.get("rows") != EXPECTED_ROWS
        or rights.get("training_ready") is not False
        or duplicates.get("population_receipt_sha256") != source.get("receipt_sha256")
        or duplicates.get("candidate_rows") != EXPECTED_ROWS
        or duplicates.get("audit_sample_deduplication_complete") is not True
        or duplicates.get("training_ready") is not False
    ):
        raise PubmedFulltextPublicationError("publication evidence differs")
    return {
        "source_snapshot": source["source_snapshot"],
        "input_rows": EXPECTED_ROWS,
        "clean_rows": clean["clean_rows"],
        "contaminated_rows": clean["contaminated_rows"],
        "word_overlap_shingles": clean["word_overlap_shingles"],
        "code_overlap_shingles": clean["code_overlap_shingles"],
        "input_by_stratum": source["by_stratum"],
        "clean_by_stratum": clean["by_stratum"],
        "input_by_license": source["by_license"],
        "recognized_license_rows": rights_summary["rows"],
        "rights_hold_rows": sum(
            row.get("rights_hold_rows", 0)
            for row in rights_summary.get("by_source", {}).values()
        ),
        "flagged_near_duplicate_pairs": duplicates["flagged_pairs"],
        "population_receipt_sha256": source["receipt_sha256"],
        "clean_population_receipt_sha256": clean["receipt_sha256"],
        "rights_receipt_sha256": rights["receipt_sha256"],
        "duplicate_receipt_sha256": duplicates["receipt_sha256"],
    }


def build_publication(
    population_root: Path,
    clean_root: Path,
    rights_path: Path,
    duplicates_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Create one sanitized evidence envelope without source text."""

    if output_path.exists() or output_path.is_symlink():
        raise PubmedFulltextPublicationError("publication output already exists")
    source_receipt_path = population_root / "receipt.json"
    batch_receipts_path = population_root / "batch_receipts.jsonl"
    clean_receipt_path = clean_root / "receipt.json"
    decisions_path = clean_root / "decisions.jsonl"
    source = _load_single(source_receipt_path)
    clean = _load_single(clean_receipt_path)
    rights = _load_single(rights_path)
    duplicates = _load_single(duplicates_path)
    summary = summarize_publication(source, clean, rights, duplicates)
    if (
        not batch_receipts_path.is_file()
        or sha256_file(batch_receipts_path)
        != source.get("batch_receipts", {}).get("sha256")
        or not decisions_path.is_file()
        or sha256_file(decisions_path) != clean.get("decisions", {}).get("sha256")
    ):
        raise PubmedFulltextPublicationError("publication custody differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete_pre_hermes_source_safe_evidence",
        **summary,
        "artifacts": {
            "source_population_receipt_file_sha256": sha256_file(source_receipt_path),
            "dataset_server_batch_receipts_file_sha256": sha256_file(
                batch_receipts_path
            ),
            "clean_population_receipt_file_sha256": sha256_file(clean_receipt_path),
            "individual_decisions_file_sha256": sha256_file(decisions_path),
            "rights_file_sha256": sha256_file(rights_path),
            "duplicate_file_sha256": sha256_file(duplicates_path),
        },
        "source_text_persisted": False,
        "benchmark_text_persisted": False,
        "individual_decisions_published": False,
        "absolute_local_paths_persisted": False,
        "source_provenance_verification_complete": False,
        "source_wide_rights_clearance_established": False,
        "hermes_judgments_complete": False,
        "quality_compilation_complete": False,
        "full_source_population_screened": False,
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
    parser.add_argument("--rights", type=Path, required=True)
    parser.add_argument("--duplicates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_publication(
        args.population_root,
        args.clean_root,
        args.rights,
        args.duplicates,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
