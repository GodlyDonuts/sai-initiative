"""Build a source-safe publication receipt for the arXiv abstract audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.arxiv_abstracts_audit_population import (
    EXPECTED_ROWS,
    REPOSITORY,
    REVISION,
    SOURCE_ID,
    SOURCE_ROWS,
    TEMPORAL_STRATA,
)
from sai.data.common_pile_rights_audit import SCHEMA as RIGHTS_SCHEMA
from sai.data.reservoir_audit_duplicates import SCHEMA as DUPLICATE_SCHEMA
from sai.data.reservoir_audit_population import SCHEMA as POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-arxiv-abstracts-audit-publication-v1"
EXPECTED_CLEAN_ROWS = 1_023
EXPECTED_CONTAMINATED_ROWS = 1


class ArxivAbstractsAuditPublicationError(RuntimeError):
    """The arXiv evidence or source-safe publication boundary differs."""


def _load_signed(path: Path, schema: str, label: str) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size > 16 << 20
    ):
        raise ArxivAbstractsAuditPublicationError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ArxivAbstractsAuditPublicationError(
            f"{label} cannot be decoded"
        ) from error
    if not isinstance(payload, dict):
        raise ArxivAbstractsAuditPublicationError(f"{label} differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise ArxivAbstractsAuditPublicationError(f"{label} receipt differs")
    return payload


def summarize_publication(
    source: dict[str, Any],
    clean: dict[str, Any],
    duplicates: dict[str, Any],
    rights: dict[str, Any],
) -> dict[str, Any]:
    """Validate exact coverage and emit a text-free result summary."""

    source_population = source.get("population", {})
    clean_population = clean.get("population", {})
    source_snapshot = source.get("source_snapshot", {})
    rights_source = rights.get("summary", {}).get("by_source", {}).get(SOURCE_ID, {})
    if (
        source.get("schema") != POPULATION_SCHEMA
        or source.get("status") != "complete"
        or source_snapshot.get("repository") != REPOSITORY
        or source_snapshot.get("revision") != REVISION
        or source_snapshot.get("rows") != SOURCE_ROWS
        or source_population.get("rows") != EXPECTED_ROWS
        or source.get("dataset_server_batches") != TEMPORAL_STRATA
        or source.get("source_disjoint_from_audit_populations") is not True
        or source.get("source_declared_cc0") is not True
        or source.get("rights_verification_complete") is not False
        or source.get("benchmark_contamination_screen_complete") is not False
        or source.get("hermes_judgments_complete") is not False
        or source.get("training_ready") is not False
        or clean.get("schema") != POPULATION_SCHEMA
        or clean.get("status") != "complete"
        or clean.get("source_population", {}).get("receipt_sha256")
        != source.get("receipt_sha256")
        or clean.get("input_rows") != EXPECTED_ROWS
        or clean.get("clean_rows") != EXPECTED_CLEAN_ROWS
        or clean.get("contaminated_rows") != EXPECTED_CONTAMINATED_ROWS
        or clean.get("word_overlap_shingles") != 0
        or clean.get("code_overlap_shingles") != 1
        or clean_population.get("rows") != EXPECTED_CLEAN_ROWS
        or clean.get("by_source") != {SOURCE_ID: EXPECTED_CLEAN_ROWS}
        or clean.get("benchmark_contamination_screen_complete") is not True
        or clean.get("hermes_judgments_complete") is not False
        or clean.get("training_ready") is not False
        or duplicates.get("schema") != DUPLICATE_SCHEMA
        or duplicates.get("status") != "complete"
        or duplicates.get("population_receipt_sha256")
        != clean.get("receipt_sha256")
        or duplicates.get("candidate_rows") != EXPECTED_CLEAN_ROWS
        or duplicates.get("candidate_pairs_compared")
        != EXPECTED_CLEAN_ROWS * (EXPECTED_CLEAN_ROWS - 1) // 2
        or duplicates.get("flagged_pairs") != 0
        or duplicates.get("cross_source_flagged_pairs") != 0
        or duplicates.get("pairs") != []
        or duplicates.get("groups") != []
        or duplicates.get("audit_sample_deduplication_complete") is not True
        or duplicates.get("full_reservoir_deduplication_complete") is not False
        or duplicates.get("training_ready") is not False
        or rights.get("schema") != RIGHTS_SCHEMA
        or rights.get("status")
        != "complete_declaration_audit_not_legal_clearance"
        or rights.get("population", {}).get("receipt_sha256")
        != source.get("receipt_sha256")
        or rights.get("summary", {}).get("rows") != EXPECTED_ROWS
        or rights_source.get("rows") != EXPECTED_ROWS
        or rights_source.get("recognized_declaration_rows") != EXPECTED_ROWS
        or rights_source.get("canonical_license:CC0-1.0") != EXPECTED_ROWS
        or rights_source.get("rights_hold_rows") != 0
        or rights_source.get("attribution_required_rows") != 0
        or rights_source.get("share_alike_required_rows") != 0
        or rights.get("source_provenance_verification_complete") is not False
        or rights.get("source_wide_rights_clearance_established") is not False
        or rights.get("legal_clearance_established") is not False
        or rights.get("training_ready") is not False
    ):
        raise ArxivAbstractsAuditPublicationError(
            "arXiv publication evidence differs"
        )
    return {
        "source_snapshot": source_snapshot,
        "temporal_strata": TEMPORAL_STRATA,
        "input_rows": EXPECTED_ROWS,
        "clean_rows": EXPECTED_CLEAN_ROWS,
        "contaminated_rows": EXPECTED_CONTAMINATED_ROWS,
        "word_overlap_shingles": 0,
        "code_overlap_shingles": 1,
        "candidate_pairs_compared": duplicates["candidate_pairs_compared"],
        "near_duplicate_pairs": 0,
        "source_population_receipt_sha256": source["receipt_sha256"],
        "clean_population_receipt_sha256": clean["receipt_sha256"],
        "duplicate_report_receipt_sha256": duplicates["receipt_sha256"],
        "rights_declaration_audit_receipt_sha256": rights["receipt_sha256"],
        "recognized_cc0_declaration_rows": EXPECTED_ROWS,
        "rights_hold_rows": 0,
    }


def _verify_descriptor(root: Path, descriptor: dict[str, Any], label: str) -> Path:
    name = descriptor.get("path")
    if not isinstance(name, str) or Path(name).name != name:
        raise ArxivAbstractsAuditPublicationError(f"{label} path differs")
    path = root / name
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise ArxivAbstractsAuditPublicationError(f"{label} custody differs")
    return path


def build_publication(
    source_root: Path,
    clean_root: Path,
    duplicate_path: Path,
    rights_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Seal custody without publishing abstracts or contamination decisions."""

    if output_path.exists() or output_path.is_symlink():
        raise ArxivAbstractsAuditPublicationError("publication output exists")
    source_receipt_path = source_root / "receipt.json"
    clean_receipt_path = clean_root / "receipt.json"
    source = _load_signed(source_receipt_path, POPULATION_SCHEMA, "source")
    clean = _load_signed(clean_receipt_path, POPULATION_SCHEMA, "clean")
    duplicates = _load_signed(duplicate_path, DUPLICATE_SCHEMA, "duplicates")
    rights = _load_signed(rights_path, RIGHTS_SCHEMA, "rights")
    summary = summarize_publication(source, clean, duplicates, rights)
    source_artifacts = {
        label: _verify_descriptor(source_root, source[label], f"source {label}")
        for label in ("population", "lineage", "batch_receipts")
    }
    clean_artifacts = {
        label: _verify_descriptor(clean_root, clean[label], f"clean {label}")
        for label in ("population", "lineage", "decisions")
    }
    payload = {
        "schema": SCHEMA,
        "status": "complete_pre_hermes_source_safe_evidence",
        **summary,
        "artifact_file_sha256": {
            "source_receipt": sha256_file(source_receipt_path),
            "clean_receipt": sha256_file(clean_receipt_path),
            "duplicate_report": sha256_file(duplicate_path),
            "rights_audit": sha256_file(rights_path),
            **{
                f"source_{label}": sha256_file(path)
                for label, path in source_artifacts.items()
            },
            **{
                f"clean_{label}": sha256_file(path)
                for label, path in clean_artifacts.items()
            },
        },
        "source_text_published": False,
        "individual_contamination_decisions_published": False,
        "absolute_local_paths_persisted": False,
        "rights_verification_complete": False,
        "hermes_judgments_complete": False,
        "quality_compilation_complete": False,
        "full_source_ingestion_authorized": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--clean-root", type=Path, required=True)
    parser.add_argument("--duplicate-report", type=Path, required=True)
    parser.add_argument("--rights-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_publication(
        args.source_root,
        args.clean_root,
        args.duplicate_report,
        args.rights_audit,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
