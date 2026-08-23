"""Seal benchmark-disjoint PEP census survivors for Hermes compilation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import CANDIDATE_SCHEMA, normalize_candidate
from sai.data.bounded_pilot_compiler_population import _load_attribution
from sai.data.common_pile_pep_census import SCHEMA as CENSUS_SCHEMA
from sai.data.common_pile_pep_census import SOURCE_ID
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.reservoir_audit_population import (
    LINEAGE_SCHEMA,
    SCHEMA,
    _excerpt,
    _write_jsonl,
)
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

STRATUM = "software_standards_and_design"
SOURCE_TYPE = "documentation"


class CommonPilePepCompilerPopulationError(RuntimeError):
    """The PEP census, attribution join, or compiler population differs."""


def _descriptor(payload: dict[str, Any], prefix: str = "output") -> dict[str, Any]:
    return {
        "path": payload.get(f"{prefix}_path"),
        "bytes": payload.get(f"{prefix}_bytes"),
        "sha256": payload.get(f"{prefix}_sha256"),
    }


def _candidate_and_lineage(
    document: dict[str, Any],
    attribution: dict[str, Any],
    census_receipt_sha256: str,
    ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = attribution.get("source")
    rights = attribution.get("rights_declaration")
    document_source = document.get("source")
    text = document.get("text")
    if (
        not isinstance(source, dict)
        or not isinstance(rights, dict)
        or not isinstance(document_source, dict)
        or not isinstance(text, str)
        or not text
        or document_source.get("dataset") != source.get("dataset")
        or document_source.get("row_id") != attribution.get("row_id")
        or document_source.get("license") != rights.get("canonical_license")
        or rights.get("rights_hold") is not False
        or rights.get("canonical_license") != "LicenseRef-Public-Domain"
        or attribution.get("record_sha256") is None
    ):
        raise CommonPilePepCompilerPopulationError("PEP census join differs")
    excerpt, excerpt_method = _excerpt(text)
    full_text_sha256 = hashlib.sha256(text.encode()).hexdigest()
    excerpt_sha256 = hashlib.sha256(excerpt.encode()).hexdigest()
    locator = {
        "source_file": source["source_file"],
        "row_index": source["row_index"],
        "row_id": attribution["row_id"],
        "retained_document_identity_sha256": document["identity_sha256"],
    }
    provenance = {
        "census_receipt_sha256": census_receipt_sha256,
        "retained_document_identity_sha256": document["identity_sha256"],
        "decontamination_evidence_sha256": document["verification"][
            "evidence_sha256"
        ],
        "attribution_record_sha256": attribution["record_sha256"],
        "source": source,
        "row_id": attribution["row_id"],
        "full_text_sha256": full_text_sha256,
        "excerpt_sha256": excerpt_sha256,
    }
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "text": excerpt,
        "source": {
            "dataset": source["dataset"],
            "revision": source["revision"],
            "row_id": attribution["row_id"],
            "license": rights["canonical_license"],
            "source_type": SOURCE_TYPE,
        },
        "source_content_sha256": excerpt_sha256,
        "provenance_sha256": canonical_sha256(provenance),
    }
    candidate["candidate_identity_sha256"] = canonical_sha256(candidate)
    candidate = normalize_candidate(candidate)
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "ordinal": ordinal,
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "source_id": SOURCE_ID,
        "stratum": STRATUM,
        "selection_key": document["identity_sha256"],
        "repository": source["dataset"],
        "revision": source["revision"],
        "license": rights["canonical_license"],
        "access": "public",
        "path": source["source_file"],
        "locator": locator,
        "full_file_content_verified": True,
        "full_text_bytes": len(text.encode()),
        "full_text_sha256": full_text_sha256,
        "excerpt_method": excerpt_method,
        "excerpt_bytes": len(excerpt.encode()),
        "excerpt_sha256": excerpt_sha256,
        "census_receipt_sha256": census_receipt_sha256,
        "attribution_record_sha256": attribution["record_sha256"],
        "benchmark_decontamination_complete": True,
        "bounded_near_duplicate_filter_complete": True,
        "raw_source_is_training_ready": False,
    }
    lineage["lineage_sha256"] = canonical_sha256(lineage)
    return candidate, lineage


def build_population(census_root: Path, output_root: Path) -> dict[str, Any]:
    """Join all PEP survivors to attribution and emit shared compiler inputs."""

    if output_root.exists() or output_root.is_symlink():
        raise CommonPilePepCompilerPopulationError(
            "PEP compiler output already exists"
        )
    census_path = census_root / "receipt.json"
    census = _load_receipt(census_path)
    duplicate = census.get("near_duplicate_filter")
    attribution_descriptor = census.get("attribution_manifest")
    if (
        census.get("schema") != CENSUS_SCHEMA
        or census.get("status") != "complete_filtered_nontraining_parent_census"
        or census.get("source_id") != SOURCE_ID
        or census.get("complete_parent_census") is not True
        or census.get("training_ready") is not False
        or not isinstance(duplicate, dict)
        or not isinstance(attribution_descriptor, dict)
    ):
        raise CommonPilePepCompilerPopulationError("PEP census receipt differs")
    documents_path = _bound_file(census_root, _descriptor(duplicate))
    attribution_path = _bound_file(census_root, _descriptor(attribution_descriptor))
    attribution = _load_attribution(attribution_path)
    documents = []
    identities = set()
    try:
        with documents_path.open() as handle:
            for line in handle:
                document = normalize_document(json.loads(line))
                identity = document["identity_sha256"]
                if identity in identities:
                    raise CommonPilePepCompilerPopulationError(
                        "PEP survivor identity repeats"
                    )
                identities.add(identity)
                documents.append(document)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise CommonPilePepCompilerPopulationError(
            "PEP survivor population differs"
        ) from error
    if (
        not documents
        or len(documents) != duplicate.get("output_documents")
        or len(documents) != attribution_descriptor.get("records")
        or identities != set(attribution)
    ):
        raise CommonPilePepCompilerPopulationError("PEP survivor coverage differs")
    candidates = []
    lineage = []
    for document in documents:
        candidate, source_lineage = _candidate_and_lineage(
            document,
            attribution[document["identity_sha256"]],
            census["receipt_sha256"],
            len(candidates),
        )
        candidates.append(candidate)
        lineage.append(source_lineage)
    candidate_identities = [row["candidate_identity_sha256"] for row in candidates]
    if len(candidate_identities) != len(set(candidate_identities)):
        raise CommonPilePepCompilerPopulationError("PEP compiler identities repeat")

    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        candidates_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        receipt_path = temporary / "receipt.json"
        _write_jsonl(candidates_path, candidates)
        _write_jsonl(lineage_path, lineage)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "selection_method": (
                "complete_pep_parent_census_then_audit_exclusion_benchmark_"
                "decontamination_near_deduplication_and_exact_attribution_join"
            ),
            "statistically_representative": False,
            "source_census": {
                "root_name": census_root.name,
                "receipt_file_sha256": sha256_file(census_path),
                "receipt_sha256": census["receipt_sha256"],
                "survivor_file_sha256": sha256_file(documents_path),
                "attribution_file_sha256": sha256_file(attribution_path),
            },
            "population": {
                "path": candidates_path.name,
                "rows": len(candidates),
                "bytes": candidates_path.stat().st_size,
                "sha256": sha256_file(candidates_path),
                "ordered_identities_sha256": canonical_sha256(candidate_identities),
            },
            "lineage": {
                "path": lineage_path.name,
                "rows": len(lineage),
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
                "ordered_rows_sha256": canonical_sha256(lineage),
            },
            "by_source": {SOURCE_ID: len(candidates)},
            "by_stratum": {STRATUM: len(candidates)},
            "complete_census_survivor_coverage": True,
            "benchmark_contamination_screen_complete": True,
            "bounded_near_duplicate_filter_complete": True,
            "exact_attribution_coverage": True,
            "hermes_judgments_complete": False,
            "quality_compilation_complete": False,
            "representation_verification_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _write_jsonl(receipt_path, [receipt])
        os.replace(temporary, output_root)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_population(args.census_root, args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
