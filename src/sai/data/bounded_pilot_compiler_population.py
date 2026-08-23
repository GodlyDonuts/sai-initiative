"""Build a provenance-complete compiler population from bounded source pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import (
    CANDIDATE_SCHEMA,
    _atomic_create,
    normalize_candidate,
)
from sai.data.attribution_manifest import SCHEMA as ATTRIBUTION_SCHEMA
from sai.data.common_pile_streaming_pilot import SCHEMA as PILOT_SCHEMA
from sai.data.cross_source_pilot_duplicates import SCHEMA as CROSS_SOURCE_SCHEMA
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-bounded-pilot-compiler-population-v1"
LINEAGE_SCHEMA = "sai-bounded-pilot-compiler-lineage-v1"
SOURCE_TYPES = {
    "common_pile_pressbooks": "textbook",
    "common_pile_public_domain_review": "educational_web",
}


class BoundedPilotCompilerPopulationError(RuntimeError):
    """A pilot, cross-source result, attribution, or population differs."""


def _descriptor(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "path": row.get(f"{prefix}_path"),
        "bytes": row.get(f"{prefix}_bytes"),
        "sha256": row.get(f"{prefix}_sha256"),
    }


def _load_attribution(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise BoundedPilotCompilerPopulationError("attribution manifest is unsafe")
    records: dict[str, dict[str, Any]] = {}
    row_ids: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BoundedPilotCompilerPopulationError(
                    f"attribution row {line_number} cannot be decoded"
                ) from error
            if not isinstance(row, dict):
                raise BoundedPilotCompilerPopulationError(
                    f"attribution row {line_number} differs"
                )
            unsigned = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            identity = row.get("identity_sha256")
            source = row.get("source")
            rights = row.get("rights_declaration")
            row_id = row.get("row_id")
            if (
                row.get("schema") != ATTRIBUTION_SCHEMA
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in records
                or not isinstance(row_id, str)
                or len(row_id) != 64
                or row_id in row_ids
                or not isinstance(source, dict)
                or any(
                    not isinstance(source.get(key), expected)
                    for key, expected in (
                        ("dataset", str),
                        ("revision", str),
                        ("source_file", str),
                        ("row_index", int),
                        ("domain", str),
                    )
                )
                or any(
                    not source[key]
                    for key in ("dataset", "revision", "source_file", "domain")
                )
                or isinstance(source.get("row_index"), bool)
                or source["row_index"] < 0
                or not isinstance(rights, dict)
                or rights.get("rights_hold") is not False
                or not isinstance(rights.get("canonical_license"), str)
                or not rights["canonical_license"]
                or row.get("record_sha256") != canonical_sha256(unsigned)
            ):
                raise BoundedPilotCompilerPopulationError(
                    f"attribution row {line_number} differs"
                )
            records[identity] = row
            row_ids.add(row_id)
    if not records:
        raise BoundedPilotCompilerPopulationError("attribution manifest is empty")
    return records


def _pilot_bindings(
    pilot_roots: list[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, dict[str, Any]]]]:
    if len(pilot_roots) < 2 or len(pilot_roots) != len(set(pilot_roots)):
        raise BoundedPilotCompilerPopulationError("pilot roots differ")
    pilots: dict[str, dict[str, Any]] = {}
    attribution_by_identity: dict[str, tuple[str, dict[str, Any]]] = {}
    for root in pilot_roots:
        receipt_path = root / "receipt.json"
        receipt = _load_receipt(receipt_path)
        source_id = receipt.get("source_id")
        attribution = receipt.get("attribution_manifest")
        if (
            receipt.get("schema") != PILOT_SCHEMA
            or receipt.get("status") != "complete_nontraining_pilot"
            or receipt.get("training_ready") is not False
            or source_id not in SOURCE_TYPES
            or source_id in pilots
            or not isinstance(attribution, dict)
        ):
            raise BoundedPilotCompilerPopulationError("pilot receipt differs")
        attribution_path = _bound_file(root, _descriptor(attribution, "output"))
        records = _load_attribution(attribution_path)
        if len(records) != attribution.get("records"):
            raise BoundedPilotCompilerPopulationError(
                "pilot attribution coverage differs"
            )
        for identity, record in records.items():
            if identity in attribution_by_identity:
                raise BoundedPilotCompilerPopulationError(
                    "pilot attribution identity repeats across sources"
                )
            attribution_by_identity[identity] = (source_id, record)
        pilots[source_id] = {
            "root": root,
            "receipt": receipt,
            "binding": {
                "root_name": root.name,
                "source_id": source_id,
                "source_type": SOURCE_TYPES[source_id],
                "receipt_file_sha256": sha256_file(receipt_path),
                "receipt_sha256": receipt["receipt_sha256"],
                "attribution_manifest": {
                    "path": attribution_path.name,
                    "rows": len(records),
                    "bytes": attribution_path.stat().st_size,
                    "sha256": sha256_file(attribution_path),
                },
            },
        }
    return pilots, attribution_by_identity


def _cross_source_documents(
    root: Path, pilots: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt_path = root / "receipt.json"
    receipt = _load_receipt(receipt_path)
    duplicate_filter = receipt.get("duplicate_filter")
    bindings = receipt.get("pilot_bindings")
    if (
        receipt.get("schema") != CROSS_SOURCE_SCHEMA
        or receipt.get("status") != "complete_nontraining_cross_source_sample"
        or receipt.get("bounded_cross_source_pilot_sample_complete") is not True
        or receipt.get("full_pilot_population_cross_source_deduplication_complete")
        is not True
        or receipt.get("training_ready") is not False
        or not isinstance(duplicate_filter, dict)
        or not isinstance(bindings, list)
    ):
        raise BoundedPilotCompilerPopulationError("cross-source receipt differs")
    expected = {
        (source_id, row["receipt"]["receipt_sha256"])
        for source_id, row in pilots.items()
    }
    observed = {
        (row.get("source_id"), row.get("receipt_sha256"))
        for row in bindings
        if isinstance(row, dict)
    }
    if observed != expected:
        raise BoundedPilotCompilerPopulationError("cross-source pilot bindings differ")
    output_path = _bound_file(root, _descriptor(duplicate_filter, "output"))
    documents = []
    identities = set()
    with output_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                document = normalize_document(json.loads(line))
            except Exception as error:
                raise BoundedPilotCompilerPopulationError(
                    f"cross-source document {line_number} differs"
                ) from error
            identity = document["identity_sha256"]
            if identity in identities:
                raise BoundedPilotCompilerPopulationError(
                    "cross-source document identity repeats"
                )
            identities.add(identity)
            documents.append(document)
    if len(documents) != duplicate_filter.get("output_documents"):
        raise BoundedPilotCompilerPopulationError(
            "cross-source output coverage differs"
        )
    return documents, {
        "root_name": root.name,
        "receipt_file_sha256": sha256_file(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "input_documents": duplicate_filter.get("input_documents"),
        "output_documents": duplicate_filter.get("output_documents"),
        "documents_dropped": duplicate_filter.get("documents_dropped"),
        "duplicate_groups": duplicate_filter.get("duplicate_groups"),
        "cross_source_duplicate_groups": duplicate_filter.get(
            "cross_source_duplicate_groups"
        ),
        "output": {
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
    }


def _candidate_and_lineage(
    document: dict[str, Any],
    source_id: str,
    attribution: dict[str, Any],
    pilot_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = attribution["source"]
    rights = attribution["rights_declaration"]
    document_source = document["source"]
    if (
        document_source["dataset"] != source["dataset"]
        or document_source["row_id"] != attribution["row_id"]
        or document_source["license"] != rights["canonical_license"]
        or document_source["domain"] != source["domain"]
    ):
        raise BoundedPilotCompilerPopulationError(
            "retained document and attribution differ"
        )
    provenance = {
        "pilot_receipt_sha256": pilot_receipt_sha256,
        "pilot_source_id": source_id,
        "retained_document_identity_sha256": document["identity_sha256"],
        "decontamination_evidence_sha256": document["verification"]["evidence_sha256"],
        "attribution_record_sha256": attribution["record_sha256"],
        "source": source,
        "row_id": attribution["row_id"],
    }
    text_sha256 = hashlib.sha256(document["text"].encode()).hexdigest()
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "text": document["text"],
        "source": {
            "dataset": source["dataset"],
            "revision": source["revision"],
            "row_id": attribution["row_id"],
            "license": rights["canonical_license"],
            "source_type": SOURCE_TYPES[source_id],
        },
        "source_content_sha256": text_sha256,
        "provenance_sha256": canonical_sha256(provenance),
    }
    candidate["candidate_identity_sha256"] = canonical_sha256(candidate)
    candidate = normalize_candidate(candidate)
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "source_content_sha256": text_sha256,
        "source_text_bytes": len(document["text"].encode()),
        "source_id": source_id,
        "source_type": SOURCE_TYPES[source_id],
        "retained_document_identity_sha256": document["identity_sha256"],
        "pilot_receipt_sha256": pilot_receipt_sha256,
        "attribution_record_sha256": attribution["record_sha256"],
        "source": source,
        "row_id": attribution["row_id"],
        "rights_declaration": rights,
        "raw_source_is_training_ready": False,
    }
    lineage["lineage_sha256"] = canonical_sha256(lineage)
    return candidate, lineage


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )


def build_population(
    pilot_roots: list[Path], cross_source_root: Path, output_root: Path
) -> dict[str, Any]:
    """Join filtered documents to exact attribution and seal compiler inputs."""

    if output_root.exists() or output_root.is_symlink():
        raise BoundedPilotCompilerPopulationError("compiler output already exists")
    pilots, attribution_by_identity = _pilot_bindings(pilot_roots)
    documents, cross_source = _cross_source_documents(cross_source_root, pilots)
    candidates = []
    lineage = []
    for document in documents:
        identity = document["identity_sha256"]
        binding = attribution_by_identity.get(identity)
        if binding is None:
            raise BoundedPilotCompilerPopulationError(
                "cross-source survivor lacks attribution"
            )
        source_id, attribution = binding
        candidate, source_lineage = _candidate_and_lineage(
            document,
            source_id,
            attribution,
            pilots[source_id]["receipt"]["receipt_sha256"],
        )
        candidates.append(candidate)
        lineage.append(source_lineage)
    candidate_identities = [row["candidate_identity_sha256"] for row in candidates]
    source_hashes = [row["source_content_sha256"] for row in candidates]
    if (
        not candidates
        or len(candidate_identities) != len(set(candidate_identities))
        or len(source_hashes) != len(set(source_hashes))
    ):
        raise BoundedPilotCompilerPopulationError(
            "compiler population identities differ"
        )
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        candidate_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        _write_jsonl(candidate_path, candidates)
        _write_jsonl(lineage_path, lineage)
        by_source = Counter(row["source_id"] for row in lineage)
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_compiler_population",
            "method": (
                "replay_full_bounded_cross_source_survivors_then_exact_"
                "attribution_join_v1"
            ),
            "pilot_bindings": [
                pilots[source_id]["binding"] for source_id in sorted(pilots)
            ],
            "cross_source_filter": cross_source,
            "population": {
                "path": candidate_path.name,
                "rows": len(candidates),
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
                "ordered_identities_sha256": canonical_sha256(candidate_identities),
            },
            "lineage": {
                "path": lineage_path.name,
                "rows": len(lineage),
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
                "ordered_rows_sha256": canonical_sha256(lineage),
                "source_text_persisted": False,
            },
            "by_source": dict(sorted(by_source.items())),
            "exact_attribution_coverage": True,
            "full_bounded_cross_source_survivor_coverage": True,
            "compiler_judgments_complete": False,
            "representation_verification_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(temporary / "receipt.json", payload)
        os.replace(temporary, output_root)
        return payload
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, action="append", required=True)
    parser.add_argument("--cross-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_population(args.pilot_root, args.cross_source_root, args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
