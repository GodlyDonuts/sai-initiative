"""Compose and validate the expanded Sai foundational syllabus candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.curriculum import PHASES
from sai.data.prerequisite import (
    CONCEPT_LIST_SCHEMA,
    TAXONOMY_SCHEMA,
    validate_taxonomy_payload,
)
from sai.data.token_stream import ALLOWED_DOMAINS, canonical_sha256

SCHEMA = "sai-foundational-syllabus-candidate-receipt-v1"
ADDITIONS_SCHEMA = "sai-foundational-syllabus-additions-v1"
_ADDITION_KEYS = {
    "concept_id",
    "name",
    "domain",
    "prerequisites",
    "earliest_phase",
}
_PHASE_MINIMUMS = {
    "grounding": {
        "grounding": 16,
        "integration": 8,
        "reasoning": 8,
        "specialization": 8,
    },
    "integration": {
        "grounding": 0,
        "integration": 16,
        "reasoning": 8,
        "specialization": 8,
    },
    "reasoning": {
        "grounding": 0,
        "integration": 0,
        "reasoning": 16,
        "specialization": 16,
    },
    "specialization": {
        "grounding": 0,
        "integration": 0,
        "reasoning": 0,
        "specialization": 16,
    },
}


class FoundationalSyllabusError(RuntimeError):
    """The base seed, additions, composed syllabus, or receipt differs."""


def _json(path: Path, *, maximum_bytes: int) -> tuple[dict[str, Any], bytes]:
    encoded = _read_regular_bytes(path, maximum_bytes=maximum_bytes)
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FoundationalSyllabusError("syllabus JSON differs") from error
    if not isinstance(payload, dict):
        raise FoundationalSyllabusError("syllabus JSON differs")
    return payload, encoded


def _earliest_phase(row: dict[str, Any]) -> str:
    for phase in PHASES:
        if row["minimum_phase_documents"][phase] > 0:
            return phase
    raise FoundationalSyllabusError("base concept phase differs")


def _prepare(
    *, base_concepts: Path, additions: Path
) -> tuple[dict[str, Any], bytes, bytes]:
    base, base_encoded = _json(base_concepts, maximum_bytes=2 << 20)
    added, additions_encoded = _json(additions, maximum_bytes=2 << 20)
    if (
        set(base) != {"schema", "status", "concepts"}
        or base["schema"] != CONCEPT_LIST_SCHEMA
        or base["status"] != "candidate"
        or not isinstance(base["concepts"], list)
        or not base["concepts"]
        or set(added) != {"schema", "status", "concepts"}
        or added["schema"] != ADDITIONS_SCHEMA
        or added["status"] != "candidate"
        or not isinstance(added["concepts"], list)
        or not added["concepts"]
    ):
        raise FoundationalSyllabusError("syllabus source differs")
    concepts = list(base["concepts"])
    known = {row["concept_id"] for row in concepts}
    if len(known) != len(concepts):
        raise FoundationalSyllabusError("base concept identity differs")
    phase_by_concept = {row["concept_id"]: _earliest_phase(row) for row in concepts}
    added_ids: set[str] = set()
    for raw in added["concepts"]:
        if (
            not isinstance(raw, dict)
            or set(raw) != _ADDITION_KEYS
            or not isinstance(raw["concept_id"], str)
            or raw["concept_id"] in known
            or raw["concept_id"] in added_ids
            or not isinstance(raw["name"], str)
            or not raw["name"].strip()
            or raw["domain"] not in ALLOWED_DOMAINS
            or not isinstance(raw["prerequisites"], list)
            or not raw["prerequisites"]
            or len(raw["prerequisites"]) != len(set(raw["prerequisites"]))
            or raw["earliest_phase"] not in PHASES
        ):
            raise FoundationalSyllabusError("syllabus addition differs")
        added_ids.add(raw["concept_id"])
        phase_by_concept[raw["concept_id"]] = raw["earliest_phase"]
    all_known = known | added_ids
    for raw in added["concepts"]:
        if any(value not in all_known for value in raw["prerequisites"]):
            raise FoundationalSyllabusError("syllabus addition differs")
        phase_index = PHASES.index(raw["earliest_phase"])
        if any(
            PHASES.index(phase_by_concept[value]) > phase_index
            for value in raw["prerequisites"]
        ):
            raise FoundationalSyllabusError("syllabus prerequisite phase differs")
        row = {
            "concept_id": raw["concept_id"],
            "name": raw["name"],
            "domain": raw["domain"],
            "prerequisites": raw["prerequisites"],
            "minimum_prior_documents": 32,
            "minimum_phase_documents": _PHASE_MINIMUMS[raw["earliest_phase"]],
        }
        concepts.append(row)
    composed = {
        "schema": CONCEPT_LIST_SCHEMA,
        "status": "candidate",
        "concepts": concepts,
    }
    # Reuse the production taxonomy validator to prove exact fields, all domains,
    # prerequisite existence, thresholds, phase geometry, and graph acyclicity.
    validator_payload: dict[str, Any] = {
        "schema": TAXONOMY_SCHEMA,
        "status": "prospective",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "minimum_annotation_confidence_ppm": 800_000,
        "minimum_evidence_codepoints_per_positive_label": 16,
        "maximum_new_concepts_per_document": 8,
        "annotation_method": {
            "method": "human",
            "annotator_identity_sha256": "1" * 64,
            "policy_sha256": "2" * 64,
            "audit_sample_receipt_sha256": "3" * 64,
        },
        "concepts": concepts,
    }
    validator_payload["receipt_sha256"] = canonical_sha256(validator_payload)
    try:
        validate_taxonomy_payload(validator_payload)
    except Exception as error:
        raise FoundationalSyllabusError(
            "composed syllabus validation failed"
        ) from error
    encoded = json.dumps(composed, sort_keys=True, indent=2).encode() + b"\n"
    domain_counts = Counter(row["domain"] for row in concepts)
    earliest_counts = Counter(phase_by_concept.values())
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "expanded_candidate_requires_human_subject_review",
        "base_concept_list": {
            "bytes": len(base_encoded),
            "sha256": hashlib.sha256(base_encoded).hexdigest(),
            "concepts": len(base["concepts"]),
        },
        "additions": {
            "bytes": len(additions_encoded),
            "sha256": hashlib.sha256(additions_encoded).hexdigest(),
            "concepts": len(added["concepts"]),
        },
        "composed": {
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "concepts": len(concepts),
            "concepts_by_domain": dict(sorted(domain_counts.items())),
            "concepts_by_earliest_phase": {
                phase: earliest_counts[phase] for phase in PHASES
            },
        },
        "policy": {
            "minimum_prior_documents_for_dependent_additions": 32,
            "phase_minimums_by_earliest_phase": _PHASE_MINIMUMS,
            "dependency_may_not_start_after_dependent": True,
            "production_taxonomy_validator_replayed": True,
        },
        "limitations": [
            "concept_names_and_edges_require_subject_matter_review",
            "concept_inventory_is_not_complete_world_knowledge",
            "annotation_policy_must_be_rebound_to_composed_bytes",
            "human_calibration_and_source_coverage_are_missing",
            "candidate_authorizes_no_data_reordering_or_training",
        ],
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return (
        receipt,
        encoded,
        json.dumps(receipt, sort_keys=True, indent=2).encode() + b"\n",
    )


def build(
    *,
    base_concepts: Path,
    additions: Path,
    concept_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    if concept_output.resolve() == receipt_output.resolve():
        raise FoundationalSyllabusError("syllabus outputs differ")
    receipt, encoded, receipt_encoded = _prepare(
        base_concepts=base_concepts, additions=additions
    )
    created = False
    try:
        _write_create_only(concept_output, encoded)
        created = True
        _write_create_only(receipt_output, receipt_encoded)
    except Exception as error:
        if created and not receipt_output.exists():
            concept_output.chmod(0o600)
            concept_output.unlink()
        raise FoundationalSyllabusError("syllabus output boundary differs") from error
    return receipt


def validate(
    *,
    base_concepts: Path,
    additions: Path,
    concept_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    receipt, encoded, receipt_encoded = _prepare(
        base_concepts=base_concepts, additions=additions
    )
    try:
        actual_concepts = _read_regular_bytes(concept_output, maximum_bytes=4 << 20)
        actual_receipt = _read_regular_bytes(receipt_output, maximum_bytes=1 << 20)
    except OSError as error:
        raise FoundationalSyllabusError("syllabus output differs") from error
    if actual_concepts != encoded or actual_receipt != receipt_encoded:
        raise FoundationalSyllabusError("syllabus output differs")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--base-concepts", type=Path, required=True)
    parser.add_argument("--additions", type=Path, required=True)
    parser.add_argument("--concept-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    payload = (build if command == "build" else validate)(**args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "concepts": payload["composed"]["concepts"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
