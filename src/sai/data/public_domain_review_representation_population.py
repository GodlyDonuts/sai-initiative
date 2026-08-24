"""Freeze source-grounded representation work from clean PDR candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_compiler_aggregate import load_population
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.data_compiler_labeling import REPRESENTATIONS
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.public_domain_review_scope_audit import SOURCE_ID
from sai.data.public_domain_review_scoped_candidates import CANDIDATE_SCHEMA
from sai.data.public_domain_review_work_lanes import (
    LANE_RECORD_SCHEMA,
)
from sai.data.public_domain_review_work_lanes import (
    SCHEMA as WORK_LANES_SCHEMA,
)
from sai.data.reservoir_audit_aggregate import _validate_compiler_receipt
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-public-domain-review-representation-population-v1"
RECORD_SCHEMA = "sai-public-domain-review-representation-candidate-v1"
MAXIMUM_REQUESTED_REPRESENTATIONS = 6
MINIMUM_SOURCE_TEXT_BYTES = 200
MAXIMUM_SOURCE_TEXT_BYTES = 262_144
DERIVATIVE_PRIORITY = (
    "conceptual_summary",
    "concise_reference",
    "beginner_explanation",
    "undergraduate_explanation",
    "prerequisite_map",
    "faq",
    "misconception_corrections",
    "comparative_analysis",
    "cross_domain_problems",
    "worked_examples",
    "graduate_explanation",
    "executable_exercises",
)
_DERIVATIVE_TYPES = frozenset(DERIVATIVE_PRIORITY)
if not _DERIVATIVE_TYPES < set(REPRESENTATIONS):  # pragma: no cover - import guard
    raise RuntimeError("representation priority differs from compiler contract")


class PublicDomainReviewRepresentationPopulationError(RuntimeError):
    """A clean candidate, work lane, or compiler binding differs."""


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PublicDomainReviewRepresentationPopulationError(f"{label} is unsafe")
    rows = []
    try:
        with path.open() as handle:
            for _line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        raise PublicDomainReviewRepresentationPopulationError(
            f"{label} row differs"
        ) from error
    if not rows:
        raise PublicDomainReviewRepresentationPopulationError(f"{label} is empty")
    return rows


def load_work_lanes(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the exact source-text-free PDR work lanes."""

    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("work_lanes")
    if (
        receipt.get("schema") != WORK_LANES_SCHEMA
        or receipt.get("status") != "complete_nontraining_pdr_work_lanes"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("compiler_route_is_verified_admission") is not False
        or receipt.get("independent_representation_verification_complete") is not False
        or receipt.get("legal_clearance_established") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
        or descriptor.get("source_text_persisted") is not False
    ):
        raise PublicDomainReviewRepresentationPopulationError(
            "work-lane receipt differs"
        )
    path = _bound_file(root, descriptor)
    rows = _load_jsonl(path, "work lanes")
    identities = set()
    for row in rows:
        identity = row.get("original_candidate_identity_sha256")
        row_unsigned = {
            key: value for key, value in row.items() if key != "record_sha256"
        }
        if (
            row.get("schema") != LANE_RECORD_SCHEMA
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or row.get("record_sha256") != canonical_sha256(row_unsigned)
            or row.get("compiler_route_is_verified_admission") is not False
            or row.get("representation_verified") is not False
            or row.get("legal_clearance_established") is not False
            or row.get("training_ready") is not False
        ):
            raise PublicDomainReviewRepresentationPopulationError(
                "work-lane row differs"
            )
        identities.add(identity)
    if len(rows) != descriptor.get("rows") or descriptor.get(
        "ordered_records_sha256"
    ) != canonical_sha256([row["record_sha256"] for row in rows]):
        raise PublicDomainReviewRepresentationPopulationError(
            "work-lane coverage differs"
        )
    return rows, receipt


def _load_priority_candidates(
    root: Path, work_receipt: dict[str, Any]
) -> list[dict[str, Any]]:
    descriptor = work_receipt.get("representation_priority_candidates")
    if not isinstance(descriptor, dict):
        raise PublicDomainReviewRepresentationPopulationError(
            "priority descriptor differs"
        )
    path = _bound_file(root, descriptor)
    rows = _load_jsonl(path, "priority candidates")
    identities = set()
    text_bytes = 0
    for row in rows:
        identity = row.get("original_candidate_identity_sha256")
        text = row.get("text")
        row_unsigned = {
            key: value for key, value in row.items() if key != "record_sha256"
        }
        if (
            row.get("schema") != CANDIDATE_SCHEMA
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or not isinstance(text, str)
            or not text
            or row.get("record_sha256") != canonical_sha256(row_unsigned)
            or row.get("content_quality_verified") is not False
            or row.get("legal_clearance_established") is not False
            or row.get("training_ready") is not False
        ):
            raise PublicDomainReviewRepresentationPopulationError(
                "priority candidate differs"
            )
        identities.add(identity)
        text_bytes += len(text.encode())
    if (
        len(rows) != descriptor.get("rows")
        or text_bytes != descriptor.get("text_bytes")
        or descriptor.get("ordered_records_sha256")
        != canonical_sha256([row["record_sha256"] for row in rows])
    ):
        raise PublicDomainReviewRepresentationPopulationError(
            "priority candidate coverage differs"
        )
    return rows


def select_derivative_representations(values: Any) -> list[str]:
    """Select a bounded deterministic subset of compiler-requested derivatives."""

    if (
        not isinstance(values, list)
        or not values
        or len(values) != len(set(values))
        or any(value not in REPRESENTATIONS for value in values)
    ):
        raise PublicDomainReviewRepresentationPopulationError(
            "recommended representations differ"
        )
    selected = [value for value in DERIVATIVE_PRIORITY if value in values]
    return selected[:MAXIMUM_REQUESTED_REPRESENTATIONS]


def build_candidate(
    source: dict[str, Any], lane: dict[str, Any], compiler: dict[str, Any]
) -> dict[str, Any] | None:
    """Bind one clean source text to its compiler-derived representation plan."""

    judgment = compiler.get("judgment")
    text = source.get("text")
    original_identity = source.get("original_candidate_identity_sha256")
    if (
        not isinstance(judgment, dict)
        or not isinstance(text, str)
        or not text
        or hashlib.sha256(text.encode()).hexdigest() != source.get("scoped_text_sha256")
        or lane.get("original_candidate_identity_sha256") != original_identity
        or lane.get("representation_priority_candidate") is not True
        or lane.get("compiler_candidate_identity_sha256")
        != compiler.get("candidate_identity_sha256")
        or lane.get("compiler_receipt_sha256") != compiler.get("receipt_sha256")
        or lane.get("compiler_judgment_sha256") != judgment.get("judgment_sha256")
    ):
        raise PublicDomainReviewRepresentationPopulationError(
            "representation candidate binding differs"
        )
    text_bytes = len(text.encode())
    if not MINIMUM_SOURCE_TEXT_BYTES <= text_bytes <= MAXIMUM_SOURCE_TEXT_BYTES:
        return None
    requested = select_derivative_representations(
        judgment.get("recommended_representations")
    )
    if not requested:
        return None
    source_binding = source.get("source")
    if (
        not isinstance(source_binding, dict)
        or source_binding.get("dataset") != "common-pile/public_domain_review_filtered"
        or source_binding.get("license") != "CC-BY-SA-4.0"
        or source.get("attribution_required") is not True
        or source.get("share_alike_required") is not True
    ):
        raise PublicDomainReviewRepresentationPopulationError(
            "representation source rights differ"
        )
    row = {
        "schema": RECORD_SCHEMA,
        "text": text,
        "source_text_sha256": source["scoped_text_sha256"],
        "source_record_sha256": source["record_sha256"],
        "original_candidate_identity_sha256": original_identity,
        "source": {
            "dataset": source_binding["dataset"],
            "row_id": source_binding["row_id"],
            "source_url": source["source_url"],
            "source_type": source["source_type"],
            "license": source_binding["license"],
            "attribution_required": True,
            "share_alike_required": True,
        },
        "compiler": {
            "candidate_identity_sha256": compiler["candidate_identity_sha256"],
            "receipt_sha256": compiler["receipt_sha256"],
            "judgment_sha256": judgment["judgment_sha256"],
            "work_record_sha256": lane["work_record_sha256"],
            "content_route": lane["content_route"],
            "rights_route": lane["rights_route"],
            "verdict": judgment["verdict"],
            "preservation_policy": judgment["preservation_policy"],
            "requested_representations": requested,
            "domains": judgment["domains"],
            "subdomains": judgment["subdomains"],
            "concepts_taught": judgment["concepts_taught"],
            "prerequisites_assumed": judgment["prerequisites_assumed"],
            "cross_domain_bridges": judgment["cross_domain_bridges"],
            "difficulty": judgment["difficulty"],
            "curriculum_phase": judgment["curriculum_phase"],
        },
        "compiler_route_is_verified_admission": False,
        "representation_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def build_population(
    work_lanes_root: Path,
    compiler_population_root: Path,
    judgments_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Seal all PDR representation-priority rows with derivative work."""

    if output_root.exists() or output_root.is_symlink():
        raise PublicDomainReviewRepresentationPopulationError(
            "representation population output differs"
        )
    lanes, work_receipt = load_work_lanes(work_lanes_root)
    priority = _load_priority_candidates(work_lanes_root, work_receipt)
    generic_candidates, lineage, compiler_population = load_population(
        compiler_population_root
    )
    generic_by_identity = {
        row["candidate_identity_sha256"]: row for row in generic_candidates
    }
    source_ids_by_compiler_identity = {
        candidate["candidate_identity_sha256"]: source["source_id"]
        for candidate, source in zip(generic_candidates, lineage, strict=True)
    }
    lanes_by_original = {
        row["original_candidate_identity_sha256"]: row for row in lanes
    }
    if len(lanes_by_original) != len(lanes):
        raise PublicDomainReviewRepresentationPopulationError(
            "representation lane identities differ"
        )
    output_rows = []
    skipped_without_derivative = 0
    skipped_outside_text_envelope = 0
    for source in priority:
        original_identity = source["original_candidate_identity_sha256"]
        lane = lanes_by_original.get(original_identity)
        if lane is None:
            raise PublicDomainReviewRepresentationPopulationError(
                "priority lane identity differs"
            )
        compiler_identity = lane["compiler_candidate_identity_sha256"]
        generic = generic_by_identity.get(compiler_identity)
        receipt_path = judgments_root / f"{compiler_identity}.compiler.json"
        if (
            generic is None
            or source_ids_by_compiler_identity.get(compiler_identity) != SOURCE_ID
            or not receipt_path.is_file()
        ):
            raise PublicDomainReviewRepresentationPopulationError(
                "compiler candidate identity differs"
            )
        compiler = _validate_compiler_receipt(_load_receipt(receipt_path), generic)
        row = build_candidate(source, lane, compiler)
        if row is None:
            text_bytes = len(source["text"].encode())
            if not MINIMUM_SOURCE_TEXT_BYTES <= text_bytes <= MAXIMUM_SOURCE_TEXT_BYTES:
                skipped_outside_text_envelope += 1
            else:
                skipped_without_derivative += 1
        else:
            output_rows.append(row)
    identities = [row["candidate_identity_sha256"] for row in output_rows]
    if not output_rows or len(identities) != len(set(identities)):
        raise PublicDomainReviewRepresentationPopulationError(
            "representation population identities differ"
        )
    output_root.mkdir(parents=True)
    try:
        candidates_path = output_root / "candidates.jsonl"
        _atomic_jsonl(candidates_path, output_rows)
        types = Counter(
            representation
            for row in output_rows
            for representation in row["compiler"]["requested_representations"]
        )
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_representation_population",
            "work_lanes": {
                "root_name": work_lanes_root.name,
                "receipt_file_sha256": sha256_file(work_lanes_root / "receipt.json"),
                "receipt_sha256": work_receipt["receipt_sha256"],
                "priority_rows": len(priority),
            },
            "compiler_population": {
                "root_name": compiler_population_root.name,
                "receipt_sha256": compiler_population["receipt_sha256"],
            },
            "candidates": {
                "path": candidates_path.name,
                "rows": len(output_rows),
                "bytes": candidates_path.stat().st_size,
                "sha256": sha256_file(candidates_path),
                "ordered_identities_sha256": canonical_sha256(identities),
                "source_text_bytes": sum(
                    len(row["text"].encode()) for row in output_rows
                ),
            },
            "source_text_byte_envelope": {
                "minimum": MINIMUM_SOURCE_TEXT_BYTES,
                "maximum": MAXIMUM_SOURCE_TEXT_BYTES,
                "decision": "exclude_before_generation",
            },
            "skipped_outside_source_text_byte_envelope": (
                skipped_outside_text_envelope
            ),
            "skipped_without_derivative_representation": skipped_without_derivative,
            "requested_representation_counts": dict(sorted(types.items())),
            "source_text_persisted": True,
            "source_license": "CC-BY-SA-4.0",
            "attribution_required": True,
            "share_alike_required": True,
            "compiler_route_is_verified_admission": False,
            "generated_representations_complete": False,
            "independent_representation_verification_complete": False,
            "legal_clearance_established": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-lanes-root", type=Path, required=True)
    parser.add_argument("--compiler-population-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_population(
        args.work_lanes_root,
        args.compiler_population_root,
        args.judgments_root,
        args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
