"""Apply unanimous structured subject review to a revised Sai syllabus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import _read_regular_bytes
from sai.data.curriculum import PHASES
from sai.data.foundational_syllabus import (
    _PHASE_MINIMUMS,
)
from sai.data.foundational_syllabus import (
    _prepare as _prepare_syllabus,
)
from sai.data.foundational_syllabus_review_compare import _prepare_comparison
from sai.data.prerequisite import (
    CONCEPT_LIST_SCHEMA,
    TAXONOMY_SCHEMA,
    validate_taxonomy_payload,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-foundational-syllabus-reviewed-candidate-receipt-v1"
SUPPORTING_SCHEMA = "sai-foundational-syllabus-supporting-edges-v1"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class FoundationalSyllabusReviewApplyError(RuntimeError):
    """Consensus, revised hard graph, supporting edges, or output differs."""


def _earliest_phase(row: dict[str, Any]) -> str:
    return next(phase for phase in PHASES if row["minimum_phase_documents"][phase] > 0)


def _stage(path: Path, encoded: bytes) -> Path:
    stage = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with stage.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return stage


def apply_reviews(
    *,
    base_concepts: Path,
    additions: Path,
    review_a: Path,
    expected_review_a_sha256: str,
    review_b: Path,
    expected_review_b_sha256: str,
    comparison: Path,
    expected_comparison_sha256: str,
    concept_output: Path,
    supporting_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    if (
        not _HEX64.fullmatch(expected_comparison_sha256)
        or sha256_file(comparison) != expected_comparison_sha256
    ):
        raise FoundationalSyllabusReviewApplyError("comparison file hash differs")
    try:
        comparison_payload = _prepare_comparison(
            base_concepts=base_concepts,
            additions=additions,
            review_a=review_a,
            expected_review_a_sha256=expected_review_a_sha256,
            review_b=review_b,
            expected_review_b_sha256=expected_review_b_sha256,
        )
        comparison_encoded = _read_regular_bytes(comparison, maximum_bytes=16 << 20)
        if comparison_encoded != (
            json.dumps(comparison_payload, sort_keys=True, indent=2).encode() + b"\n"
        ):
            raise ValueError
        composition_receipt, concept_encoded, _ = _prepare_syllabus(
            base_concepts=base_concepts, additions=additions
        )
        source_payload = json.loads(concept_encoded)
    except Exception as error:
        raise FoundationalSyllabusReviewApplyError(
            "review consensus replay differs"
        ) from error
    if (
        comparison_payload["status"]
        != "complete_consensus_requires_final_syllabus_application"
        or comparison_payload["summary"]["unresolved_concepts"] != 0
    ):
        raise FoundationalSyllabusReviewApplyError("review consensus is unresolved")
    source_by_identity = {row["concept_id"]: row for row in source_payload["concepts"]}
    comparison_by_identity = {
        row["concept_id"]: row for row in comparison_payload["concept_comparisons"]
    }
    if set(source_by_identity) != set(comparison_by_identity):
        raise FoundationalSyllabusReviewApplyError("consensus population differs")
    phase_by_identity: dict[str, str] = {}
    for concept_id in source_by_identity:
        consensus = comparison_by_identity[concept_id]["consensus"]
        if (
            not isinstance(consensus, dict)
            or consensus["concept_verdict"] == "reject"
            or consensus["granularity"] != "appropriate"
        ):
            raise FoundationalSyllabusReviewApplyError(
                "consensus requires manual concept redesign"
            )
        phase_by_identity[concept_id] = consensus["proposed_earliest_phase"]

    concepts = []
    supporting_rows = []
    hard_edges = 0
    supporting_edges = 0
    removed_edges = 0
    added_hard_edges = 0
    added_supporting_edges = 0
    renamed_concepts = 0
    rephased_concepts = 0
    for concept_id, source in source_by_identity.items():
        consensus = comparison_by_identity[concept_id]["consensus"]
        existing = consensus["edge_classifications"]
        missing = consensus["missing_prerequisites"]
        hard = sorted(
            [identity for identity, value in existing.items() if value == "hard"]
            + [identity for identity, value in missing.items() if value == "hard"]
        )
        supporting = sorted(
            [identity for identity, value in existing.items() if value == "supporting"]
            + [identity for identity, value in missing.items() if value == "supporting"]
        )
        if len(set(hard + supporting)) != len(hard) + len(supporting):
            raise FoundationalSyllabusReviewApplyError("consensus edge differs")
        phase_index = PHASES.index(phase_by_identity[concept_id])
        if any(
            prerequisite not in source_by_identity
            or PHASES.index(phase_by_identity[prerequisite]) > phase_index
            for prerequisite in hard
        ):
            raise FoundationalSyllabusReviewApplyError(
                "reviewed hard prerequisite phase differs"
            )
        name = consensus["proposed_name"] or source["name"]
        current_phase = _earliest_phase(source)
        proposed_phase = phase_by_identity[concept_id]
        concepts.append(
            {
                "concept_id": concept_id,
                "name": name,
                "domain": source["domain"],
                "prerequisites": hard,
                "minimum_prior_documents": 32 if hard else 0,
                "minimum_phase_documents": (
                    source["minimum_phase_documents"]
                    if proposed_phase == current_phase
                    else _PHASE_MINIMUMS[proposed_phase]
                ),
            }
        )
        supporting_rows.append(
            {
                "concept_id": concept_id,
                "supporting_concepts": supporting,
            }
        )
        hard_edges += len(hard)
        supporting_edges += len(supporting)
        removed_edges += sum(value == "remove" for value in existing.values())
        added_hard_edges += sum(value == "hard" for value in missing.values())
        added_supporting_edges += sum(
            value == "supporting" for value in missing.values()
        )
        renamed_concepts += int(name != source["name"])
        rephased_concepts += int(proposed_phase != current_phase)

    concept_payload = {
        "schema": CONCEPT_LIST_SCHEMA,
        "status": "candidate",
        "concepts": concepts,
    }
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
        raise FoundationalSyllabusReviewApplyError(
            "reviewed hard graph validation failed"
        ) from error
    supporting_payload = {
        "schema": SUPPORTING_SCHEMA,
        "status": "candidate_supporting_context_not_hard_prerequisites",
        "concept_list_sha256": canonical_sha256(concept_payload),
        "rows": supporting_rows,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    supporting_payload["receipt_sha256"] = canonical_sha256(supporting_payload)
    concept_bytes = (
        json.dumps(concept_payload, sort_keys=True, indent=2).encode() + b"\n"
    )
    supporting_bytes = (
        json.dumps(supporting_payload, sort_keys=True, indent=2).encode() + b"\n"
    )
    domain_counts = Counter(row["domain"] for row in concepts)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "subject_review_consensus_applied_requires_annotation_calibration",
        "composition_receipt_sha256": composition_receipt["receipt_sha256"],
        "comparison": {
            "file_sha256": expected_comparison_sha256,
            "receipt_sha256": comparison_payload["receipt_sha256"],
            "reviewer_ids": [
                row["reviewer_id"] for row in comparison_payload["reviews"]
            ],
        },
        "reviewed_concept_output": {
            "path": str(concept_output.resolve()),
            "bytes": len(concept_bytes),
            "sha256": hashlib.sha256(concept_bytes).hexdigest(),
            "concepts": len(concepts),
            "concepts_by_domain": dict(sorted(domain_counts.items())),
        },
        "supporting_edge_output": {
            "path": str(supporting_output.resolve()),
            "bytes": len(supporting_bytes),
            "sha256": hashlib.sha256(supporting_bytes).hexdigest(),
            "edges": supporting_edges,
        },
        "changes": {
            "hard_edges": hard_edges,
            "supporting_edges": supporting_edges,
            "removed_existing_edges": removed_edges,
            "added_hard_edges": added_hard_edges,
            "added_supporting_edges": added_supporting_edges,
            "renamed_concepts": renamed_concepts,
            "rephased_concepts": rephased_concepts,
        },
        "hard_graph_revalidated": True,
        "subject_review_consensus_applied": True,
        "subject_review_qualified": False,
        "limitations": [
            "annotation_policy_must_bind_reviewed_concept_bytes",
            "human_annotation_calibration_is_missing",
            "source_document_coverage_is_missing",
            "supporting_edges_are_context_not_exposure_gates",
            "reviewed_candidate_authorizes_no_training",
        ],
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_bytes = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    outputs = (concept_output, supporting_output, receipt_output)
    if (
        len({path.resolve() for path in outputs}) != len(outputs)
        or len({path.parent.resolve() for path in outputs}) != 1
    ):
        raise FoundationalSyllabusReviewApplyError("reviewed outputs differ")
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FoundationalSyllabusReviewApplyError("reviewed output already exists")
    stages: list[Path] = []
    try:
        concept_output.parent.mkdir(parents=True, exist_ok=True)
        for path, encoded in zip(
            outputs, (concept_bytes, supporting_bytes, receipt_bytes), strict=True
        ):
            stages.append(_stage(path, encoded))
        for stage, output in zip(stages, outputs, strict=True):
            os.replace(stage, output)
    except BaseException:
        for stage in stages:
            stage.unlink(missing_ok=True)
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-concepts", type=Path, required=True)
    parser.add_argument("--additions", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--expected-review-a-sha256", required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--expected-review-b-sha256", required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--expected-comparison-sha256", required=True)
    parser.add_argument("--concept-output", type=Path, required=True)
    parser.add_argument("--supporting-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    payload = apply_reviews(**vars(parser.parse_args(argv)))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "hard_edges": payload["changes"]["hard_edges"],
                "supporting_edges": payload["changes"]["supporting_edges"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
