"""Audit hard-prerequisite depth and centrality in a Sai syllabus candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import _read_regular_bytes, _write_create_only
from sai.data.foundational_syllabus import _prepare as _prepare_syllabus
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-foundational-syllabus-graph-audit-v1"
HIGH_DIRECT_DEPENDENT_COUNT = 8
DEEP_HARD_PREREQUISITE_DEPTH = 10


class FoundationalSyllabusAuditError(RuntimeError):
    """The composed syllabus or graph audit differs."""


def _prepare(*, base_concepts: Path, additions: Path) -> tuple[dict[str, Any], bytes]:
    try:
        composition_receipt, concept_encoded, _ = _prepare_syllabus(
            base_concepts=base_concepts, additions=additions
        )
        concept_payload = json.loads(concept_encoded)
    except Exception as error:
        raise FoundationalSyllabusAuditError("composed syllabus differs") from error
    rows = concept_payload["concepts"]
    by_identity = {row["concept_id"]: row for row in rows}
    downstream: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        for prerequisite in row["prerequisites"]:
            downstream[prerequisite].append(row["concept_id"])

    @cache
    def depth(concept_id: str) -> int:
        prerequisites = by_identity[concept_id]["prerequisites"]
        return (
            0 if not prerequisites else 1 + max(depth(value) for value in prerequisites)
        )

    @cache
    def ancestors(concept_id: str) -> frozenset[str]:
        result: set[str] = set()
        for prerequisite in by_identity[concept_id]["prerequisites"]:
            result.add(prerequisite)
            result.update(ancestors(prerequisite))
        return frozenset(result)

    concept_rows = []
    for concept_id in sorted(by_identity):
        row = by_identity[concept_id]
        cross_domain = sorted(
            prerequisite
            for prerequisite in row["prerequisites"]
            if by_identity[prerequisite]["domain"] != row["domain"]
        )
        concept_rows.append(
            {
                "concept_id": concept_id,
                "domain": row["domain"],
                "hard_prerequisite_depth": depth(concept_id),
                "transitive_hard_prerequisites": len(ancestors(concept_id)),
                "direct_hard_prerequisites": len(row["prerequisites"]),
                "direct_dependents": len(downstream[concept_id]),
                "cross_domain_hard_prerequisites": cross_domain,
                "risk_flags": sorted(
                    [
                        *(
                            ["deep_hard_prerequisite_chain"]
                            if depth(concept_id) >= DEEP_HARD_PREREQUISITE_DEPTH
                            else []
                        ),
                        *(
                            ["high_direct_dependent_centrality"]
                            if len(downstream[concept_id])
                            >= HIGH_DIRECT_DEPENDENT_COUNT
                            else []
                        ),
                        *(
                            ["cross_domain_hard_edge_requires_subject_review"]
                            if cross_domain
                            else []
                        ),
                    ]
                ),
            }
        )
    roots = sorted(row["concept_id"] for row in rows if not row["prerequisites"])
    leaves = sorted(
        row["concept_id"] for row in rows if not downstream[row["concept_id"]]
    )
    edge_count = sum(len(row["prerequisites"]) for row in rows)
    cross_domain_edges = sum(
        by_identity[prerequisite]["domain"] != row["domain"]
        for row in rows
        for prerequisite in row["prerequisites"]
    )
    depth_counts = Counter(depth(row["concept_id"]) for row in rows)
    flagged = [row for row in concept_rows if row["risk_flags"]]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "graph_risk_review_required",
        "composition": {
            "receipt_sha256": composition_receipt["receipt_sha256"],
            "concept_sha256": hashlib.sha256(concept_encoded).hexdigest(),
            "concepts": len(rows),
        },
        "policy": {
            "all_declared_prerequisites_are_currently_hard": True,
            "high_direct_dependent_count": HIGH_DIRECT_DEPENDENT_COUNT,
            "deep_hard_prerequisite_depth": DEEP_HARD_PREREQUISITE_DEPTH,
            "cross_domain_hard_edges_require_subject_review": True,
            "risk_flag_is_diagnostic_not_automatic_rejection": True,
        },
        "summary": {
            "roots": len(roots),
            "root_concept_ids": roots,
            "leaves": len(leaves),
            "leaf_concept_ids": leaves,
            "hard_edges": edge_count,
            "cross_domain_hard_edges": cross_domain_edges,
            "maximum_hard_prerequisite_depth": max(depth_counts),
            "concepts_by_hard_prerequisite_depth": {
                str(value): depth_counts[value] for value in sorted(depth_counts)
            },
            "flagged_concepts": len(flagged),
        },
        "concept_graph_rows": concept_rows,
        "review_requirements": [
            "classify_each_flagged_edge_as_hard_supporting_or_remove",
            "review_two_root_bottleneck_for_false_serialization",
            "review_every_depth_at_least_ten_chain",
            "review_every_prerequisite_with_at_least_eight_direct_dependents",
            "recompute_graph_after_adjudication_before_document_annotation",
        ],
        "progression_qualified": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    encoded = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    return payload, encoded


def build(*, base_concepts: Path, additions: Path, output: Path) -> dict[str, Any]:
    payload, encoded = _prepare(base_concepts=base_concepts, additions=additions)
    try:
        _write_create_only(output, encoded)
    except Exception as error:
        raise FoundationalSyllabusAuditError("graph audit output differs") from error
    return payload


def validate(*, base_concepts: Path, additions: Path, output: Path) -> dict[str, Any]:
    payload, encoded = _prepare(base_concepts=base_concepts, additions=additions)
    try:
        actual = _read_regular_bytes(output, maximum_bytes=4 << 20)
    except OSError as error:
        raise FoundationalSyllabusAuditError("graph audit output differs") from error
    if actual != encoded:
        raise FoundationalSyllabusAuditError("graph audit output differs")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--base-concepts", type=Path, required=True)
    parser.add_argument("--additions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    payload = (build if command == "build" else validate)(**args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "roots": payload["summary"]["roots"],
                "hard_edges": payload["summary"]["hard_edges"],
                "maximum_depth": payload["summary"]["maximum_hard_prerequisite_depth"],
                "flagged_concepts": payload["summary"]["flagged_concepts"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
