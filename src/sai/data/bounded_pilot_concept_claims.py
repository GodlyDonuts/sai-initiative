"""Compile text-free concept, prerequisite, and bridge claims from a pilot."""

from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_compiler_aggregate import load_population
from sai.data.bounded_pilot_work_queue import (
    RECORD_SCHEMA as WORK_RECORD_SCHEMA,
)
from sai.data.bounded_pilot_work_queue import (
    SCHEMA as WORK_QUEUE_SCHEMA,
)
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.nous_compiler_worker import COMPILER_REASONING_EFFORT
from sai.data.reservoir_audit_aggregate import _validate_compiler_receipt
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-bounded-pilot-concept-claims-v1"
CLAIM_SCHEMA = "sai-bounded-pilot-concept-claim-v1"
NODE_SCHEMA = "sai-bounded-pilot-concept-node-v1"
EDGE_SCHEMA = "sai-bounded-pilot-prerequisite-cooccurrence-v1"
BRIDGE_SCHEMA = "sai-bounded-pilot-cross-domain-bridge-claim-v1"


class BoundedPilotConceptClaimError(RuntimeError):
    """The compiler, work queue, or concept-claim custody differs."""


def normalize_label(value: Any) -> str:
    """Create a conservative comparison key without claiming semantic identity."""

    if not isinstance(value, str):
        raise BoundedPilotConceptClaimError("concept label differs")
    label = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if (
        not label
        or len(label) > 192
        or any(unicodedata.category(character) == "Cc" for character in label)
    ):
        raise BoundedPilotConceptClaimError("concept label differs")
    return label


def _normalized_labels(values: Any, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(values, list):
        raise BoundedPilotConceptClaimError("concept label list differs")
    labels = sorted({normalize_label(value) for value in values})
    if not labels and not allow_empty:
        raise BoundedPilotConceptClaimError("concept label list differs")
    return labels


def _load_work_queue(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("queue")
    if (
        receipt.get("schema") != WORK_QUEUE_SCHEMA
        or receipt.get("status") != "complete_text_free_nontraining_work_queue"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("exact_identity_coverage") is not True
        or receipt.get("content_and_rights_lanes_are_independent") is not True
        or receipt.get("source_text_persisted") is not False
        or receipt.get("representation_verification_complete") is not False
        or receipt.get("legal_clearance_established") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise BoundedPilotConceptClaimError("work queue receipt differs")
    path = _bound_file(root, descriptor)
    rows = []
    by_identity = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            identity = row.get("candidate_identity_sha256")
            row_unsigned = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            if (
                row.get("schema") != WORK_RECORD_SCHEMA
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in by_identity
                or row.get("record_sha256") != canonical_sha256(row_unsigned)
                or row.get("content_and_rights_lanes_are_independent") is not True
                or row.get("representation_verified") is not False
                or row.get("legal_clearance_established") is not False
                or row.get("training_ready") is not False
            ):
                raise BoundedPilotConceptClaimError("work queue row differs")
            rows.append(row)
            by_identity[identity] = row
    if len(rows) != descriptor.get("rows") or descriptor.get(
        "ordered_records_sha256"
    ) != canonical_sha256([row["record_sha256"] for row in rows]):
        raise BoundedPilotConceptClaimError("work queue coverage differs")
    return by_identity, receipt


def build_claim(
    source: dict[str, Any], receipt: dict[str, Any], work: dict[str, Any]
) -> dict[str, Any]:
    """Bind one model annotation to independent content and rights routes."""

    judgment = receipt.get("judgment")
    candidate_identity = receipt.get("candidate_identity_sha256")
    if (
        not isinstance(judgment, dict)
        or work.get("candidate_identity_sha256") != candidate_identity
        or work.get("retained_document_identity_sha256")
        != source.get("retained_document_identity_sha256")
        or work.get("source_id") != source.get("source_id")
        or work.get("compiler_receipt_sha256") != receipt.get("receipt_sha256")
        or work.get("compiler_judgment_sha256") != judgment.get("judgment_sha256")
    ):
        raise BoundedPilotConceptClaimError("concept claim binding differs")
    claim = {
        "schema": CLAIM_SCHEMA,
        "candidate_identity_sha256": candidate_identity,
        "retained_document_identity_sha256": source[
            "retained_document_identity_sha256"
        ],
        "source_id": source["source_id"],
        "compiler_receipt_sha256": receipt["receipt_sha256"],
        "compiler_judgment_sha256": judgment["judgment_sha256"],
        "work_record_sha256": work["record_sha256"],
        "content_route": work["content_route"],
        "rights_route": work["rights_route"],
        "expected_license_evidence_observed": work[
            "expected_license_evidence_observed"
        ],
        "curriculum_phase_claim": judgment["curriculum_phase"],
        "difficulty_claim": judgment["difficulty"],
        "prerequisite_burden_claim": judgment["prerequisite_burden"],
        "domains": _normalized_labels(judgment["domains"]),
        "subdomains": _normalized_labels(judgment["subdomains"], allow_empty=True),
        "epistemic_functions": _normalized_labels(judgment["epistemic_functions"]),
        "concepts_taught": _normalized_labels(
            judgment["concepts_taught"], allow_empty=True
        ),
        "prerequisites_assumed": _normalized_labels(
            judgment["prerequisites_assumed"], allow_empty=True
        ),
        "cross_domain_bridges": _normalized_labels(
            judgment["cross_domain_bridges"], allow_empty=True
        ),
        "pairwise_prerequisite_edges_are_document_level_cooccurrence": True,
        "model_annotations_independently_verified": False,
        "semantic_edges_verified": False,
        "curriculum_ready": False,
        "training_ready": False,
    }
    claim["claim_sha256"] = canonical_sha256(claim)
    return claim


def summarize_claims(
    claims: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create replayable node, co-occurrence-edge, and bridge claim indexes."""

    if not claims or len(
        {row.get("candidate_identity_sha256") for row in claims}
    ) != len(claims):
        raise BoundedPilotConceptClaimError("concept claim population differs")
    nodes: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "taught": set(),
            "assumed": set(),
            "sources": set(),
            "routes": Counter(),
        }
    )
    edges: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"identities": set(), "sources": set(), "routes": Counter()}
    )
    bridges: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"identities": set(), "sources": set(), "routes": Counter()}
    )
    for claim in claims:
        identity = claim["candidate_identity_sha256"]
        source = claim["source_id"]
        route = claim["content_route"]
        for concept in claim["concepts_taught"]:
            nodes[concept]["taught"].add(identity)
            nodes[concept]["sources"].add(source)
            nodes[concept]["routes"][route] += 1
        for prerequisite in claim["prerequisites_assumed"]:
            nodes[prerequisite]["assumed"].add(identity)
            nodes[prerequisite]["sources"].add(source)
            nodes[prerequisite]["routes"][route] += 1
        for prerequisite in claim["prerequisites_assumed"]:
            for concept in claim["concepts_taught"]:
                if prerequisite == concept:
                    continue
                edge = edges[(prerequisite, concept)]
                edge["identities"].add(identity)
                edge["sources"].add(source)
                edge["routes"][route] += 1
        for label in claim["cross_domain_bridges"]:
            bridge = bridges[label]
            bridge["identities"].add(identity)
            bridge["sources"].add(source)
            bridge["routes"][route] += 1
    node_rows = []
    for label, values in sorted(nodes.items()):
        row = {
            "schema": NODE_SCHEMA,
            "label": label,
            "taught_candidate_identity_sha256s": sorted(values["taught"]),
            "assumed_candidate_identity_sha256s": sorted(values["assumed"]),
            "source_ids": sorted(values["sources"]),
            "content_route_claim_counts": dict(sorted(values["routes"].items())),
            "semantic_identity_verified": False,
            "curriculum_ready": False,
        }
        row["record_sha256"] = canonical_sha256(row)
        node_rows.append(row)
    edge_rows = []
    for (prerequisite, concept), values in sorted(edges.items()):
        row = {
            "schema": EDGE_SCHEMA,
            "prerequisite_label": prerequisite,
            "concept_label": concept,
            "supporting_candidate_identity_sha256s": sorted(values["identities"]),
            "source_ids": sorted(values["sources"]),
            "content_route_claim_counts": dict(sorted(values["routes"].items())),
            "pairing_inferred_from_document_level_cooccurrence": True,
            "semantic_edge_verified": False,
            "curriculum_ready": False,
        }
        row["record_sha256"] = canonical_sha256(row)
        edge_rows.append(row)
    bridge_rows = []
    for label, values in sorted(bridges.items()):
        row = {
            "schema": BRIDGE_SCHEMA,
            "bridge_claim": label,
            "supporting_candidate_identity_sha256s": sorted(values["identities"]),
            "source_ids": sorted(values["sources"]),
            "content_route_claim_counts": dict(sorted(values["routes"].items())),
            "bridge_verified": False,
            "curriculum_ready": False,
        }
        row["record_sha256"] = canonical_sha256(row)
        bridge_rows.append(row)
    return node_rows, edge_rows, bridge_rows


def build_graph(
    population_root: Path,
    judgments_root: Path,
    work_queue_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Replay the full compiler population into a text-free claim graph."""

    if output_root.exists() or output_root.is_symlink():
        raise BoundedPilotConceptClaimError("concept claim output differs")
    candidates, lineage, population = load_population(population_root)
    work_by_identity, work_receipt = _load_work_queue(work_queue_root)
    if (
        set(work_by_identity)
        != {row["candidate_identity_sha256"] for row in candidates}
        or work_receipt.get("population_receipt_sha256") != population["receipt_sha256"]
    ):
        raise BoundedPilotConceptClaimError("concept claim coverage differs")
    claims = []
    for candidate, source in zip(candidates, lineage, strict=True):
        identity = candidate["candidate_identity_sha256"]
        receipt = _validate_compiler_receipt(
            _load_receipt(judgments_root / f"{identity}.compiler.json"), candidate
        )
        if receipt.get("request_reasoning_effort") != COMPILER_REASONING_EFFORT:
            raise BoundedPilotConceptClaimError("compiler reasoning effort differs")
        claims.append(build_claim(source, receipt, work_by_identity[identity]))
    nodes, edges, bridges = summarize_claims(claims)
    output_root.mkdir(parents=True)
    try:
        descriptors = {}
        for name, rows in (
            ("claims", claims),
            ("nodes", nodes),
            ("prerequisite_cooccurrences", edges),
            ("cross_domain_bridges", bridges),
        ):
            path = output_root / f"{name}.jsonl"
            _atomic_jsonl(path, rows)
            descriptors[name] = {
                "path": path.name,
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "ordered_records_sha256": canonical_sha256(
                    [row.get("claim_sha256", row.get("record_sha256")) for row in rows]
                ),
            }
        payload = {
            "schema": SCHEMA,
            "status": "complete_text_free_unverified_concept_claim_graph",
            "population_receipt_sha256": population["receipt_sha256"],
            "work_queue_receipt_sha256": work_receipt["receipt_sha256"],
            **descriptors,
            "records_by_content_route": dict(
                sorted(Counter(row["content_route"] for row in claims).items())
            ),
            "records_by_rights_route": dict(
                sorted(Counter(row["rights_route"] for row in claims).items())
            ),
            "source_text_persisted": False,
            "evidence_quotes_persisted": False,
            "model_annotations_independently_verified": False,
            "semantic_edges_verified": False,
            "verified_semantic_edges": 0,
            "curriculum_ready": False,
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
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--work-queue-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_graph(
        args.population_root,
        args.judgments_root,
        args.work_queue_root,
        args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
