"""Aggregate same-family prerequisite verification into source-text-free lanes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.compiler_prerequisite_edge_labeling import (
    JUDGMENT_SCHEMA,
    RUBRIC,
    RUBRIC_SHA256,
    normalize_model_judgment,
)
from sai.data.compiler_prerequisite_edge_population import (
    SCHEMA as POPULATION_SCHEMA,
)
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.nous_compiler_prerequisite_edge_verifier import (
    OUTPUT_SUFFIX,
    REASONING_EFFORT,
    RECEIPT_SCHEMA,
    SUMMARY_SCHEMA,
    load_candidates,
)
from sai.data.nous_label_worker import DEFAULT_MODEL, _assigned
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-compiler-prerequisite-edge-verification-aggregate-v1"
STRICT_SCHEMA = "sai-same-family-strict-prerequisite-edge-candidate-v1"
HELPFUL_SCHEMA = "sai-same-family-helpful-foundation-edge-candidate-v1"
NONEDGE_SCHEMA = "sai-prerequisite-co-taught-nonedge-record-v1"
UNSUPPORTED_SCHEMA = "sai-prerequisite-unsupported-edge-record-v1"
RAW_JUDGMENT_KEYS = tuple(RUBRIC)


class CompilerPrerequisiteEdgeAggregateError(RuntimeError):
    """The prerequisite population, receipt, shard, or route differs."""


def load_population(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the exact complete-compiler edge proposal population."""

    receipt = _load_receipt(root / "receipt.json")
    descriptor = receipt.get("candidates")
    if (
        receipt.get("schema") != POPULATION_SCHEMA
        or receipt.get("status") != "complete_nontraining_prerequisite_edge_proposals"
        or receipt.get("all_compiler_populations_complete") is not True
        or receipt.get("source_disjoint_support") is not True
        or receipt.get("compiler_cooccurrence_is_verified_edge") is not False
        or receipt.get("directional_prerequisite_verification_complete") is not False
        or receipt.get("acyclic_graph_construction_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise CompilerPrerequisiteEdgeAggregateError(
            "prerequisite edge population differs"
        )
    path = _bound_file(root, descriptor)
    try:
        candidates = load_candidates(path)
    except RuntimeError as error:
        raise CompilerPrerequisiteEdgeAggregateError(
            "prerequisite edge candidates differ"
        ) from error
    identities = [row["edge_identity_sha256"] for row in candidates]
    if (
        len(candidates) != descriptor.get("rows")
        or receipt.get("selection", {}).get("selected_edges") != len(candidates)
        or receipt.get("selection", {}).get("ordered_edge_identities_sha256")
        != canonical_sha256(identities)
        or descriptor.get("text_bytes")
        != sum(
            len(anchor["text"].encode())
            for row in candidates
            for anchor in row["supporting_anchors"]
        )
    ):
        raise CompilerPrerequisiteEdgeAggregateError(
            "prerequisite edge population coverage differs"
        )
    return candidates, receipt


def validate_receipt(
    receipt: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Replay one exact Hermès edge-verification receipt."""

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    judgment = receipt.get("judgment")
    if not isinstance(judgment, dict):
        raise CompilerPrerequisiteEdgeAggregateError(
            "prerequisite verification judgment differs"
        )
    raw = {key: judgment.get(key) for key in RAW_JUDGMENT_KEYS}
    try:
        replay = normalize_model_judgment(raw, candidate)
    except RuntimeError as error:
        raise CompilerPrerequisiteEdgeAggregateError(
            "prerequisite verification judgment differs"
        ) from error
    attempt_hashes = receipt.get("attempt_request_sha256s")
    attempts = receipt.get("attempts")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("candidate_identity_sha256")
        != candidate["candidate_identity_sha256"]
        or receipt.get("requested_model") != DEFAULT_MODEL
        or receipt.get("rubric_sha256") != RUBRIC_SHA256
        or receipt.get("request_reasoning_effort") != REASONING_EFFORT
        or receipt.get("api_key_persisted") is not False
        or receipt.get("tools_enabled") is not False
        or receipt.get("raw_source_is_training_data") is not False
        or receipt.get("training_ready") is not False
        or judgment.get("schema") != JUDGMENT_SCHEMA
        or judgment != replay
        or judgment.get("same_model_family_as_compiler") is not True
        or judgment.get("independent_request_verification_complete") is not True
        or judgment.get("independent_model_family_verification_complete") is not False
        or judgment.get("directional_prerequisite_verified") is not False
        or judgment.get("acyclic_graph_construction_complete") is not False
        or judgment.get("training_ready") is not False
        or not isinstance(attempt_hashes, list)
        or not attempt_hashes
        or not isinstance(attempts, list)
        or len(attempts) != len(attempt_hashes)
        or receipt.get("successful_request_sha256") != attempt_hashes[-1]
    ):
        raise CompilerPrerequisiteEdgeAggregateError(
            "prerequisite verification receipt differs"
        )
    return receipt


def _validate_summaries(
    candidates: list[dict[str, Any]], judgments_root: Path, logical_shards: int
) -> list[str]:
    expected_paths = {
        judgments_root / f"shard_{index:05d}.summary.json"
        for index in range(logical_shards)
    }
    if set(judgments_root.glob("shard_*.summary.json")) != expected_paths:
        raise CompilerPrerequisiteEdgeAggregateError(
            "prerequisite verification shard population differs"
        )
    hashes = []
    for index in range(logical_shards):
        summary = _load_receipt(judgments_root / f"shard_{index:05d}.summary.json")
        expected = sum(
            _assigned(row["candidate_identity_sha256"], logical_shards, index)
            for row in candidates
        )
        created = summary.get("created_judgments")
        preexisting = summary.get("preexisting_judgments")
        if (
            summary.get("schema") != SUMMARY_SCHEMA
            or summary.get("status") != "complete"
            or summary.get("model") != DEFAULT_MODEL
            or summary.get("rubric_sha256") != RUBRIC_SHA256
            or summary.get("logical_shards") != logical_shards
            or summary.get("shard_index") != index
            or summary.get("candidate_rows") != expected
            or summary.get("expected_judgments") != expected
            or not isinstance(created, int)
            or isinstance(created, bool)
            or created < 0
            or not isinstance(preexisting, int)
            or isinstance(preexisting, bool)
            or preexisting < 0
            or created + preexisting != expected
            or summary.get("api_key_persisted") is not False
            or summary.get("training_ready") is not False
        ):
            raise CompilerPrerequisiteEdgeAggregateError(
                "prerequisite verification shard summary differs"
            )
        hashes.append(summary["receipt_sha256"])
    return hashes


def _evidence_hashes(judgment: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "anchor_index": check["anchor_index"],
            "concept_present": check["concept_present"],
            "prerequisite_assumed": check["prerequisite_assumed"],
            "concept_quote_sha256": (
                hashlib.sha256(check["concept_quote"].encode()).hexdigest()
                if check["concept_present"]
                else None
            ),
            "prerequisite_quote_sha256": (
                hashlib.sha256(check["prerequisite_quote"].encode()).hexdigest()
                if check["prerequisite_assumed"]
                else None
            ),
        }
        for check in judgment["source_checks"]
    ]


def route_candidate(
    candidate: dict[str, Any], receipt: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Emit one source-text-free edge or nonedge record."""

    judgment = receipt["judgment"]
    verdict = judgment["verdict"]
    common = {
        "edge_identity_sha256": candidate["edge_identity_sha256"],
        "verification_candidate_identity_sha256": candidate[
            "candidate_identity_sha256"
        ],
        "verification_receipt_sha256": receipt["receipt_sha256"],
        "verification_judgment_sha256": judgment["judgment_sha256"],
        "prerequisite": candidate["prerequisite"],
        "concept": candidate["concept"],
        "primary_domain": candidate["primary_domain"],
        "supporting_documents": candidate["supporting_documents"],
        "supporting_anchor_candidate_identity_sha256s": [
            anchor["candidate_identity_sha256"]
            for anchor in candidate["supporting_anchors"]
        ],
        "supporting_anchor_content_sha256s": [
            anchor["source_content_sha256"]
            for anchor in candidate["supporting_anchors"]
        ],
        "source_evidence": _evidence_hashes(judgment),
        "source_text_persisted": False,
        "same_model_family_verification_complete": True,
        "independent_model_family_verification_complete": False,
        "directional_prerequisite_verified": False,
        "acyclic_graph_construction_complete": False,
        "training_ready": False,
    }
    if verdict == "strict_prerequisite":
        route = "strict"
        row = {
            "schema": STRICT_SCHEMA,
            **common,
            "prerequisite_definition": judgment["prerequisite_definition"],
            "concept_definition": judgment["concept_definition"],
            "limitations": judgment["limitations"],
            "confidence_ppm": judgment["confidence_ppm"],
        }
    elif verdict == "helpful_foundation":
        route = "helpful"
        row = {
            "schema": HELPFUL_SCHEMA,
            **common,
            "prerequisite_definition": judgment["prerequisite_definition"],
            "concept_definition": judgment["concept_definition"],
            "limitations": judgment["limitations"],
            "confidence_ppm": judgment["confidence_ppm"],
        }
    elif verdict == "co_taught_not_prerequisite":
        route = "co_taught"
        row = {
            "schema": NONEDGE_SCHEMA,
            **common,
            "defects": judgment["defects"],
            "nonedge_reason_sha256": hashlib.sha256(
                judgment["rationale"].encode()
            ).hexdigest(),
        }
    elif verdict == "unsupported":
        route = "unsupported"
        row = {
            "schema": UNSUPPORTED_SCHEMA,
            **common,
            "defects": judgment["defects"],
            "rejection_reason_sha256": hashlib.sha256(
                judgment["rationale"].encode()
            ).hexdigest(),
        }
    else:  # pragma: no cover - normalized verifier contract
        raise CompilerPrerequisiteEdgeAggregateError(
            "prerequisite verification route differs"
        )
    row["record_sha256"] = canonical_sha256(row)
    return route, row


def build_aggregate(
    population_root: Path,
    judgments_root: Path,
    output_root: Path,
    *,
    logical_shards: int = 64,
) -> dict[str, Any]:
    """Seal all decisions while keeping graph construction explicitly open."""

    if (
        isinstance(logical_shards, bool)
        or not isinstance(logical_shards, int)
        or not 1 <= logical_shards <= 10_000
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise CompilerPrerequisiteEdgeAggregateError(
            "prerequisite aggregate geometry differs"
        )
    candidates, population = load_population(population_root)
    expected_receipts = {
        judgments_root / f"{row['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        for row in candidates
    }
    if set(judgments_root.glob(f"*.{OUTPUT_SUFFIX}.json")) != expected_receipts:
        raise CompilerPrerequisiteEdgeAggregateError(
            "prerequisite verification receipt population differs"
        )
    rows_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    receipt_hashes = []
    usage: Counter[str] = Counter()
    for candidate in candidates:
        path = judgments_root / (
            f"{candidate['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        )
        receipt = validate_receipt(_load_receipt(path), candidate)
        receipt_hashes.append(receipt["receipt_sha256"])
        route, row = route_candidate(candidate, receipt)
        rows_by_route[route].append(row)
        usage.update(
            {
                key: value
                for key, value in receipt.get("usage", {}).items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
        )
    summary_hashes = _validate_summaries(candidates, judgments_root, logical_shards)
    output_root.mkdir(parents=True)
    try:
        descriptors = {}
        for route, filename in (
            ("strict", "strict_prerequisite_candidates.jsonl"),
            ("helpful", "helpful_foundation_candidates.jsonl"),
            ("co_taught", "co_taught_nonedges.jsonl"),
            ("unsupported", "unsupported_edges.jsonl"),
        ):
            rows = rows_by_route[route]
            path = output_root / filename
            _atomic_jsonl(path, rows)
            descriptors[route] = {
                "path": path.name,
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in rows]
                ),
            }
        payload = {
            "schema": SCHEMA,
            "status": "complete_same_family_prerequisite_edge_routes",
            "population": {
                "root_name": population_root.name,
                "receipt_file_sha256": sha256_file(population_root / "receipt.json"),
                "receipt_sha256": population["receipt_sha256"],
                "candidate_rows": len(candidates),
            },
            "logical_shards": logical_shards,
            "ordered_shard_summaries_sha256": canonical_sha256(summary_hashes),
            "ordered_verification_receipts_sha256": canonical_sha256(receipt_hashes),
            "verification_receipts": len(receipt_hashes),
            "strict_prerequisite_candidates": descriptors["strict"],
            "helpful_foundation_candidates": descriptors["helpful"],
            "co_taught_nonedges": descriptors["co_taught"],
            "unsupported_edges": descriptors["unsupported"],
            "route_counts": {
                route: len(rows_by_route[route])
                for route in ("strict", "helpful", "co_taught", "unsupported")
            },
            "usage": dict(sorted(usage.items())),
            "source_text_persisted_in_outputs": False,
            "same_model_family_as_compiler": True,
            "independent_request_verification_complete": True,
            "independent_model_family_verification_complete": False,
            "directional_prerequisite_verification_complete": False,
            "acyclic_graph_construction_complete": False,
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, default=64)
    args = parser.parse_args()
    result = build_aggregate(
        args.population_root,
        args.judgments_root,
        args.output_root,
        logical_shards=args.logical_shards,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
