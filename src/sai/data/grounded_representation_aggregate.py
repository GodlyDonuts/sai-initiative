"""Aggregate generated PDR representations into an unverified candidate corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.grounded_representation_labeling import (
    JUDGMENT_SCHEMA,
    RUBRIC_SHA256,
    normalize_model_judgment,
)
from sai.data.nous_grounded_representation_worker import (
    OUTPUT_SUFFIX,
    REASONING_EFFORT,
    RECEIPT_SCHEMA,
    SUMMARY_SCHEMA,
    load_candidates,
)
from sai.data.nous_label_worker import DEFAULT_MODEL
from sai.data.public_domain_review_representation_population import (
    SCHEMA as POPULATION_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-representation-aggregate-v1"
REPRESENTATION_SCHEMA = "sai-generated-grounded-representation-candidate-v1"
EDGE_SCHEMA = "sai-generated-prerequisite-edge-candidate-v1"
BRIDGE_SCHEMA = "sai-generated-cross-domain-bridge-candidate-v1"
LOGICAL_SHARDS = 128


class GroundedRepresentationAggregateError(RuntimeError):
    """The population, receipt, shard, or generated row differs."""


def load_population(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the exact nontraining representation population."""

    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("candidates")
    if (
        receipt.get("schema") != POPULATION_SCHEMA
        or receipt.get("status") != "complete_nontraining_representation_population"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("source_license") != "CC-BY-SA-4.0"
        or receipt.get("attribution_required") is not True
        or receipt.get("share_alike_required") is not True
        or receipt.get("generated_representations_complete") is not False
        or receipt.get("independent_representation_verification_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise GroundedRepresentationAggregateError("representation population differs")
    path = _bound_file(root, descriptor)
    try:
        candidates = load_candidates(path)
    except RuntimeError as error:
        raise GroundedRepresentationAggregateError(
            "representation candidates differ"
        ) from error
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if (
        len(candidates) != descriptor.get("rows")
        or descriptor.get("ordered_identities_sha256") != canonical_sha256(identities)
        or descriptor.get("source_text_bytes")
        != sum(len(row["text"].encode()) for row in candidates)
    ):
        raise GroundedRepresentationAggregateError(
            "representation population coverage differs"
        )
    return candidates, receipt


def validate_receipt(
    receipt: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Replay one exact model receipt and its normalized judgment."""

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    judgment = receipt.get("judgment")
    if not isinstance(judgment, dict):
        raise GroundedRepresentationAggregateError(
            "grounded representation judgment differs"
        )
    raw_payload = {
        key: judgment.get(key)
        for key in (
            "representations",
            "prerequisite_edges",
            "cross_domain_bridge_candidates",
            "coverage_note",
        )
    }
    try:
        replay = normalize_model_judgment(raw_payload, candidate)
    except RuntimeError as error:
        raise GroundedRepresentationAggregateError(
            "grounded representation judgment differs"
        ) from error
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
        or judgment.get("candidate_identity_sha256")
        != candidate["candidate_identity_sha256"]
        or judgment.get("representation_verified") is not False
        or judgment.get("training_ready") is not False
    ):
        raise GroundedRepresentationAggregateError(
            "grounded representation receipt differs"
        )
    return receipt


def _validate_summaries(
    candidates: list[dict[str, Any]], judgments_root: Path
) -> list[str]:
    expected_paths = {
        judgments_root / f"shard_{index:05d}.summary.json"
        for index in range(LOGICAL_SHARDS)
    }
    if set(judgments_root.glob("shard_*.summary.json")) != expected_paths:
        raise GroundedRepresentationAggregateError(
            "grounded representation shard population differs"
        )
    hashes = []
    for index in range(LOGICAL_SHARDS):
        summary = _load_receipt(judgments_root / f"shard_{index:05d}.summary.json")
        expected = sum(
            int(row["candidate_identity_sha256"], 16) % LOGICAL_SHARDS == index
            for row in candidates
        )
        created = summary.get("created_judgments")
        preexisting = summary.get("preexisting_judgments")
        if (
            summary.get("schema") != SUMMARY_SCHEMA
            or summary.get("status") != "complete"
            or summary.get("model") != DEFAULT_MODEL
            or summary.get("rubric_sha256") != RUBRIC_SHA256
            or summary.get("logical_shards") != LOGICAL_SHARDS
            or summary.get("shard_index") != index
            or summary.get("candidate_rows") != expected
            or summary.get("expected_judgments") != expected
            or not isinstance(created, int)
            or isinstance(created, bool)
            or not isinstance(preexisting, int)
            or isinstance(preexisting, bool)
            or created < 0
            or preexisting < 0
            or created + preexisting != expected
            or summary.get("api_key_persisted") is not False
            or summary.get("training_ready") is not False
        ):
            raise GroundedRepresentationAggregateError(
                "grounded representation shard summary differs"
            )
        hashes.append(summary["receipt_sha256"])
    return hashes


def build_candidate_rows(
    candidate: dict[str, Any], receipt: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert one receipt into generated text, edge, and bridge candidates."""

    judgment = receipt["judgment"]
    source = candidate["source"]
    common = {
        "source_candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "source_text_sha256": candidate["source_text_sha256"],
        "source_record_sha256": candidate["source_record_sha256"],
        "compiler_judgment_sha256": candidate["compiler"]["judgment_sha256"],
        "generator_receipt_sha256": receipt["receipt_sha256"],
        "generator_judgment_sha256": judgment["judgment_sha256"],
        "source": {
            "dataset": source["dataset"],
            "row_id": source["row_id"],
            "source_url": source["source_url"],
            "license": source["license"],
        },
        "attribution_required": True,
        "share_alike_required": True,
    }
    representations = []
    for index, value in enumerate(judgment["representations"]):
        row = {
            "schema": REPRESENTATION_SCHEMA,
            **common,
            "representation_index": index,
            "representation_type": value["type"],
            "title": value["title"],
            "text": value["text"],
            "text_sha256": hashlib.sha256(value["text"].encode()).hexdigest(),
            "concepts": value["concepts"],
            "difficulty": value["difficulty"],
            "evidence_quote_sha256s": [
                hashlib.sha256(quote.encode()).hexdigest()
                for quote in value["evidence_quotes"]
            ],
            "source_claims_independently_verified": False,
            "benchmark_decontamination_complete": False,
            "global_deduplication_complete": False,
            "representation_verified": False,
            "training_ready": False,
        }
        row["record_sha256"] = canonical_sha256(row)
        representations.append(row)
    edges = []
    for index, value in enumerate(judgment["prerequisite_edges"]):
        row = {
            "schema": EDGE_SCHEMA,
            **common,
            "edge_index": index,
            "prerequisite": value["prerequisite"],
            "concept": value["concept"],
            "relation": value["relation"],
            "evidence_quote_sha256s": [
                hashlib.sha256(quote.encode()).hexdigest()
                for quote in value["evidence_quotes"]
            ],
            "semantic_edge_verified": False,
            "curriculum_ready": False,
            "training_ready": False,
        }
        row["record_sha256"] = canonical_sha256(row)
        edges.append(row)
    bridges = []
    for index, value in enumerate(judgment["cross_domain_bridge_candidates"]):
        row = {
            "schema": BRIDGE_SCHEMA,
            **common,
            "bridge_index": index,
            "bridge_label": value["bridge_label"],
            "connection": value["connection"],
            "connection_sha256": hashlib.sha256(
                value["connection"].encode()
            ).hexdigest(),
            "source_evidence_quote_sha256s": [
                hashlib.sha256(quote.encode()).hexdigest()
                for quote in value["source_evidence_quotes"]
            ],
            "external_anchor_required": True,
            "external_anchor_verified": False,
            "bridge_verified": False,
            "training_ready": False,
        }
        row["record_sha256"] = canonical_sha256(row)
        bridges.append(row)
    return representations, edges, bridges


def build_aggregate(
    population_root: Path, judgments_root: Path, output_root: Path
) -> dict[str, Any]:
    """Seal complete generation custody and emit unverified derivative candidates."""

    if output_root.exists() or output_root.is_symlink():
        raise GroundedRepresentationAggregateError(
            "grounded representation aggregate output differs"
        )
    candidates, population = load_population(population_root)
    expected_receipts = {
        judgments_root / f"{row['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        for row in candidates
    }
    if set(judgments_root.glob(f"*.{OUTPUT_SUFFIX}.json")) != expected_receipts:
        raise GroundedRepresentationAggregateError(
            "grounded representation receipt population differs"
        )
    receipts = []
    receipt_hashes = []
    representations = []
    edges = []
    bridges = []
    usage = Counter()
    for candidate in candidates:
        path = judgments_root / (
            f"{candidate['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        )
        receipt = validate_receipt(_load_receipt(path), candidate)
        receipts.append(receipt)
        receipt_hashes.append(receipt["receipt_sha256"])
        generated, generated_edges, generated_bridges = build_candidate_rows(
            candidate, receipt
        )
        representations.extend(generated)
        edges.extend(generated_edges)
        bridges.extend(generated_bridges)
        usage.update(
            {
                key: value
                for key, value in receipt.get("usage", {}).items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
        )
    shard_hashes = _validate_summaries(candidates, judgments_root)
    output_root.mkdir(parents=True)
    try:
        descriptors = {}
        for name, rows in (
            ("representations", representations),
            ("prerequisite_edges", edges),
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
                    [row["record_sha256"] for row in rows]
                ),
            }
        type_counts = Counter(row["representation_type"] for row in representations)
        payload = {
            "schema": SCHEMA,
            "status": "complete_unverified_grounded_representation_candidates",
            "population": {
                "root_name": population_root.name,
                "receipt_file_sha256": sha256_file(population_root / "receipt.json"),
                "receipt_sha256": population["receipt_sha256"],
                "candidate_rows": len(candidates),
            },
            "logical_shards": LOGICAL_SHARDS,
            "ordered_shard_summaries_sha256": canonical_sha256(shard_hashes),
            "ordered_generator_receipts_sha256": canonical_sha256(receipt_hashes),
            "generator_receipts": len(receipts),
            **descriptors,
            "representation_type_counts": dict(sorted(type_counts.items())),
            "usage": dict(sorted(usage.items())),
            "source_text_persisted_in_candidate_outputs": False,
            "evidence_quote_text_persisted_in_candidate_outputs": False,
            "source_license": "CC-BY-SA-4.0",
            "attribution_required": True,
            "share_alike_required": True,
            "source_claims_independently_verified": False,
            "external_bridge_anchors_verified": False,
            "benchmark_decontamination_complete": False,
            "global_deduplication_complete": False,
            "representation_verification_complete": False,
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
    args = parser.parse_args()
    result = build_aggregate(
        args.population_root, args.judgments_root, args.output_root
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
