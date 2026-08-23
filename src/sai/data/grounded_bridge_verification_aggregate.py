"""Aggregate same-family bridge verification into conservative work lanes."""

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
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.grounded_bridge_verification_population import (
    SCHEMA as POPULATION_SCHEMA,
)
from sai.data.grounded_bridge_verifier_labeling import (
    JUDGMENT_SCHEMA,
    RUBRIC,
    RUBRIC_SHA256,
    normalize_model_judgment,
)
from sai.data.nous_grounded_bridge_verifier import (
    OUTPUT_SUFFIX,
    REASONING_EFFORT,
    RECEIPT_SCHEMA,
    SUMMARY_SCHEMA,
    load_candidates,
)
from sai.data.nous_label_worker import DEFAULT_MODEL
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-cross-domain-bridge-verification-aggregate-v1"
RETAINED_SCHEMA = "sai-same-family-retained-grounded-cross-domain-bridge-v1"
REVISION_SCHEMA = "sai-grounded-cross-domain-bridge-revision-work-v1"
REJECTION_SCHEMA = "sai-grounded-cross-domain-bridge-rejection-record-v1"
RAW_JUDGMENT_KEYS = tuple(RUBRIC)


class GroundedBridgeVerificationAggregateError(RuntimeError):
    """The verification population, receipt, shard, or route differs."""


def load_population(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the complete private bridge verification population."""

    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("candidates")
    if (
        receipt.get("schema") != POPULATION_SCHEMA
        or receipt.get("status")
        != "complete_nontraining_bridge_verification_population"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("source_disjoint_pairs") is not True
        or receipt.get("same_model_family_as_generator") is not True
        or receipt.get("independent_request_verification_complete") is not False
        or receipt.get("independent_model_family_verification_complete") is not False
        or receipt.get("bridge_verification_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise GroundedBridgeVerificationAggregateError(
            "bridge verification population differs"
        )
    path = _bound_file(root, descriptor)
    try:
        candidates = load_candidates(path)
    except RuntimeError as error:
        raise GroundedBridgeVerificationAggregateError(
            "bridge verification candidates differ"
        ) from error
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if (
        len(candidates) != descriptor.get("rows")
        or descriptor.get("ordered_identities_sha256") != canonical_sha256(identities)
        or descriptor.get("anchor_a_text_bytes")
        != sum(len(row["anchor_a_text"].encode()) for row in candidates)
        or descriptor.get("anchor_b_text_bytes")
        != sum(len(row["anchor_b_text"].encode()) for row in candidates)
        or descriptor.get("generated_text_bytes")
        != sum(len(row["generated_text"].encode()) for row in candidates)
    ):
        raise GroundedBridgeVerificationAggregateError(
            "bridge verification population coverage differs"
        )
    return candidates, receipt


def validate_receipt(
    receipt: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Replay one exact verifier receipt and its normalized decision."""

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    judgment = receipt.get("judgment")
    if not isinstance(judgment, dict):
        raise GroundedBridgeVerificationAggregateError(
            "bridge verification judgment differs"
        )
    raw = {key: judgment.get(key) for key in RAW_JUDGMENT_KEYS}
    try:
        replay = normalize_model_judgment(raw, candidate)
    except RuntimeError as error:
        raise GroundedBridgeVerificationAggregateError(
            "bridge verification judgment differs"
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
        or judgment.get("pair_identity_sha256") != candidate["pair_identity_sha256"]
        or judgment.get("same_model_family_as_generator") is not True
        or judgment.get("independent_request_verification_complete") is not True
        or judgment.get("independent_model_family_verification_complete") is not False
        or judgment.get("bridge_verified") is not False
        or judgment.get("training_ready") is not False
        or not isinstance(attempt_hashes, list)
        or not attempt_hashes
        or not isinstance(attempts, list)
        or len(attempts) != len(attempt_hashes)
        or receipt.get("successful_request_sha256") != attempt_hashes[-1]
    ):
        raise GroundedBridgeVerificationAggregateError(
            "bridge verification receipt differs"
        )
    return receipt


def _validate_summaries(
    candidates: list[dict[str, Any]],
    judgments_root: Path,
    logical_shards: int,
) -> list[str]:
    expected_paths = {
        judgments_root / f"shard_{index:05d}.summary.json"
        for index in range(logical_shards)
    }
    if set(judgments_root.glob("shard_*.summary.json")) != expected_paths:
        raise GroundedBridgeVerificationAggregateError(
            "bridge verification shard population differs"
        )
    hashes = []
    for index in range(logical_shards):
        summary = _load_receipt(judgments_root / f"shard_{index:05d}.summary.json")
        expected = sum(
            int(row["candidate_identity_sha256"], 16) % logical_shards == index
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
            or not isinstance(preexisting, int)
            or isinstance(preexisting, bool)
            or created < 0
            or preexisting < 0
            or created + preexisting != expected
            or summary.get("api_key_persisted") is not False
            or summary.get("training_ready") is not False
        ):
            raise GroundedBridgeVerificationAggregateError(
                "bridge verification shard summary differs"
            )
        hashes.append(summary["receipt_sha256"])
    return hashes


def _evidence_hashes(judgment: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_evidence_sha256s": [
            (
                hashlib.sha256(check["evidence_quote"].encode()).hexdigest()
                if check["supported"]
                else None
            )
            for check in judgment["claim_checks"]
        ],
        "anchor_a_evidence_quote_sha256s": [
            hashlib.sha256(quote.encode()).hexdigest()
            for quote in judgment["anchor_a_evidence_quotes"]
        ],
        "anchor_b_evidence_quote_sha256s": [
            hashlib.sha256(quote.encode()).hexdigest()
            for quote in judgment["anchor_b_evidence_quotes"]
        ],
        "generated_evidence_quote_sha256s": [
            hashlib.sha256(quote.encode()).hexdigest()
            for quote in judgment["generated_evidence_quotes"]
        ],
    }


def route_candidate(
    candidate: dict[str, Any], receipt: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Emit one anchor-text-free retain, revision, or rejection record."""

    judgment = receipt["judgment"]
    generated = candidate["generated"]
    verdict = judgment["verdict"]
    common = {
        "verification_candidate_identity_sha256": candidate[
            "candidate_identity_sha256"
        ],
        "pair_identity_sha256": candidate["pair_identity_sha256"],
        "generated_candidate_identity_sha256": candidate[
            "generated_candidate_identity_sha256"
        ],
        "anchor_a_candidate_identity_sha256": candidate[
            "anchor_a_candidate_identity_sha256"
        ],
        "anchor_a_source_content_sha256": candidate["anchor_a_source_content_sha256"],
        "anchor_b_candidate_identity_sha256": candidate[
            "anchor_b_candidate_identity_sha256"
        ],
        "anchor_b_source_content_sha256": candidate["anchor_b_source_content_sha256"],
        "generator_receipt_sha256": candidate["generator_receipt_sha256"],
        "generator_judgment_sha256": candidate["generator_judgment_sha256"],
        "verification_receipt_sha256": receipt["receipt_sha256"],
        "verification_judgment_sha256": judgment["judgment_sha256"],
        "bridge_label": candidate["bridge_label"],
        "verification_confidence_ppm": judgment["confidence_ppm"],
        "source_disjoint": True,
        "source_text_persisted": False,
        "same_model_family_verification_complete": True,
        "independent_model_family_verification_complete": False,
        "benchmark_decontamination_complete": False,
        "global_deduplication_complete": False,
        "transfer_ablation_complete": False,
        "bridge_verified": False,
        "training_ready": False,
        **_evidence_hashes(judgment),
    }
    if verdict == "retain":
        row = {
            "schema": RETAINED_SCHEMA,
            **common,
            "bridge_thesis": generated["bridge_thesis"],
            "shared_structure": generated["shared_structure"],
            "claims": generated["claims"],
            "representations": generated["representations"],
            "prerequisite_map": generated["prerequisite_map"],
            "analogy_failure_modes": generated["analogy_failure_modes"],
            "verification_questions": generated["verification_questions"],
            "same_family_retention_passed": True,
        }
    elif verdict == "revise":
        row = {
            "schema": REVISION_SCHEMA,
            **common,
            "bridge_thesis": generated["bridge_thesis"],
            "shared_structure": generated["shared_structure"],
            "claims": generated["claims"],
            "representations": generated["representations"],
            "prerequisite_map": generated["prerequisite_map"],
            "analogy_failure_modes": generated["analogy_failure_modes"],
            "verification_questions": generated["verification_questions"],
            "unsupported_generated_claims": judgment["unsupported_generated_claims"],
            "defects": judgment["defects"],
            "revision_brief": judgment["revision_brief"],
            "revision_complete": False,
        }
    elif verdict == "reject":
        row = {
            "schema": REJECTION_SCHEMA,
            **common,
            "defects": judgment["defects"],
            "unsupported_generated_claims": judgment["unsupported_generated_claims"],
            "rejection_reason_sha256": hashlib.sha256(
                judgment["rationale"].encode()
            ).hexdigest(),
            "generated_text_persisted": False,
        }
    else:  # pragma: no cover - normalized verifier contract
        raise GroundedBridgeVerificationAggregateError(
            "bridge verification route differs"
        )
    row["record_sha256"] = canonical_sha256(row)
    return verdict, row


def build_aggregate(
    population_root: Path,
    judgments_root: Path,
    output_root: Path,
    *,
    logical_shards: int = 64,
) -> dict[str, Any]:
    """Seal every verifier decision while keeping all lanes nontraining."""

    if (
        isinstance(logical_shards, bool)
        or not isinstance(logical_shards, int)
        or not 1 <= logical_shards <= 10_000
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise GroundedBridgeVerificationAggregateError(
            "bridge verification aggregate geometry differs"
        )
    candidates, population = load_population(population_root)
    expected_receipts = {
        judgments_root / f"{row['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        for row in candidates
    }
    if set(judgments_root.glob(f"*.{OUTPUT_SUFFIX}.json")) != expected_receipts:
        raise GroundedBridgeVerificationAggregateError(
            "bridge verification receipt population differs"
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
    shard_hashes = _validate_summaries(candidates, judgments_root, logical_shards)
    output_root.mkdir(parents=True)
    try:
        descriptors = {}
        for route, filename in (
            ("retain", "retained_bridges.jsonl"),
            ("revise", "revision_queue.jsonl"),
            ("reject", "rejections.jsonl"),
        ):
            path = output_root / filename
            rows = rows_by_route[route]
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
            "status": "complete_same_family_bridge_verification_routes",
            "population": {
                "root_name": population_root.name,
                "receipt_file_sha256": sha256_file(population_root / "receipt.json"),
                "receipt_sha256": population["receipt_sha256"],
                "candidate_rows": len(candidates),
            },
            "logical_shards": logical_shards,
            "ordered_shard_summaries_sha256": canonical_sha256(shard_hashes),
            "ordered_verification_receipts_sha256": canonical_sha256(receipt_hashes),
            "verification_receipts": len(receipt_hashes),
            "retained": descriptors["retain"],
            "revision_queue": descriptors["revise"],
            "rejections": descriptors["reject"],
            "route_counts": {
                route: len(rows_by_route[route])
                for route in ("retain", "revise", "reject")
            },
            "usage": dict(sorted(usage.items())),
            "source_text_persisted_in_outputs": False,
            "same_model_family_as_generator": True,
            "independent_request_verification_complete": True,
            "independent_model_family_verification_complete": False,
            "benchmark_decontamination_complete": False,
            "global_deduplication_complete": False,
            "transfer_ablation_complete": False,
            "bridge_verification_complete": False,
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
