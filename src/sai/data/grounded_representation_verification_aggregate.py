"""Aggregate same-family verifier decisions into retain/revise/reject lanes."""

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
from sai.data.grounded_representation_verification_population import (
    SCHEMA as POPULATION_SCHEMA,
)
from sai.data.grounded_representation_verifier_labeling import (
    JUDGMENT_SCHEMA,
    RUBRIC_SHA256,
    normalize_model_judgment,
)
from sai.data.nous_grounded_representation_verifier import (
    OUTPUT_SUFFIX,
    REASONING_EFFORT,
    RECEIPT_SCHEMA,
    SUMMARY_SCHEMA,
    load_candidates,
)
from sai.data.nous_label_worker import DEFAULT_MODEL
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-representation-verification-aggregate-v1"
RETAINED_SCHEMA = "sai-same-family-retained-grounded-representation-v1"
REVISION_SCHEMA = "sai-grounded-representation-revision-work-v1"
REJECTION_SCHEMA = "sai-grounded-representation-rejection-record-v1"
LOGICAL_SHARDS = 128


class GroundedRepresentationVerificationAggregateError(RuntimeError):
    """The verification population, receipt, shard, or route differs."""


def load_population(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the exact benchmark-disjoint verification population."""

    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("candidates")
    if (
        receipt.get("schema") != POPULATION_SCHEMA
        or receipt.get("status") != "complete_nontraining_verification_population"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("benchmark_decontamination_complete") is not True
        or receipt.get("same_model_family_as_generator") is not True
        or receipt.get("independent_request_verification_complete") is not False
        or receipt.get("independent_model_family_verification_complete") is not False
        or receipt.get("representation_verification_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise GroundedRepresentationVerificationAggregateError(
            "verification population differs"
        )
    path = _bound_file(root, descriptor)
    try:
        candidates = load_candidates(path)
    except RuntimeError as error:
        raise GroundedRepresentationVerificationAggregateError(
            "verification candidates differ"
        ) from error
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if (
        len(candidates) != descriptor.get("rows")
        or descriptor.get("ordered_identities_sha256") != canonical_sha256(identities)
        or descriptor.get("source_text_bytes")
        != sum(len(row["source_text"].encode()) for row in candidates)
        or descriptor.get("generated_text_bytes")
        != sum(len(row["generated_text"].encode()) for row in candidates)
    ):
        raise GroundedRepresentationVerificationAggregateError(
            "verification population coverage differs"
        )
    return candidates, receipt


def validate_receipt(
    receipt: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Replay one exact verifier receipt and normalized judgment."""

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    judgment = receipt.get("judgment")
    if not isinstance(judgment, dict):
        raise GroundedRepresentationVerificationAggregateError(
            "verification judgment differs"
        )
    raw_payload = {
        key: judgment.get(key)
        for key in (
            "verdict",
            "scores",
            "external_claims_present",
            "source_uncertainty_preserved",
            "cultural_specificity_preserved",
            "generic_model_style",
            "excessive_source_copying",
            "defects",
            "source_evidence_quotes",
            "representation_evidence_quotes",
            "revision_brief",
            "rationale",
        )
    }
    try:
        replay = normalize_model_judgment(raw_payload, candidate)
    except RuntimeError as error:
        raise GroundedRepresentationVerificationAggregateError(
            "verification judgment differs"
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
        or judgment.get("same_model_family_as_generator") is not True
        or judgment.get("independent_request_verification_complete") is not True
        or judgment.get("independent_model_family_verification_complete") is not False
        or judgment.get("training_ready") is not False
    ):
        raise GroundedRepresentationVerificationAggregateError(
            "verification receipt differs"
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
        raise GroundedRepresentationVerificationAggregateError(
            "verification shard population differs"
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
            raise GroundedRepresentationVerificationAggregateError(
                "verification shard summary differs"
            )
        hashes.append(summary["receipt_sha256"])
    return hashes


def route_candidate(
    candidate: dict[str, Any], receipt: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Emit one source-text-free retain, revision, or rejection record."""

    judgment = receipt["judgment"]
    verdict = judgment["verdict"]
    common = {
        "verification_candidate_identity_sha256": candidate[
            "candidate_identity_sha256"
        ],
        "source_candidate_identity_sha256": candidate[
            "source_candidate_identity_sha256"
        ],
        "generated_record_sha256": candidate["generated_record_sha256"],
        "clean_record_sha256": candidate["clean_record_sha256"],
        "generator_receipt_sha256": candidate["generator_receipt_sha256"],
        "verification_receipt_sha256": receipt["receipt_sha256"],
        "verification_judgment_sha256": judgment["judgment_sha256"],
        "representation_index": candidate["representation_index"],
        "representation_type": candidate["representation_type"],
        "source": candidate["source"],
        "attribution_required": True,
        "share_alike_required": True,
        "benchmark_decontamination_complete": True,
        "same_model_family_verification_complete": True,
        "independent_model_family_verification_complete": False,
        "global_deduplication_complete": False,
        "representation_verified": False,
        "training_ready": False,
    }
    if verdict == "retain":
        row = {
            "schema": RETAINED_SCHEMA,
            **common,
            "title": candidate["title"],
            "text": candidate["generated_text"],
            "text_sha256": candidate["generated_text_sha256"],
            "concepts": candidate["concepts"],
            "difficulty": candidate["difficulty"],
            "verification_scores": judgment["scores"],
            "same_family_retention_passed": True,
        }
    elif verdict == "revise":
        row = {
            "schema": REVISION_SCHEMA,
            **common,
            "title": candidate["title"],
            "text": candidate["generated_text"],
            "text_sha256": candidate["generated_text_sha256"],
            "defects": judgment["defects"],
            "revision_brief": judgment["revision_brief"],
            "source_evidence_quote_sha256s": [
                hashlib.sha256(quote.encode()).hexdigest()
                for quote in judgment["source_evidence_quotes"]
            ],
            "representation_evidence_quote_sha256s": [
                hashlib.sha256(quote.encode()).hexdigest()
                for quote in judgment["representation_evidence_quotes"]
            ],
            "revision_complete": False,
        }
    elif verdict == "reject":
        row = {
            "schema": REJECTION_SCHEMA,
            **common,
            "defects": judgment["defects"],
            "rejection_reason_sha256": hashlib.sha256(
                judgment["rationale"].encode()
            ).hexdigest(),
            "generated_text_persisted": False,
        }
    else:  # pragma: no cover - normalized verifier contract
        raise GroundedRepresentationVerificationAggregateError(
            "verification route differs"
        )
    row["record_sha256"] = canonical_sha256(row)
    return verdict, row


def build_aggregate(
    population_root: Path, judgments_root: Path, output_root: Path
) -> dict[str, Any]:
    """Seal all verifier decisions and conservative downstream work lanes."""

    if output_root.exists() or output_root.is_symlink():
        raise GroundedRepresentationVerificationAggregateError(
            "verification aggregate output differs"
        )
    candidates, population = load_population(population_root)
    expected_receipts = {
        judgments_root / f"{row['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        for row in candidates
    }
    if set(judgments_root.glob(f"*.{OUTPUT_SUFFIX}.json")) != expected_receipts:
        raise GroundedRepresentationVerificationAggregateError(
            "verification receipt population differs"
        )
    rows_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    receipt_hashes = []
    usage = Counter()
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
    shard_hashes = _validate_summaries(candidates, judgments_root)
    output_root.mkdir(parents=True)
    try:
        descriptors = {}
        for route, filename in (
            ("retain", "retained_representations.jsonl"),
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
            if route == "retain":
                descriptors[route]["text_bytes"] = sum(
                    len(row["text"].encode()) for row in rows
                )
        payload = {
            "schema": SCHEMA,
            "status": "complete_same_family_verification_routes",
            "population": {
                "root_name": population_root.name,
                "receipt_file_sha256": sha256_file(population_root / "receipt.json"),
                "receipt_sha256": population["receipt_sha256"],
                "candidate_rows": len(candidates),
            },
            "logical_shards": LOGICAL_SHARDS,
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
