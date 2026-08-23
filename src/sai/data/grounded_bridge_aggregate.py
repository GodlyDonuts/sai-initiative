"""Aggregate paired-source bridge synthesis into unverified candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.grounded_bridge_labeling import (
    JUDGMENT_SCHEMA,
    RUBRIC_SHA256,
    normalize_candidate,
    normalize_model_judgment,
)
from sai.data.grounded_bridge_population import SCHEMA as POPULATION_SCHEMA
from sai.data.nous_grounded_bridge_worker import (
    OUTPUT_SUFFIX,
    REASONING_EFFORT,
    RECEIPT_SCHEMA,
    SUMMARY_SCHEMA,
    load_candidates,
)
from sai.data.nous_label_worker import DEFAULT_MODEL
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-cross-domain-bridge-aggregate-v1"
CANDIDATE_SCHEMA = "sai-grounded-cross-domain-bridge-unverified-candidate-v1"
RAW_JUDGMENT_KEYS = (
    "bridge_label",
    "bridge_thesis",
    "shared_structure",
    "claims",
    "representations",
    "prerequisite_map",
    "analogy_failure_modes",
    "verification_questions",
    "confidence_ppm",
)


class GroundedBridgeAggregateError(RuntimeError):
    """The population, generator custody, or aggregate differs."""


def _load_json(path: Path) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size <= 0
        or path.stat().st_size > (4 << 20)
    ):
        raise GroundedBridgeAggregateError("bridge receipt is missing or unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise GroundedBridgeAggregateError(
            "bridge receipt cannot be decoded"
        ) from error
    if not isinstance(value, dict):
        raise GroundedBridgeAggregateError("bridge receipt differs")
    return value


def load_population(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay one exact proposal population without trusting its receipt alone."""

    receipt = _load_json(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("population")
    path = root / "candidates.jsonl"
    if (
        receipt.get("schema") != POPULATION_SCHEMA
        or receipt.get("status") != "complete_proposal_population_not_connection_data"
        or receipt.get("qualification_sha256") is None
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("source_disjoint_pairs") is not True
        or receipt.get("grounded_synthesis_complete") is not False
        or receipt.get("independent_verification_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
        or descriptor.get("rows") is None
        or descriptor.get("bytes") != path.stat().st_size
        or descriptor.get("sha256") != sha256_file(path)
    ):
        raise GroundedBridgeAggregateError("bridge population receipt differs")
    try:
        candidates = load_candidates(path)
    except RuntimeError as error:
        raise GroundedBridgeAggregateError("bridge population differs") from error
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if (
        len(candidates) != descriptor["rows"]
        or len(identities) != len(set(identities))
        or receipt.get("selection", {}).get("selected_pairs") != len(candidates)
        or receipt.get("selection", {}).get("ordered_pair_identity_sha256")
        != canonical_sha256(identities)
    ):
        raise GroundedBridgeAggregateError("bridge population coverage differs")
    return candidates, receipt


def validate_receipt(
    receipt: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Replay one exact generation receipt and its grounded judgment."""

    candidate = normalize_candidate(candidate)
    if not isinstance(receipt, dict):
        raise GroundedBridgeAggregateError("bridge generator receipt differs")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    judgment = receipt.get("judgment")
    if not isinstance(judgment, dict):
        raise GroundedBridgeAggregateError("bridge generator judgment differs")
    raw = {key: judgment.get(key) for key in RAW_JUDGMENT_KEYS}
    try:
        replay = normalize_model_judgment(raw, candidate)
    except RuntimeError as error:
        raise GroundedBridgeAggregateError(
            "bridge generator judgment differs"
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
        or judgment.get("grounded_synthesis_verified") is not False
        or judgment.get("benchmark_decontamination_complete") is not False
        or judgment.get("training_ready") is not False
        or not isinstance(attempt_hashes, list)
        or not attempt_hashes
        or not isinstance(attempts, list)
        or len(attempts) != len(attempt_hashes)
        or receipt.get("successful_request_sha256") != attempt_hashes[-1]
    ):
        raise GroundedBridgeAggregateError("bridge generator receipt differs")
    return receipt


def _validate_summaries(
    candidates: list[dict[str, Any]], judgments_root: Path, logical_shards: int
) -> list[str]:
    expected_paths = {
        judgments_root / f"shard_{index:05d}.summary.json"
        for index in range(logical_shards)
    }
    if set(judgments_root.glob("shard_*.summary.json")) != expected_paths:
        raise GroundedBridgeAggregateError("bridge shard population differs")
    hashes = []
    for index in range(logical_shards):
        summary = _load_json(judgments_root / f"shard_{index:05d}.summary.json")
        unsigned = {
            key: value for key, value in summary.items() if key != "receipt_sha256"
        }
        expected = sum(
            int(row["candidate_identity_sha256"], 16) % logical_shards == index
            for row in candidates
        )
        created = summary.get("created_judgments")
        preexisting = summary.get("preexisting_judgments")
        if (
            summary.get("schema") != SUMMARY_SCHEMA
            or summary.get("status") != "complete"
            or summary.get("receipt_sha256") != canonical_sha256(unsigned)
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
            raise GroundedBridgeAggregateError("bridge shard summary differs")
        hashes.append(summary["receipt_sha256"])
    return hashes


def build_candidate_row(
    candidate: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    """Strip source text and expose generated material only as unverified data."""

    judgment = receipt["judgment"]
    claims = [
        {
            "claim": claim["claim"],
            "anchor_side": claim["anchor_side"],
            "evidence_quote_sha256": hashlib.sha256(
                claim["evidence_quote"].encode()
            ).hexdigest(),
        }
        for claim in judgment["claims"]
    ]
    row = {
        "schema": CANDIDATE_SCHEMA,
        "pair_identity_sha256": candidate["pair_identity_sha256"],
        "bridge_label": candidate["bridge_label"],
        "bridge_endpoints": candidate["bridge_endpoints"],
        "anchor_a_candidate_identity_sha256": candidate["anchor_a"][
            "candidate_identity_sha256"
        ],
        "anchor_a_source_content_sha256": candidate["anchor_a"][
            "source_content_sha256"
        ],
        "anchor_b_candidate_identity_sha256": candidate["anchor_b"][
            "candidate_identity_sha256"
        ],
        "anchor_b_source_content_sha256": candidate["anchor_b"][
            "source_content_sha256"
        ],
        "generator_receipt_sha256": receipt["receipt_sha256"],
        "generator_judgment_sha256": judgment["judgment_sha256"],
        "bridge_thesis": judgment["bridge_thesis"],
        "shared_structure": judgment["shared_structure"],
        "claims": claims,
        "representations": judgment["representations"],
        "prerequisite_map": judgment["prerequisite_map"],
        "analogy_failure_modes": judgment["analogy_failure_modes"],
        "verification_questions": judgment["verification_questions"],
        "confidence_ppm": judgment["confidence_ppm"],
        "source_disjoint": True,
        "source_quotes_retained_in_candidate": False,
        "grounded_synthesis_verified": False,
        "independent_claim_verification_complete": False,
        "independent_transfer_verification_complete": False,
        "benchmark_decontamination_complete": False,
        "global_deduplication_complete": False,
        "training_ready": False,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_aggregate(
    population_root: Path,
    judgments_root: Path,
    output_root: Path,
    *,
    logical_shards: int = 64,
) -> dict[str, Any]:
    """Seal complete generation custody and emit no training-ready rows."""

    if (
        isinstance(logical_shards, bool)
        or not isinstance(logical_shards, int)
        or not 1 <= logical_shards <= 10_000
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise GroundedBridgeAggregateError("bridge aggregate geometry differs")
    candidates, population = load_population(population_root)
    expected_receipts = {
        judgments_root / f"{row['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        for row in candidates
    }
    if set(judgments_root.glob(f"*.{OUTPUT_SUFFIX}.json")) != expected_receipts:
        raise GroundedBridgeAggregateError("bridge generator population differs")
    receipt_hashes = []
    generated = []
    usage: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    confidence_sum = 0
    for candidate in candidates:
        path = judgments_root / (
            f"{candidate['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        )
        receipt = validate_receipt(_load_json(path), candidate)
        receipt_hashes.append(receipt["receipt_sha256"])
        row = build_candidate_row(candidate, receipt)
        generated.append(row)
        labels[row["bridge_label"]] += 1
        confidence_sum += row["confidence_ppm"]
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
        candidate_path = output_root / "bridge_candidates.jsonl"
        _atomic_jsonl(candidate_path, generated)
        payload = {
            "schema": SCHEMA,
            "status": "complete_unverified_grounded_bridge_candidates",
            "population_receipt_sha256": population["receipt_sha256"],
            "population_file_sha256": population["population"]["sha256"],
            "rubric_sha256": RUBRIC_SHA256,
            "requested_model": DEFAULT_MODEL,
            "logical_shards": logical_shards,
            "rows": len(generated),
            "directed_bridge_labels": len(labels),
            "label_counts": dict(sorted(labels.items())),
            "mean_confidence_ppm": confidence_sum // len(generated),
            "usage": dict(sorted(usage.items())),
            "ordered_generator_receipts_sha256": canonical_sha256(receipt_hashes),
            "ordered_shard_receipts_sha256": canonical_sha256(shard_hashes),
            "candidates": {
                "path": candidate_path.name,
                "rows": len(generated),
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
                "ordered_identities_sha256": canonical_sha256(
                    [row["candidate_identity_sha256"] for row in generated]
                ),
            },
            "source_disjoint_pairs": True,
            "source_quotes_retained_in_candidates": False,
            "grounded_synthesis_complete": True,
            "independent_claim_verification_complete": False,
            "independent_transfer_verification_complete": False,
            "benchmark_decontamination_complete": False,
            "global_deduplication_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        receipt_path = output_root / "receipt.json"
        temporary = output_root / f".receipt.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        os.replace(temporary, receipt_path)
    except BaseException:
        for path in output_root.iterdir():
            path.unlink(missing_ok=True)
        output_root.rmdir()
        raise
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, default=64)
    args = parser.parse_args()
    payload = build_aggregate(
        args.population_root,
        args.judgments_root,
        args.output_root,
        logical_shards=args.logical_shards,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["rows"],
                "directed_bridge_labels": payload["directed_bridge_labels"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
