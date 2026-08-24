"""Aggregate independent-model-family bridge verification into work lanes."""

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
from sai.data.grounded_bridge_verification_aggregate import (
    REJECTION_SCHEMA as SAME_FAMILY_REJECTION_SCHEMA,
)
from sai.data.grounded_bridge_verification_aggregate import (
    RETAINED_SCHEMA as SAME_FAMILY_RETAINED_SCHEMA,
)
from sai.data.grounded_bridge_verification_aggregate import (
    REVISION_SCHEMA as SAME_FAMILY_REVISION_SCHEMA,
)
from sai.data.grounded_bridge_verification_aggregate import (
    SCHEMA as SAME_FAMILY_AGGREGATE_SCHEMA,
)
from sai.data.grounded_bridge_verification_aggregate import (
    _evidence_hashes,
    load_population,
)
from sai.data.grounded_bridge_verifier_labeling import (
    RUBRIC_SHA256 as SAME_FAMILY_RUBRIC_SHA256,
)
from sai.data.nemotron_grounded_bridge_verifier import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OUTPUT_SUFFIX,
    REASONING_EFFORT,
    RECEIPT_SCHEMA,
    SUMMARY_SCHEMA,
)
from sai.data.nemotron_grounded_bridge_verifier_labeling import (
    INDEPENDENT_RUBRIC_SHA256,
    JUDGMENT_SCHEMA,
    normalize_model_judgment,
)
from sai.data.nemotron_grounded_bridge_verifier_labeling import (
    RUBRIC as INDEPENDENT_RUBRIC,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = (
    "sai-grounded-cross-domain-independent-model-family-"
    "bridge-verification-aggregate-v1"
)
RETAINED_SCHEMA = (
    "sai-independent-model-family-retained-grounded-cross-domain-bridge-v1"
)
REVISION_SCHEMA = (
    "sai-grounded-cross-domain-bridge-independent-model-family-" "revision-work-v1"
)
REJECTION_SCHEMA = (
    "sai-grounded-cross-domain-bridge-independent-model-family-" "rejection-record-v1"
)
RAW_JUDGMENT_KEYS = tuple(INDEPENDENT_RUBRIC)


class NemotronBridgeVerificationAggregateError(RuntimeError):
    """The verification population, receipt, shard, or route differs."""


def load_same_family_routes(
    root: Path, population: dict[str, Any], expected_rows: int
) -> tuple[dict[str, str], dict[str, Any]]:
    """Replay exact Hermès routes so Nemotron cannot overwrite a hold."""

    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != SAME_FAMILY_AGGREGATE_SCHEMA
        or receipt.get("status") != "complete_same_family_bridge_verification_routes"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("population", {}).get("receipt_sha256")
        != population.get("receipt_sha256")
        or receipt.get("population", {}).get("candidate_rows") != expected_rows
        or receipt.get("same_model_family_as_generator") is not True
        or receipt.get("independent_model_family_verification_complete") is not False
        or receipt.get("training_ready") is not False
    ):
        raise NemotronBridgeVerificationAggregateError(
            "same-family bridge aggregate differs"
        )
    routes: dict[str, str] = {}
    for route, descriptor_key, schema in (
        ("retain", "retained", SAME_FAMILY_RETAINED_SCHEMA),
        ("revise", "revision_queue", SAME_FAMILY_REVISION_SCHEMA),
        ("reject", "rejections", SAME_FAMILY_REJECTION_SCHEMA),
    ):
        descriptor = receipt.get(descriptor_key)
        if not isinstance(descriptor, dict):
            raise NemotronBridgeVerificationAggregateError(
                "same-family bridge route descriptor differs"
            )
        path = _bound_file(root, descriptor)
        rows = 0
        try:
            with path.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    identity = row.get("verification_candidate_identity_sha256")
                    unsigned_row = {
                        key: value
                        for key, value in row.items()
                        if key != "record_sha256"
                    }
                    if (
                        row.get("schema") != schema
                        or not isinstance(identity, str)
                        or identity in routes
                        or row.get("record_sha256") != canonical_sha256(unsigned_row)
                        or row.get("source_text_persisted") is not False
                        or row.get("training_ready") is not False
                    ):
                        raise NemotronBridgeVerificationAggregateError(
                            "same-family bridge route row differs"
                        )
                    routes[identity] = route
                    rows += 1
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise NemotronBridgeVerificationAggregateError(
                "same-family bridge route row differs"
            ) from error
        if rows != descriptor.get("rows"):
            raise NemotronBridgeVerificationAggregateError(
                "same-family bridge route coverage differs"
            )
    if len(routes) != expected_rows:
        raise NemotronBridgeVerificationAggregateError(
            "same-family bridge aggregate coverage differs"
        )
    return routes, receipt


def request_accounting(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    """Account for provider attempts even when streamed token usage is absent."""

    outcomes: Counter[str] = Counter()
    attempts = 0
    missing_usage = 0
    for receipt in receipts:
        rows = receipt.get("attempts")
        usage = receipt.get("usage")
        if not isinstance(rows, list) or not rows or not isinstance(usage, dict):
            raise NemotronBridgeVerificationAggregateError(
                "bridge verification request accounting differs"
            )
        attempts += len(rows)
        for row in rows:
            outcome = row.get("outcome") if isinstance(row, dict) else None
            if not isinstance(outcome, str) or not outcome:
                raise NemotronBridgeVerificationAggregateError(
                    "bridge verification request accounting differs"
                )
            outcomes[outcome] += 1
        if not any(
            isinstance(value, int) and not isinstance(value, bool)
            for value in usage.values()
        ):
            missing_usage += 1
    return {
        "receipts": len(receipts),
        "provider_attempts": attempts,
        "attempt_outcomes": dict(sorted(outcomes.items())),
        "receipts_without_provider_token_usage": missing_usage,
        "missing_provider_token_usage_is_zero_usage": False,
    }


def validate_receipt(
    receipt: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Replay one exact independent verifier receipt and its decision."""

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    judgment = receipt.get("judgment")
    if not isinstance(judgment, dict):
        raise NemotronBridgeVerificationAggregateError(
            "bridge verification judgment differs"
        )
    raw = {key: judgment.get(key) for key in RAW_JUDGMENT_KEYS}
    try:
        replay = normalize_model_judgment(raw, candidate)
    except RuntimeError as error:
        raise NemotronBridgeVerificationAggregateError(
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
        or receipt.get("endpoint_origin") != DEFAULT_BASE_URL
        or receipt.get("credential_transport") != "direct_portal_bearer"
        or receipt.get("rubric_sha256") != INDEPENDENT_RUBRIC_SHA256
        or receipt.get("request_reasoning_effort") != REASONING_EFFORT
        or receipt.get("api_key_persisted") is not False
        or receipt.get("tools_enabled") is not False
        or receipt.get("raw_source_is_training_data") is not False
        or receipt.get("training_ready") is not False
        or judgment.get("schema") != JUDGMENT_SCHEMA
        or judgment != replay
        or judgment.get("pair_identity_sha256") != candidate["pair_identity_sha256"]
        or judgment.get("same_family_rubric_sha256") != SAME_FAMILY_RUBRIC_SHA256
        or judgment.get("same_family_judgment_schema")
        != "sai-grounded-cross-domain-bridge-verification-judgment-v1"
        or judgment.get("same_model_family_as_generator") is not False
        or judgment.get("independent_request_verification_complete") is not True
        or judgment.get("independent_model_family_verification_complete") is not True
        or judgment.get("bridge_verified") is not False
        or judgment.get("training_ready") is not False
        or not isinstance(attempt_hashes, list)
        or not attempt_hashes
        or not isinstance(attempts, list)
        or len(attempts) != len(attempt_hashes)
        or receipt.get("successful_request_sha256") != attempt_hashes[-1]
    ):
        raise NemotronBridgeVerificationAggregateError(
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
        raise NemotronBridgeVerificationAggregateError(
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
            or summary.get("rubric_sha256") != INDEPENDENT_RUBRIC_SHA256
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
            raise NemotronBridgeVerificationAggregateError(
                "bridge verification shard summary differs"
            )
        hashes.append(summary["receipt_sha256"])
    return hashes


def route_candidate(
    candidate: dict[str, Any], receipt: dict[str, Any], *, same_family_route: str
) -> tuple[str, dict[str, Any]]:
    """Emit one anchor-text-free retain, revision, or rejection record."""

    judgment = receipt["judgment"]
    generated = candidate["generated"]
    verdict = judgment["verdict"]
    if same_family_route not in {"retain", "revise", "reject"}:
        raise NemotronBridgeVerificationAggregateError(
            "same-family bridge route differs"
        )
    cross_family_route = (
        "reject"
        if "reject" in {same_family_route, verdict}
        else "revise" if "revise" in {same_family_route, verdict} else "retain"
    )
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
        "same_family_route": same_family_route,
        "independent_family_route": verdict,
        "verification_confidence_ppm": judgment["confidence_ppm"],
        "source_disjoint": True,
        "source_text_persisted": False,
        "same_model_family_verification_complete": True,
        "independent_model_family_verification_complete": True,
        "benchmark_decontamination_complete": False,
        "global_deduplication_complete": False,
        "transfer_ablation_complete": False,
        "bridge_verified": False,
        "training_ready": False,
        **_evidence_hashes(judgment),
    }
    if cross_family_route == "retain":
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
            "independent_family_retention_passed": True,
        }
    elif cross_family_route == "revise":
        same_family_hold = same_family_route != "retain"
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
            "defects": (
                [*judgment["defects"], "same_family_revision_hold"]
                if same_family_hold
                else judgment["defects"]
            ),
            "revision_brief": (
                "Resolve the sealed same-family revision route before retention."
                if same_family_hold and not judgment["revision_brief"]
                else judgment["revision_brief"]
            ),
            "same_family_retention_passed": not same_family_hold,
            "independent_family_retention_passed": verdict == "retain",
            "revision_complete": False,
        }
    elif cross_family_route == "reject":
        row = {
            "schema": REJECTION_SCHEMA,
            **common,
            "defects": judgment["defects"],
            "unsupported_generated_claims": judgment["unsupported_generated_claims"],
            "rejection_reason_sha256": hashlib.sha256(
                judgment["rationale"].encode()
            ).hexdigest(),
            "generated_text_persisted": False,
            "same_family_retention_passed": same_family_route == "retain",
            "independent_family_retention_passed": verdict == "retain",
        }
    else:  # pragma: no cover - normalized verifier contract
        raise NemotronBridgeVerificationAggregateError(
            "bridge verification route differs"
        )
    row["record_sha256"] = canonical_sha256(row)
    return cross_family_route, row


def build_aggregate(
    population_root: Path,
    same_family_aggregate_root: Path,
    judgments_root: Path,
    output_root: Path,
    *,
    logical_shards: int = 64,
) -> dict[str, Any]:
    """Seal every independent verifier decision while keeping all lanes nontraining."""

    if (
        isinstance(logical_shards, bool)
        or not isinstance(logical_shards, int)
        or not 1 <= logical_shards <= 10_000
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise NemotronBridgeVerificationAggregateError(
            "bridge verification aggregate geometry differs"
        )
    candidates, population = load_population(population_root)
    same_family_routes, same_family_aggregate = load_same_family_routes(
        same_family_aggregate_root, population, len(candidates)
    )
    expected_receipts = {
        judgments_root / f"{row['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        for row in candidates
    }
    if set(judgments_root.glob(f"*.{OUTPUT_SUFFIX}.json")) != expected_receipts:
        raise NemotronBridgeVerificationAggregateError(
            "bridge verification receipt population differs"
        )
    rows_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    receipts = []
    receipt_hashes = []
    usage: Counter[str] = Counter()
    for candidate in candidates:
        path = judgments_root / (
            f"{candidate['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        )
        receipt = validate_receipt(_load_receipt(path), candidate)
        receipts.append(receipt)
        receipt_hashes.append(receipt["receipt_sha256"])
        route, row = route_candidate(
            candidate,
            receipt,
            same_family_route=same_family_routes[
                candidate["candidate_identity_sha256"]
            ],
        )
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
            "status": "complete_independent_model_family_bridge_verification_routes",
            "population": {
                "root_name": population_root.name,
                "receipt_file_sha256": sha256_file(population_root / "receipt.json"),
                "receipt_sha256": population["receipt_sha256"],
                "candidate_rows": len(candidates),
            },
            "same_family_aggregate": {
                "root_name": same_family_aggregate_root.name,
                "receipt_file_sha256": sha256_file(
                    same_family_aggregate_root / "receipt.json"
                ),
                "receipt_sha256": same_family_aggregate["receipt_sha256"],
            },
            "requested_model": DEFAULT_MODEL,
            "endpoint_origin": DEFAULT_BASE_URL,
            "same_family_rubric_sha256": SAME_FAMILY_RUBRIC_SHA256,
            "independent_rubric_sha256": INDEPENDENT_RUBRIC_SHA256,
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
            "request_accounting": request_accounting(receipts),
            "source_text_persisted_in_outputs": False,
            "same_model_family_as_generator": False,
            "independent_request_verification_complete": True,
            "independent_model_family_verification_complete": True,
            "benchmark_decontamination_complete": False,
            "global_deduplication_complete": False,
            "transfer_ablation_complete": False,
            "bridge_verification_complete": False,
            "bridge_verified": False,
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
    parser.add_argument("--same-family-aggregate-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, default=64)
    args = parser.parse_args()
    result = build_aggregate(
        args.population_root,
        args.same_family_aggregate_root,
        args.judgments_root,
        args.output_root,
        logical_shards=args.logical_shards,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
