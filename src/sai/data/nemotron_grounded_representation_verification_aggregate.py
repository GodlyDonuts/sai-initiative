"""Combine same-family and independent grounded-representation reviews."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.data_yield_ledger import _load_receipt
from sai.data.grounded_representation_verification_aggregate import (
    LOGICAL_SHARDS,
    load_population,
)
from sai.data.grounded_representation_verification_aggregate import (
    _validate_summaries as validate_same_family_summaries,
)
from sai.data.grounded_representation_verification_aggregate import (
    validate_receipt as validate_same_family_receipt,
)
from sai.data.grounded_representation_verifier_labeling import (
    OUTPUT_TEMPLATE,
)
from sai.data.nemotron_grounded_representation_verifier import (
    DEFAULT_MODEL,
    OUTPUT_SUFFIX,
    REASONING_EFFORT,
    RECEIPT_SCHEMA,
    SUMMARY_SCHEMA,
)
from sai.data.nemotron_grounded_representation_verifier_labeling import (
    INDEPENDENT_RUBRIC_SHA256,
    JUDGMENT_SCHEMA,
    normalize_model_judgment,
)
from sai.data.nous_grounded_representation_verifier import (
    OUTPUT_SUFFIX as SAME_FAMILY_OUTPUT_SUFFIX,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-cross-model-grounded-representation-verification-aggregate-v1"
RETAINED_SCHEMA = "sai-cross-model-retained-grounded-representation-v1"
REVISION_SCHEMA = "sai-cross-model-grounded-representation-revision-work-v1"
REJECTION_SCHEMA = "sai-cross-model-grounded-representation-rejection-record-v1"


class NemotronGroundedRepresentationAggregateError(RuntimeError):
    """The population, verifier receipt, or conservative route differs."""


def validate_independent_receipt(
    receipt: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Replay one exact independent-model-family receipt."""

    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    judgment = receipt.get("judgment")
    if not isinstance(judgment, dict):
        raise NemotronGroundedRepresentationAggregateError(
            "independent representation judgment differs"
        )
    raw_payload = {key: judgment.get(key) for key in OUTPUT_TEMPLATE}
    try:
        replay = normalize_model_judgment(raw_payload, candidate)
    except RuntimeError as error:
        raise NemotronGroundedRepresentationAggregateError(
            "independent representation judgment differs"
        ) from error
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("candidate_identity_sha256")
        != candidate["candidate_identity_sha256"]
        or receipt.get("requested_model") != DEFAULT_MODEL
        or receipt.get("rubric_sha256") != INDEPENDENT_RUBRIC_SHA256
        or receipt.get("request_reasoning_effort") != REASONING_EFFORT
        or receipt.get("api_key_persisted") is not False
        or receipt.get("tools_enabled") is not False
        or receipt.get("raw_source_is_training_data") is not False
        or receipt.get("training_ready") is not False
        or judgment.get("schema") != JUDGMENT_SCHEMA
        or judgment != replay
        or judgment.get("same_model_family_as_generator") is not False
        or judgment.get("independent_request_verification_complete") is not True
        or judgment.get("independent_model_family_verification_complete") is not True
        or judgment.get("training_ready") is not False
    ):
        raise NemotronGroundedRepresentationAggregateError(
            "independent representation receipt differs"
        )
    return receipt


def _validate_independent_summaries(
    candidates: list[dict[str, Any]], judgments_root: Path
) -> list[str]:
    expected_paths = {
        judgments_root / f"shard_{index:05d}.summary.json"
        for index in range(LOGICAL_SHARDS)
    }
    if set(judgments_root.glob("shard_*.summary.json")) != expected_paths:
        raise NemotronGroundedRepresentationAggregateError(
            "independent shard population differs"
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
            or summary.get("rubric_sha256") != INDEPENDENT_RUBRIC_SHA256
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
            raise NemotronGroundedRepresentationAggregateError(
                "independent shard summary differs"
            )
        hashes.append(summary["receipt_sha256"])
    return hashes


def route_candidate(
    candidate: dict[str, Any],
    same_family_receipt: dict[str, Any],
    independent_receipt: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Route by conservative cross-model agreement without training admission."""

    same = same_family_receipt["judgment"]
    independent = independent_receipt["judgment"]
    verdicts = (same["verdict"], independent["verdict"])
    if "reject" in verdicts:
        route = "reject"
    elif verdicts == ("retain", "retain"):
        route = "retain"
    else:
        route = "revise"
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
        "same_family_verification_receipt_sha256": same_family_receipt[
            "receipt_sha256"
        ],
        "same_family_verification_judgment_sha256": same["judgment_sha256"],
        "independent_verification_receipt_sha256": independent_receipt[
            "receipt_sha256"
        ],
        "independent_verification_judgment_sha256": independent[
            "judgment_sha256"
        ],
        "same_family_verdict": same["verdict"],
        "independent_verdict": independent["verdict"],
        "same_family_scores": same["scores"],
        "independent_scores": independent["scores"],
        "representation_index": candidate["representation_index"],
        "representation_type": candidate["representation_type"],
        "source": candidate["source"],
        "attribution_required": True,
        "share_alike_required": True,
        "benchmark_decontamination_complete": True,
        "same_family_verification_complete": True,
        "independent_model_family_verification_complete": True,
        "source_claims_independently_verified": False,
        "global_deduplication_complete": False,
        "training_ready": False,
    }
    if route == "retain":
        row = {
            "schema": RETAINED_SCHEMA,
            **common,
            "title": candidate["title"],
            "text": candidate["generated_text"],
            "text_sha256": candidate["generated_text_sha256"],
            "concepts": candidate["concepts"],
            "difficulty": candidate["difficulty"],
            "cross_model_retention_passed": True,
            "representation_verified": True,
        }
    elif route == "revise":
        row = {
            "schema": REVISION_SCHEMA,
            **common,
            "title": candidate["title"],
            "text": candidate["generated_text"],
            "text_sha256": candidate["generated_text_sha256"],
            "defects": sorted(set(same["defects"] + independent["defects"])),
            "same_family_revision_brief": same["revision_brief"],
            "independent_revision_brief": independent["revision_brief"],
            "representation_verified": False,
            "revision_complete": False,
        }
    else:
        row = {
            "schema": REJECTION_SCHEMA,
            **common,
            "defects": sorted(set(same["defects"] + independent["defects"])),
            "representation_verified": False,
            "generated_text_persisted": False,
        }
    row["record_sha256"] = canonical_sha256(row)
    return route, row


def build_aggregate(
    population_root: Path,
    same_family_judgments_root: Path,
    independent_judgments_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Seal complete cross-model routes for the benchmark-disjoint population."""

    if output_root.exists() or output_root.is_symlink():
        raise NemotronGroundedRepresentationAggregateError(
            "cross-model aggregate output differs"
        )
    candidates, population = load_population(population_root)
    expected_same = {
        same_family_judgments_root
        / f"{row['candidate_identity_sha256']}.{SAME_FAMILY_OUTPUT_SUFFIX}.json"
        for row in candidates
    }
    expected_independent = {
        independent_judgments_root
        / f"{row['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        for row in candidates
    }
    if (
        set(
            same_family_judgments_root.glob(
                f"*.{SAME_FAMILY_OUTPUT_SUFFIX}.json"
            )
        )
        != expected_same
        or set(independent_judgments_root.glob(f"*.{OUTPUT_SUFFIX}.json"))
        != expected_independent
    ):
        raise NemotronGroundedRepresentationAggregateError(
            "cross-model receipt population differs"
        )
    rows_by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    agreement = Counter()
    usage = Counter()
    same_receipt_hashes = []
    independent_receipt_hashes = []
    for candidate in candidates:
        identity = candidate["candidate_identity_sha256"]
        same_receipt = validate_same_family_receipt(
            _load_receipt(
                same_family_judgments_root
                / f"{identity}.{SAME_FAMILY_OUTPUT_SUFFIX}.json"
            ),
            candidate,
        )
        independent_receipt = validate_independent_receipt(
            _load_receipt(
                independent_judgments_root / f"{identity}.{OUTPUT_SUFFIX}.json"
            ),
            candidate,
        )
        same_receipt_hashes.append(same_receipt["receipt_sha256"])
        independent_receipt_hashes.append(independent_receipt["receipt_sha256"])
        agreement[
            f"{same_receipt['judgment']['verdict']}::{independent_receipt['judgment']['verdict']}"
        ] += 1
        route, row = route_candidate(candidate, same_receipt, independent_receipt)
        rows_by_route[route].append(row)
        for receipt in (same_receipt, independent_receipt):
            usage.update(
                {
                    key: value
                    for key, value in receipt.get("usage", {}).items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }
            )
    same_summary_hashes = validate_same_family_summaries(
        candidates, same_family_judgments_root
    )
    independent_summary_hashes = _validate_independent_summaries(
        candidates, independent_judgments_root
    )
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
            if route in {"retain", "revise"}:
                descriptors[route]["text_bytes"] = sum(
                    len(row["text"].encode()) for row in rows
                )
        payload = {
            "schema": SCHEMA,
            "status": "complete_cross_model_representation_verification_routes",
            "population": {
                "root_name": population_root.name,
                "receipt_file_sha256": sha256_file(population_root / "receipt.json"),
                "receipt_sha256": population["receipt_sha256"],
                "candidate_rows": len(candidates),
            },
            "logical_shards": LOGICAL_SHARDS,
            "ordered_same_family_shard_summaries_sha256": canonical_sha256(
                same_summary_hashes
            ),
            "ordered_independent_shard_summaries_sha256": canonical_sha256(
                independent_summary_hashes
            ),
            "ordered_same_family_receipts_sha256": canonical_sha256(
                same_receipt_hashes
            ),
            "ordered_independent_receipts_sha256": canonical_sha256(
                independent_receipt_hashes
            ),
            "agreement_counts": dict(sorted(agreement.items())),
            "retained": descriptors["retain"],
            "revision_queue": descriptors["revise"],
            "rejections": descriptors["reject"],
            "route_counts": {
                route: len(rows_by_route[route])
                for route in ("retain", "revise", "reject")
            },
            "usage": dict(sorted(usage.items())),
            "source_text_persisted_in_outputs": False,
            "benchmark_decontamination_complete": True,
            "same_family_verification_complete": True,
            "independent_model_family_verification_complete": True,
            "source_claims_independently_verified": False,
            "representation_verification_complete": True,
            "global_deduplication_complete": False,
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
    parser.add_argument("--same-family-judgments-root", type=Path, required=True)
    parser.add_argument("--independent-judgments-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_aggregate(
        args.population_root,
        args.same_family_judgments_root,
        args.independent_judgments_root,
        args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
