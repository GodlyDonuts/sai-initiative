"""Apply complete foundation decisions to grounded-bridge curriculum candidates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.grounded_bridge_curriculum_candidates import (
    RECEIPT_SCHEMA as CANDIDATE_RECEIPT_SCHEMA,
)
from sai.data.grounded_bridge_curriculum_candidates import STATUS as CANDIDATE_STATUS
from sai.data.grounded_bridge_foundation_query import SCHEMA as QUERY_SCHEMA
from sai.data.grounded_bridge_foundation_query import STATUS as QUERY_STATUS
from sai.data.grounded_bridge_foundation_query import _validate_candidate
from sai.data.grounded_bridge_foundation_scan_aggregate import (
    DOCUMENT_DECISION_SCHEMA,
    PAIR_DECISION_SCHEMA,
)
from sai.data.grounded_bridge_foundation_scan_aggregate import (
    SCHEMA as AGGREGATE_SCHEMA,
)
from sai.data.grounded_bridge_foundation_scan_aggregate import (
    STATUS as AGGREGATE_STATUS,
)
from sai.data.token_stream import canonical_sha256, sha256_file

ROW_SCHEMA = "sai-grounded-bridge-foundation-reconciled-candidate-v1"
SCHEMA = "sai-grounded-bridge-foundation-reconciled-candidates-v1"
STATUS = "complete_nontraining_grounded_bridge_foundation_reconciliation"
SPLIT_POLICY = {
    "schema": "sai-grounded-bridge-foundation-reconciled-split-policy-v1",
    "grouping": "bridge_pair_identity_sha256",
    "priority": [
        "matching_foundation_anchor_source_group_split",
        "provisional_pair_disjoint_split_when_no_anchor_is_in_foundation",
    ],
    "conflict_decision": "exclude_entire_pair",
    "exact_foundation_overlap_decision": "hold_overlapping_representation",
    "all_representations_overlap_decision": "exclude_entire_pair",
}
SPLIT_POLICY_SHA256 = canonical_sha256(SPLIT_POLICY)


class GroundedBridgeFoundationReconcileError(RuntimeError):
    """Candidate, global decision, split, or retained coverage differs."""


def _decision_rows(root: Path, descriptor: Any, schema: str) -> list[dict[str, Any]]:
    if (
        not isinstance(descriptor, dict)
        or descriptor.get("source_text_persisted") is not False
    ):
        raise GroundedBridgeFoundationReconcileError("decision descriptor differs")
    path = _bound_file(root, descriptor)
    rows = []
    ordered = []
    with path.open() as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise GroundedBridgeFoundationReconcileError(
                    "decision row differs"
                ) from error
            unsigned = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            if (
                row.get("schema") != schema
                or row.get("record_sha256") != canonical_sha256(unsigned)
                or row.get("training_ready") is not False
            ):
                raise GroundedBridgeFoundationReconcileError("decision row differs")
            rows.append(row)
            ordered.append(row["record_sha256"])
    if len(rows) != descriptor.get("rows") or canonical_sha256(
        ordered
    ) != descriptor.get("ordered_records_sha256"):
        raise GroundedBridgeFoundationReconcileError("decision coverage differs")
    return rows


def reconcile_candidates(
    candidate_root: Path,
    query_root: Path,
    aggregate_root: Path,
    output_root: Path,
    durable_receipt: Path,
) -> dict[str, Any]:
    """Write only globally unique, source-disjoint, ablation-pending bridges."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or durable_receipt.exists()
        or durable_receipt.is_symlink()
    ):
        raise GroundedBridgeFoundationReconcileError("reconciliation output differs")
    candidate_receipt = _load_receipt(candidate_root / "receipt.json")
    candidate_descriptor = candidate_receipt.get("curriculum_candidates")
    if (
        candidate_receipt.get("schema") != CANDIDATE_RECEIPT_SCHEMA
        or candidate_receipt.get("status") != CANDIDATE_STATUS
        or candidate_receipt.get("source_disjoint_against_foundation_complete")
        is not False
        or candidate_receipt.get("global_deduplication_against_foundation_complete")
        is not False
        or candidate_receipt.get("training_ready") is not False
        or not isinstance(candidate_descriptor, dict)
    ):
        raise GroundedBridgeFoundationReconcileError("candidate receipt differs")
    candidate_path = _bound_file(candidate_root, candidate_descriptor)
    query_receipt = _load_receipt(query_root / "receipt.json")
    if (
        query_receipt.get("schema") != QUERY_SCHEMA
        or query_receipt.get("status") != QUERY_STATUS
        or query_receipt.get("source_candidate_receipt_sha256")
        != candidate_receipt["receipt_sha256"]
        or query_receipt.get("training_ready") is not False
    ):
        raise GroundedBridgeFoundationReconcileError("query receipt differs")
    aggregate = _load_receipt(aggregate_root / "receipt.json")
    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("status") != AGGREGATE_STATUS
        or aggregate.get("source_query_receipt_sha256")
        != query_receipt["receipt_sha256"]
        or aggregate.get("global_foundation_scan_complete") is not True
        or aggregate.get("global_deduplication_against_foundation_complete") is not True
        or aggregate.get("source_disjoint_against_foundation_complete") is not True
        or aggregate.get("positive_transfer_ablation_complete") is not False
        or aggregate.get("training_ready") is not False
    ):
        raise GroundedBridgeFoundationReconcileError("scan aggregate differs")
    document_decisions = _decision_rows(
        aggregate_root,
        aggregate.get("document_decisions"),
        DOCUMENT_DECISION_SCHEMA,
    )
    pair_decisions = _decision_rows(
        aggregate_root, aggregate.get("pair_decisions"), PAIR_DECISION_SCHEMA
    )
    by_document = {}
    for row in document_decisions:
        identity = row.get("document_identity_sha256")
        if not isinstance(identity, str) or identity in by_document:
            raise GroundedBridgeFoundationReconcileError(
                "document decision identity differs"
            )
        by_document[identity] = row
    by_pair = {}
    for row in pair_decisions:
        pair = row.get("pair_identity_sha256")
        if not isinstance(pair, str) or pair in by_pair:
            raise GroundedBridgeFoundationReconcileError(
                "pair decision identity differs"
            )
        by_pair[pair] = row

    retained = []
    seen_candidates = set()
    seen_content = set()
    seen_normalized = set()
    input_records = []
    counts: Counter[str] = Counter()
    with candidate_path.open() as handle:
        for line in handle:
            try:
                candidate = _validate_candidate(json.loads(line))
            except (json.JSONDecodeError, KeyError) as error:
                raise GroundedBridgeFoundationReconcileError(
                    "candidate row differs"
                ) from error
            identity = candidate["document_identity_sha256"]
            pair = candidate["pair_identity_sha256"]
            decision = by_document.get(identity)
            pair_decision = by_pair.get(pair)
            if (
                identity in seen_candidates
                or decision is None
                or pair_decision is None
                or decision.get("pair_identity_sha256") != pair
                or decision.get("candidate_record_sha256") != candidate["record_sha256"]
                or decision.get("provisional_split") != candidate["corpus_split"]
                or pair_decision.get("provisional_split") != candidate["corpus_split"]
            ):
                raise GroundedBridgeFoundationReconcileError(
                    "candidate decision binding differs"
                )
            seen_candidates.add(identity)
            input_records.append(candidate["record_sha256"])
            route = decision.get("decision")
            counts[f"decision::{route}::documents"] += 1
            if route != "retain_pending_positive_transfer_ablation":
                continue
            resolved_split = decision.get("resolved_split")
            if (
                pair_decision.get("decision")
                != "retain_pending_positive_transfer_ablation"
                or resolved_split not in {"train", "development"}
                or resolved_split != pair_decision.get("resolved_split")
                or decision.get("exact_foundation_overlap")
                != {"word_signatures": 0, "code_signatures": 0}
            ):
                raise GroundedBridgeFoundationReconcileError(
                    "retained bridge decision differs"
                )
            content = candidate["content_sha256"]
            normalized = candidate["normalized_content_sha256"]
            if content in seen_content or normalized in seen_normalized:
                raise GroundedBridgeFoundationReconcileError(
                    "retained bridge duplicate differs"
                )
            seen_content.add(content)
            seen_normalized.add(normalized)
            row = {
                key: value
                for key, value in candidate.items()
                if key
                not in {
                    "schema",
                    "record_sha256",
                    "source_group_bucket",
                    "corpus_split",
                    "split_policy_sha256",
                    "source_disjoint_against_foundation_complete",
                    "global_deduplication_against_foundation_complete",
                }
            }
            row.update(
                {
                    "schema": ROW_SCHEMA,
                    "provisional_source_group_bucket": candidate["source_group_bucket"],
                    "provisional_corpus_split": candidate["corpus_split"],
                    "provisional_split_policy_sha256": candidate["split_policy_sha256"],
                    "corpus_split": resolved_split,
                    "split_policy_sha256": SPLIT_POLICY_SHA256,
                    "foundation_scan_aggregate_receipt_sha256": aggregate[
                        "receipt_sha256"
                    ],
                    "foundation_document_decision_sha256": decision["record_sha256"],
                    "foundation_pair_decision_sha256": pair_decision["record_sha256"],
                    "source_disjoint_against_foundation_complete": True,
                    "global_deduplication_against_foundation_complete": True,
                    "positive_transfer_ablation_complete": False,
                    "transfer_ablation_complete": False,
                    "bridge_verified": False,
                    "training_ready": False,
                }
            )
            row["record_sha256"] = canonical_sha256(row)
            retained.append(row)
            counts["retained_documents"] += 1
            counts[f"retained_split::{resolved_split}::documents"] += 1
            counts[f"retained_type::{row['document_type']}::documents"] += 1
            for domain in row["semantic_domains"]:
                counts[f"retained_domain::{domain}::documents"] += 1
    if (
        len(input_records) != candidate_descriptor.get("rows")
        or canonical_sha256(input_records)
        != candidate_descriptor.get("ordered_records_sha256")
        or seen_candidates != set(by_document)
        or set(by_pair)
        != {candidate["pair_identity_sha256"] for candidate in retained}
        | {
            row["pair_identity_sha256"]
            for row in pair_decisions
            if row["decision"] != "retain_pending_positive_transfer_ablation"
        }
        or not retained
    ):
        raise GroundedBridgeFoundationReconcileError(
            "reconciled bridge coverage differs"
        )
    retained_pairs = {row["pair_identity_sha256"] for row in retained}
    for pair in retained_pairs:
        expected = by_pair[pair]["representations"]["retained"]
        if sum(row["pair_identity_sha256"] == pair for row in retained) != expected:
            raise GroundedBridgeFoundationReconcileError(
                "retained pair representation coverage differs"
            )
    counts["input_documents"] = len(input_records)
    counts["input_pairs"] = len(by_pair)
    counts["retained_pairs"] = len(retained_pairs)
    counts["excluded_pairs"] = len(by_pair) - len(retained_pairs)

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        output = stage / "reconciled_candidates.jsonl"
        _atomic_jsonl(output, retained)
        payload = {
            "schema": SCHEMA,
            "status": STATUS,
            "source_candidate_receipt_sha256": candidate_receipt["receipt_sha256"],
            "source_query_receipt_sha256": query_receipt["receipt_sha256"],
            "source_scan_aggregate_receipt_sha256": aggregate["receipt_sha256"],
            "split_policy": SPLIT_POLICY,
            "split_policy_sha256": SPLIT_POLICY_SHA256,
            "counts": dict(sorted(counts.items())),
            "reconciled_candidates": {
                "path": output.name,
                "rows": len(retained),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in retained]
                ),
                "generated_text_persisted": True,
                "source_anchor_text_persisted": False,
            },
            "foundation_reconciliation_complete": True,
            "source_disjoint_against_foundation_complete": True,
            "global_deduplication_against_foundation_complete": True,
            "positive_transfer_ablation_complete": False,
            "bridge_verified": False,
            "huggingface_publication_authorized": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage / "receipt.json", payload)
        os.replace(stage, output_root)
        try:
            _atomic_create(durable_receipt, payload)
        except BaseException:
            shutil.rmtree(output_root, ignore_errors=True)
            raise
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--query-root", type=Path, required=True)
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--durable-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile_candidates(
        args.candidate_root,
        args.query_root,
        args.aggregate_root,
        args.output_root,
        args.durable_receipt,
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
