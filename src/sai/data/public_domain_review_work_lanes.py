"""Join clean Public Domain Review text to independent compiler work lanes."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_concept_claims import _load_work_queue
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.public_domain_review_decontamination import (
    SCHEMA as DECONTAMINATION_SCHEMA,
)
from sai.data.public_domain_review_scope_audit import SOURCE_ID
from sai.data.public_domain_review_scoped_candidates import CANDIDATE_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-public-domain-review-work-lanes-v1"
LANE_RECORD_SCHEMA = "sai-public-domain-review-work-lane-record-v1"


class PublicDomainReviewWorkLaneError(RuntimeError):
    """The clean candidate or independent work-lane binding differs."""


def _load_clean_candidates(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("benchmark_disjoint_candidates")
    if (
        receipt.get("schema") != DECONTAMINATION_SCHEMA
        or receipt.get("status") != "complete_post_scope_benchmark_screen"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("post_transformation_benchmark_screen_complete") is not True
        or receipt.get("contaminated_rows") != 0
        or receipt.get("content_quality_verified") is not False
        or receipt.get("legal_clearance_established") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise PublicDomainReviewWorkLaneError("decontamination receipt differs")
    path = _bound_file(root, descriptor)
    rows = []
    identities = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            identity = row.get("original_candidate_identity_sha256")
            row_unsigned = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            if (
                row.get("schema") != CANDIDATE_SCHEMA
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in identities
                or row.get("record_sha256") != canonical_sha256(row_unsigned)
                or row.get("content_quality_verified") is not False
                or row.get("legal_clearance_established") is not False
                or row.get("training_ready") is not False
            ):
                raise PublicDomainReviewWorkLaneError("clean candidate differs")
            identities.add(identity)
            rows.append(row)
    if len(rows) != descriptor.get("rows") or descriptor.get(
        "ordered_records_sha256"
    ) != canonical_sha256([row["record_sha256"] for row in rows]):
        raise PublicDomainReviewWorkLaneError("clean candidate coverage differs")
    return rows, receipt


def build_lane_record(
    candidate: dict[str, Any], work: dict[str, Any]
) -> dict[str, Any]:
    identity = candidate.get("original_candidate_identity_sha256")
    if (
        work.get("retained_document_identity_sha256") != identity
        or work.get("source_id") != SOURCE_ID
        or work.get("representation_verified") is not False
        or work.get("legal_clearance_established") is not False
        or work.get("training_ready") is not False
    ):
        raise PublicDomainReviewWorkLaneError("work lane identity differs")
    row = {
        "schema": LANE_RECORD_SCHEMA,
        "original_candidate_identity_sha256": identity,
        "scoped_candidate_record_sha256": candidate["record_sha256"],
        "compiler_candidate_identity_sha256": work["candidate_identity_sha256"],
        "compiler_receipt_sha256": work["compiler_receipt_sha256"],
        "compiler_judgment_sha256": work["compiler_judgment_sha256"],
        "work_record_sha256": work["record_sha256"],
        "compiler_verdict": work["compiler_verdict"],
        "content_route": work["content_route"],
        "content_work_lane": work["content_work_lane"],
        "rights_record_sha256": work["rights_record_sha256"],
        "rights_route": work["rights_route"],
        "expected_license_evidence_observed": work[
            "expected_license_evidence_observed"
        ],
        "representation_priority_candidate": (
            work["content_route"] == "representation_verification"
        ),
        "compiler_route_is_verified_admission": False,
        "representation_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def build_work_lanes(
    decontamination_root: Path, work_queue_root: Path, output_root: Path
) -> dict[str, Any]:
    """Create exact PDR content/rights lanes and a representation priority set."""

    if output_root.exists() or output_root.is_symlink():
        raise PublicDomainReviewWorkLaneError("work lane output differs")
    candidates, decontamination = _load_clean_candidates(decontamination_root)
    work_by_compiler_identity, work_receipt = _load_work_queue(work_queue_root)
    pdr_work = [
        row
        for row in work_by_compiler_identity.values()
        if row["source_id"] == SOURCE_ID
    ]
    by_retained = {}
    for row in pdr_work:
        identity = row["retained_document_identity_sha256"]
        if identity in by_retained:
            raise PublicDomainReviewWorkLaneError("work lane retained identity differs")
        by_retained[identity] = row
    candidate_identities = {
        row["original_candidate_identity_sha256"] for row in candidates
    }
    if (
        len(pdr_work) != 1_342
        or not candidate_identities <= set(by_retained)
        or work_receipt.get("queue", {}).get("rows") != len(work_by_compiler_identity)
    ):
        raise PublicDomainReviewWorkLaneError("work lane coverage differs")
    lanes = [
        build_lane_record(
            candidate, by_retained[candidate["original_candidate_identity_sha256"]]
        )
        for candidate in candidates
    ]
    priority = [
        candidate
        for candidate, lane in zip(candidates, lanes, strict=True)
        if lane["representation_priority_candidate"]
    ]
    route_counts = Counter(row["content_route"] for row in lanes)
    rights_counts = Counter(row["rights_route"] for row in lanes)
    output_root.mkdir(parents=True)
    try:
        lanes_path = output_root / "work_lanes.jsonl"
        priority_path = output_root / "representation_priority_candidates.jsonl"
        _atomic_jsonl(lanes_path, lanes)
        _atomic_jsonl(priority_path, priority)
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_pdr_work_lanes",
            "decontamination": {
                "root_name": decontamination_root.name,
                "receipt_file_sha256": sha256_file(
                    decontamination_root / "receipt.json"
                ),
                "receipt_sha256": decontamination["receipt_sha256"],
                "candidate_file_sha256": decontamination[
                    "benchmark_disjoint_candidates"
                ]["sha256"],
            },
            "work_queue": {
                "root_name": work_queue_root.name,
                "receipt_file_sha256": sha256_file(work_queue_root / "receipt.json"),
                "receipt_sha256": work_receipt["receipt_sha256"],
            },
            "source_work_records": len(pdr_work),
            "materialized_clean_candidates": len(candidates),
            "work_lanes": {
                "path": lanes_path.name,
                "rows": len(lanes),
                "bytes": lanes_path.stat().st_size,
                "sha256": sha256_file(lanes_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in lanes]
                ),
                "source_text_persisted": False,
            },
            "representation_priority_candidates": {
                "path": priority_path.name,
                "rows": len(priority),
                "bytes": priority_path.stat().st_size,
                "sha256": sha256_file(priority_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in priority]
                ),
                "text_bytes": sum(row["scoped_text_bytes"] for row in priority),
            },
            "materialized_records_by_content_route": dict(sorted(route_counts.items())),
            "materialized_records_by_rights_route": dict(sorted(rights_counts.items())),
            "compiler_route_is_verified_admission": False,
            "independent_representation_verification_complete": False,
            "legal_clearance_established": False,
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
    parser.add_argument("--decontamination-root", type=Path, required=True)
    parser.add_argument("--work-queue-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_work_lanes(
        args.decontamination_root, args.work_queue_root, args.output_root
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
