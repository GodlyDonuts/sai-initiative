"""Seal exact per-identity content and rights work for the bounded pilot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_compiler_aggregate import (
    SCHEMA as AGGREGATE_SCHEMA,
)
from sai.data.bounded_pilot_compiler_aggregate import (
    combine_rights_and_model_routes,
    load_population,
    load_rights_queue,
)
from sai.data.data_yield_ledger import DataYieldLedgerError, _load_receipt
from sai.data.nous_compiler_worker import COMPILER_REASONING_EFFORT
from sai.data.reservoir_audit_aggregate import (
    _triage_route,
    _validate_compiler_receipt,
    summarize,
)
from sai.data.reservoir_audit_decision import LANES
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-bounded-pilot-work-queue-v1"
RECORD_SCHEMA = "sai-bounded-pilot-work-record-v1"


class BoundedPilotWorkQueueError(RuntimeError):
    """The aggregate, per-identity evidence, or work queue differs."""


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_aggregate(path: Path) -> dict[str, Any]:
    try:
        aggregate = _load_receipt(path)
    except DataYieldLedgerError as error:
        raise BoundedPilotWorkQueueError("aggregate receipt differs") from error
    unsigned = {
        key: value for key, value in aggregate.items() if key != "receipt_sha256"
    }
    if (
        aggregate.get("schema") != AGGREGATE_SCHEMA
        or aggregate.get("status") != "complete_nontraining_joint_evidence"
        or aggregate.get("compiler_judgments_are_verified_admissions") is not False
        or aggregate.get("independent_representation_verification_complete")
        is not False
        or aggregate.get("rights_provenance_verified") is not False
        or aggregate.get("legal_clearance_established") is not False
        or aggregate.get("training_ready") is not False
        or aggregate.get("four_b_training_authorized") is not False
        or aggregate.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise BoundedPilotWorkQueueError("aggregate receipt differs")
    return aggregate


def build_records(
    lineage: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    rights_by_identity: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join two independent work lanes without promoting either to admission."""

    if not lineage or len(lineage) != len(receipts):
        raise BoundedPilotWorkQueueError("work record inputs differ")
    records = []
    used_rights = set()
    candidate_identities = set()
    for source, receipt in zip(lineage, receipts, strict=True):
        candidate_identity = receipt.get("candidate_identity_sha256")
        retained_identity = source.get("retained_document_identity_sha256")
        rights = rights_by_identity.get(retained_identity)
        judgment = receipt.get("judgment")
        if (
            not isinstance(candidate_identity, str)
            or len(candidate_identity) != 64
            or candidate_identity in candidate_identities
            or not isinstance(retained_identity, str)
            or len(retained_identity) != 64
            or not isinstance(rights, dict)
            or rights.get("identity_sha256") != retained_identity
            or rights.get("source_id") != source.get("source_id")
            or not isinstance(judgment, dict)
        ):
            raise BoundedPilotWorkQueueError("work record identity differs")
        candidate_identities.add(candidate_identity)
        used_rights.add(retained_identity)
        content_route = _triage_route(judgment)
        record = {
            "schema": RECORD_SCHEMA,
            "candidate_identity_sha256": candidate_identity,
            "retained_document_identity_sha256": retained_identity,
            "source_id": source["source_id"],
            "compiler_receipt_sha256": receipt["receipt_sha256"],
            "compiler_judgment_sha256": judgment["judgment_sha256"],
            "compiler_verdict": judgment["verdict"],
            "content_route": content_route,
            "content_work_lane": LANES[content_route],
            "rights_record_sha256": rights["record_sha256"],
            "rights_route": rights["adjudication_route"],
            "expected_license_evidence_observed": rights[
                "expected_license_evidence_observed"
            ],
            "content_and_rights_lanes_are_independent": True,
            "model_retain_overrides_content_or_rights_lane": False,
            "rights_provenance_verified": False,
            "legal_clearance_established": False,
            "representation_verified": False,
            "training_ready": False,
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)
    if used_rights != set(rights_by_identity):
        raise BoundedPilotWorkQueueError("work record rights coverage differs")
    return records


def build_queue(
    population_root: Path,
    judgments_root: Path,
    rights_root: Path,
    aggregate_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Replay all evidence and publish a source-text-free operational queue."""

    if output_root.exists() or output_root.is_symlink():
        raise BoundedPilotWorkQueueError("work queue output differs")
    aggregate = _load_aggregate(aggregate_path)
    candidates, lineage, population = load_population(population_root)
    rights_by_identity, rights = load_rights_queue(rights_root)
    receipts = []
    expected_paths = set()
    for candidate in candidates:
        identity = candidate["candidate_identity_sha256"]
        path = judgments_root / f"{identity}.compiler.json"
        expected_paths.add(path)
        receipt = _validate_compiler_receipt(_load_receipt(path), candidate)
        if receipt["request_reasoning_effort"] != COMPILER_REASONING_EFFORT:
            raise BoundedPilotWorkQueueError("compiler reasoning effort differs")
        receipts.append(receipt)
    if set(judgments_root.glob("*.compiler.json")) != expected_paths:
        raise BoundedPilotWorkQueueError("compiler receipt population differs")

    summary_lineage = [
        {"source_id": row["source_id"], "stratum": row["source_type"]}
        for row in lineage
    ]
    compiler_summary = summarize(summary_lineage, receipts)
    joint_summary = combine_rights_and_model_routes(
        lineage, receipts, rights_by_identity
    )
    if (
        aggregate.get("population", {}).get("receipt_sha256")
        != population["receipt_sha256"]
        or aggregate.get("population", {}).get("rows") != len(candidates)
        or aggregate.get("rights_adjudication", {}).get("receipt_sha256")
        != rights["receipt_sha256"]
        or aggregate.get("compiler_summary") != compiler_summary
        or aggregate.get("joint_rights_and_model_routes") != joint_summary
    ):
        raise BoundedPilotWorkQueueError("aggregate evidence binding differs")

    records = build_records(lineage, receipts, rights_by_identity)
    content_counts = Counter(row["content_route"] for row in records)
    rights_counts = Counter(row["rights_route"] for row in records)
    source_counts = Counter(row["source_id"] for row in records)
    joint_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        joint_counts[row["rights_route"]][row["content_route"]] += 1

    output_root.mkdir(parents=True)
    try:
        queue_path = output_root / "work_queue.jsonl"
        _atomic_jsonl(queue_path, records)
        payload = {
            "schema": SCHEMA,
            "status": "complete_text_free_nontraining_work_queue",
            "aggregate": {
                "path": aggregate_path.name,
                "bytes": aggregate_path.stat().st_size,
                "sha256": sha256_file(aggregate_path),
                "receipt_sha256": aggregate["receipt_sha256"],
            },
            "population_receipt_sha256": population["receipt_sha256"],
            "rights_receipt_sha256": rights["receipt_sha256"],
            "queue": {
                "path": queue_path.name,
                "rows": len(records),
                "bytes": queue_path.stat().st_size,
                "sha256": sha256_file(queue_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in records]
                ),
            },
            "records_by_source": dict(sorted(source_counts.items())),
            "records_by_content_route": dict(sorted(content_counts.items())),
            "records_by_rights_route": dict(sorted(rights_counts.items())),
            "content_route_by_rights_route": {
                route: dict(sorted(counts.items()))
                for route, counts in sorted(joint_counts.items())
            },
            "exact_identity_coverage": True,
            "content_and_rights_lanes_are_independent": True,
            "source_text_persisted": False,
            "source_page_text_persisted": False,
            "compiler_judgments_are_verified_admissions": False,
            "rights_provenance_verified": False,
            "legal_clearance_established": False,
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
    parser.add_argument("--rights-root", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_queue(
        args.population_root,
        args.judgments_root,
        args.rights_root,
        args.aggregate,
        args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
