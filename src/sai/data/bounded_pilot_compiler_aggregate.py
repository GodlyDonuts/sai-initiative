"""Aggregate bounded-pilot compiler judgments jointly with rights routing."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create, normalize_candidate
from sai.data.bounded_pilot_compiler_population import (
    LINEAGE_SCHEMA,
)
from sai.data.bounded_pilot_compiler_population import (
    SCHEMA as POPULATION_SCHEMA,
)
from sai.data.data_compiler_labeling import RUBRIC_SHA256
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.external_rights_adjudication_queue import (
    RECORD_SCHEMA as RIGHTS_RECORD_SCHEMA,
)
from sai.data.external_rights_adjudication_queue import SCHEMA as RIGHTS_SCHEMA
from sai.data.nous_compiler_worker import (
    COMPILER_REASONING_EFFORT,
    SUMMARY_SCHEMA,
)
from sai.data.nous_label_worker import DEFAULT_MODEL, _assigned
from sai.data.reservoir_audit_aggregate import (
    _triage_route,
    _valid_sha256,
    _validate_compiler_receipt,
    summarize,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-bounded-pilot-compiler-aggregate-v1"
LOGICAL_SHARDS = 128


class BoundedPilotCompilerAggregateError(RuntimeError):
    """Population, compiler, rights, or shard custody differs."""


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise BoundedPilotCompilerAggregateError("aggregate input is unsafe")
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise BoundedPilotCompilerAggregateError(
                    f"aggregate row {line_number} cannot be decoded"
                ) from error
            if not isinstance(row, dict):
                raise BoundedPilotCompilerAggregateError(
                    f"aggregate row {line_number} differs"
                )
            rows.append(row)
    if not rows:
        raise BoundedPilotCompilerAggregateError("aggregate input is empty")
    return rows


def load_population(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Replay the exact bounded candidates and their text-free lineage."""

    receipt = _load_receipt(root / "receipt.json")
    population_descriptor = receipt.get("population")
    lineage_descriptor = receipt.get("lineage")
    if (
        receipt.get("schema") != POPULATION_SCHEMA
        or receipt.get("status") != "complete_nontraining_compiler_population"
        or receipt.get("compiler_judgments_complete") is not False
        or receipt.get("representation_verification_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(population_descriptor, dict)
        or not isinstance(lineage_descriptor, dict)
        or lineage_descriptor.get("source_text_persisted") is not False
    ):
        raise BoundedPilotCompilerAggregateError("population receipt differs")
    candidate_path = _bound_file(root, population_descriptor)
    lineage_path = _bound_file(root, lineage_descriptor)
    candidates = [normalize_candidate(row) for row in _load_rows(candidate_path)]
    lineage = _load_rows(lineage_path)
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if (
        len(candidates) != len(lineage)
        or len(identities) != len(set(identities))
        or population_descriptor.get("rows") != len(candidates)
        or population_descriptor.get("ordered_identities_sha256")
        != canonical_sha256(identities)
        or lineage_descriptor.get("rows") != len(lineage)
        or lineage_descriptor.get("ordered_rows_sha256") != canonical_sha256(lineage)
    ):
        raise BoundedPilotCompilerAggregateError("population coverage differs")
    retained_identities = set()
    by_source = Counter()
    for candidate, source in zip(candidates, lineage, strict=True):
        unsigned = {
            key: value for key, value in source.items() if key != "lineage_sha256"
        }
        source_binding = source.get("source")
        rights = source.get("rights_declaration")
        retained = source.get("retained_document_identity_sha256")
        if (
            source.get("schema") != LINEAGE_SCHEMA
            or source.get("candidate_identity_sha256")
            != candidate["candidate_identity_sha256"]
            or source.get("source_content_sha256") != candidate["source_content_sha256"]
            or source.get("source_text_bytes") != len(candidate["text"].encode())
            or not isinstance(source_binding, dict)
            or not isinstance(rights, dict)
            or rights.get("rights_hold") is not False
            or not isinstance(retained, str)
            or len(retained) != 64
            or retained in retained_identities
            or source.get("raw_source_is_training_ready") is not False
            or source.get("lineage_sha256") != canonical_sha256(unsigned)
            or candidate["source"]
            != {
                "dataset": source_binding.get("dataset"),
                "revision": source_binding.get("revision"),
                "row_id": source.get("row_id"),
                "license": rights.get("canonical_license"),
                "source_type": source.get("source_type"),
            }
        ):
            raise BoundedPilotCompilerAggregateError("population lineage differs")
        retained_identities.add(retained)
        by_source[source["source_id"]] += 1
    if receipt.get("by_source") != dict(sorted(by_source.items())):
        raise BoundedPilotCompilerAggregateError("population source counts differ")
    return candidates, lineage, receipt


def load_rights_queue(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Replay the exact fail-closed per-identity rights routes."""

    receipt = _load_receipt(root / "receipt.json")
    descriptor = receipt.get("queue")
    if (
        receipt.get("schema") != RIGHTS_SCHEMA
        or receipt.get("status") != "complete_text_free_fail_closed_queue"
        or receipt.get("exact_identity_coverage") is not True
        or receipt.get("automated_legal_decision_made") is not False
        or receipt.get("access_control_bypassed") is not False
        or receipt.get("rights_provenance_verified") is not False
        or receipt.get("legal_clearance_established") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise BoundedPilotCompilerAggregateError("rights queue receipt differs")
    path = _bound_file(root, descriptor)
    rows = _load_rows(path)
    by_identity = {}
    route_counts = Counter()
    for row in rows:
        unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
        identity = row.get("identity_sha256")
        route = row.get("adjudication_route")
        if (
            row.get("schema") != RIGHTS_RECORD_SCHEMA
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in by_identity
            or not isinstance(route, str)
            or not route
            or not isinstance(row.get("expected_license_evidence_observed"), bool)
            or row.get("record_sha256") != canonical_sha256(unsigned)
            or row.get("rights_provenance_verified") is not False
            or row.get("legal_clearance_established") is not False
            or row.get("training_ready") is not False
        ):
            raise BoundedPilotCompilerAggregateError("rights queue row differs")
        by_identity[identity] = row
        route_counts[route] += 1
    if (
        len(rows) != descriptor.get("rows")
        or len(rows) != receipt.get("population_records")
        or descriptor.get("ordered_records_sha256")
        != canonical_sha256([row["record_sha256"] for row in rows])
        or receipt.get("records_by_adjudication_route")
        != dict(sorted(route_counts.items()))
    ):
        raise BoundedPilotCompilerAggregateError("rights queue coverage differs")
    return by_identity, receipt


def validate_shard_summaries(
    candidates: list[dict[str, Any]], judgments_root: Path
) -> list[str]:
    """Require one exact summary for every immutable identity shard."""

    expected_paths = {
        judgments_root / f"shard_{index:05d}.summary.json"
        for index in range(LOGICAL_SHARDS)
    }
    if set(judgments_root.glob("shard_*.summary.json")) != expected_paths:
        raise BoundedPilotCompilerAggregateError("shard summary population differs")
    hashes = []
    for index in range(LOGICAL_SHARDS):
        summary = _load_receipt(judgments_root / f"shard_{index:05d}.summary.json")
        created = summary.get("created_judgments")
        preexisting = summary.get("preexisting_judgments")
        expected = sum(
            _assigned(row["candidate_identity_sha256"], LOGICAL_SHARDS, index)
            for row in candidates
        )
        if (
            summary.get("schema") != SUMMARY_SCHEMA
            or summary.get("status") != "complete"
            or summary.get("model") != DEFAULT_MODEL
            or summary.get("rubric_sha256") != RUBRIC_SHA256
            or summary.get("logical_shards") != LOGICAL_SHARDS
            or summary.get("shard_index") != index
            or summary.get("candidate_rows") != expected
            or summary.get("expected_judgments") != expected
            or isinstance(created, bool)
            or not isinstance(created, int)
            or created < 0
            or isinstance(preexisting, bool)
            or not isinstance(preexisting, int)
            or preexisting < 0
            or created + preexisting != expected
            or not _valid_sha256(summary.get("created_receipts_sha256"))
            or summary.get("api_key_persisted") is not False
            or summary.get("training_ready") is not False
        ):
            raise BoundedPilotCompilerAggregateError("shard summary differs")
        hashes.append(summary["receipt_sha256"])
    return hashes


def combine_rights_and_model_routes(
    lineage: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    rights_by_identity: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Cross-tab independent model triage against fail-closed rights routing."""

    if not lineage or len(lineage) != len(receipts):
        raise BoundedPilotCompilerAggregateError("joint route inputs differ")
    verdict_by_rights: dict[str, Counter[str]] = defaultdict(Counter)
    triage_by_rights: dict[str, Counter[str]] = defaultdict(Counter)
    observed_representation_rows = 0
    used = set()
    for source, receipt in zip(lineage, receipts, strict=True):
        retained = source["retained_document_identity_sha256"]
        rights = rights_by_identity.get(retained)
        if rights is None or rights.get("source_id") != source["source_id"]:
            raise BoundedPilotCompilerAggregateError("joint rights identity differs")
        used.add(retained)
        route = rights["adjudication_route"]
        judgment = receipt["judgment"]
        model_triage = _triage_route(judgment)
        verdict_by_rights[route][judgment["verdict"]] += 1
        triage_by_rights[route][model_triage] += 1
        observed_representation_rows += (
            rights["expected_license_evidence_observed"]
            and model_triage == "representation_verification"
        )
    if used != set(rights_by_identity):
        raise BoundedPilotCompilerAggregateError("joint rights coverage differs")

    def nested(values: dict[str, Counter[str]]) -> dict[str, dict[str, int]]:
        return {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(values.items())
        }

    return {
        "compiler_verdict_by_rights_route": nested(verdict_by_rights),
        "model_triage_by_rights_route": nested(triage_by_rights),
        "rows_with_observed_evidence_and_representation_route": (
            observed_representation_rows
        ),
        "rights_route_overrides_model_retain_for_admission": True,
        "joint_route_is_training_admission": False,
    }


def build_aggregate(
    population_root: Path,
    judgments_root: Path,
    rights_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Replay every bounded candidate, judgment, shard, and rights route."""

    if output_path.exists() or output_path.is_symlink():
        raise BoundedPilotCompilerAggregateError("aggregate output differs")
    candidates, lineage, population = load_population(population_root)
    rights_by_identity, rights = load_rights_queue(rights_root)
    retained_identities = {row["retained_document_identity_sha256"] for row in lineage}
    if retained_identities != set(rights_by_identity):
        raise BoundedPilotCompilerAggregateError("population and rights differ")
    receipts = []
    receipt_hashes = []
    expected_paths = set()
    for candidate in candidates:
        identity = candidate["candidate_identity_sha256"]
        path = judgments_root / f"{identity}.compiler.json"
        expected_paths.add(path)
        receipt = _validate_compiler_receipt(_load_receipt(path), candidate)
        if receipt["request_reasoning_effort"] != COMPILER_REASONING_EFFORT:
            raise BoundedPilotCompilerAggregateError(
                "bounded compiler reasoning effort differs"
            )
        receipts.append(receipt)
        receipt_hashes.append(receipt["receipt_sha256"])
    if set(judgments_root.glob("*.compiler.json")) != expected_paths:
        raise BoundedPilotCompilerAggregateError("compiler receipt population differs")
    summary_hashes = validate_shard_summaries(candidates, judgments_root)
    summary_lineage = [
        {"source_id": row["source_id"], "stratum": row["source_type"]}
        for row in lineage
    ]
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_joint_evidence",
        "population": {
            "root_name": population_root.name,
            "receipt_file_sha256": sha256_file(population_root / "receipt.json"),
            "receipt_sha256": population["receipt_sha256"],
            "rows": len(candidates),
        },
        "rights_adjudication": {
            "root_name": rights_root.name,
            "receipt_file_sha256": sha256_file(rights_root / "receipt.json"),
            "receipt_sha256": rights["receipt_sha256"],
            "queue_sha256": rights["queue"]["sha256"],
        },
        "logical_shards": LOGICAL_SHARDS,
        "ordered_shard_summaries_sha256": canonical_sha256(summary_hashes),
        "ordered_compiler_receipts_sha256": canonical_sha256(receipt_hashes),
        "compiler_summary": summarize(summary_lineage, receipts),
        "joint_rights_and_model_routes": combine_rights_and_model_routes(
            lineage, receipts, rights_by_identity
        ),
        "compiler_judgments_are_verified_admissions": False,
        "independent_representation_verification_complete": False,
        "rights_provenance_verified": False,
        "legal_clearance_established": False,
        "full_reservoir_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--rights-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_aggregate(
        args.population_root,
        args.judgments_root,
        args.rights_root,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
