"""Aggregate exact three-perspective semantic audit judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_candidate_population import SCHEMA as POPULATION_SCHEMA
from sai.data.agent_labeling import (
    PERSPECTIVES,
    RUBRIC_SHA256,
    _atomic_create,
    aggregate_judgments,
    normalize_candidate,
)
from sai.data.nous_label_worker import SCHEMA as JUDGMENT_RECEIPT_SCHEMA
from sai.data.pleias_production_materializer import _load_signed
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-three-perspective-semantic-audit-aggregate-v1"
DECISION_SCHEMA = "sai-three-perspective-semantic-audit-decision-v1"
SUMMARY_SCHEMA = "sai-nous-agent-label-shard-summary-v1"


class SemanticAuditAggregateError(RuntimeError):
    """Population, receipt, perspective, or exact coverage differs."""


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise SemanticAuditAggregateError("candidate population is unsafe")
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(normalize_candidate(json.loads(line)))
            except (json.JSONDecodeError, RuntimeError) as error:
                raise SemanticAuditAggregateError(
                    f"candidate row {line_number} differs"
                ) from error
    identities = [row["candidate_identity_sha256"] for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise SemanticAuditAggregateError("candidate coverage differs")
    return rows


def _signed_receipt(path: Path, schema: str) -> dict[str, Any]:
    try:
        return _load_signed(path, schema)
    except RuntimeError as error:
        raise SemanticAuditAggregateError("signed receipt differs") from error


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        for row in rows:
            os.write(
                descriptor,
                (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode(),
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def build_aggregate(
    candidates_path: Path,
    population_receipt_path: Path,
    judgments_root: Path,
    output_root: Path,
    *,
    expected_model: str,
    logical_shards: int,
) -> dict[str, Any]:
    """Verify every slot and emit source-text-free conservative decisions."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not isinstance(expected_model, str)
        or not expected_model
        or isinstance(logical_shards, bool)
        or not 1 <= logical_shards <= 10_000
    ):
        raise SemanticAuditAggregateError("aggregate arguments differ")
    population = _signed_receipt(population_receipt_path, POPULATION_SCHEMA)
    candidates = _load_candidates(candidates_path)
    descriptor = population.get("population")
    if (
        population.get("status") != "complete"
        or not isinstance(descriptor, dict)
        or descriptor.get("rows") != len(candidates)
        or descriptor.get("bytes") != candidates_path.stat().st_size
        or descriptor.get("sha256") != sha256_file(candidates_path)
    ):
        raise SemanticAuditAggregateError("population receipt differs")
    assigned_counts = Counter(
        int(row["candidate_identity_sha256"], 16) % logical_shards for row in candidates
    )
    summaries = []
    for shard_index in range(logical_shards):
        summary = _signed_receipt(
            judgments_root / f"shard_{shard_index:05d}.summary.json",
            SUMMARY_SCHEMA,
        )
        if (
            summary.get("status") != "complete"
            or summary.get("model") != expected_model
            or summary.get("rubric_sha256") != RUBRIC_SHA256
            or summary.get("logical_shards") != logical_shards
            or summary.get("shard_index") != shard_index
            or summary.get("candidate_rows") != assigned_counts[shard_index]
            or summary.get("judgments_per_candidate") != len(PERSPECTIVES)
            or summary.get("expected_judgments")
            != assigned_counts[shard_index] * len(PERSPECTIVES)
            or summary.get("created_judgments", 0)
            + summary.get("preexisting_judgments", 0)
            != summary.get("expected_judgments")
            or summary.get("api_key_persisted") is not False
        ):
            raise SemanticAuditAggregateError("shard summary differs")
        summaries.append(summary["receipt_sha256"])
    decisions = []
    counts: Counter[str] = Counter()
    ordered_receipts = hashlib.sha256()
    for candidate in candidates:
        identity = candidate["candidate_identity_sha256"]
        judgments = []
        receipt_hashes = []
        for slot, perspective in enumerate(PERSPECTIVES):
            receipt = _signed_receipt(
                judgments_root / f"{identity}.slot{slot}.json",
                JUDGMENT_RECEIPT_SCHEMA,
            )
            if (
                receipt.get("status") != "complete"
                or receipt.get("candidate_identity_sha256") != identity
                or receipt.get("annotator_slot") != slot
                or receipt.get("perspective") != perspective
                or receipt.get("rubric_sha256") != RUBRIC_SHA256
                or receipt.get("requested_model") != expected_model
                or receipt.get("api_key_persisted") is not False
                or receipt.get("tools_enabled") is not False
                or not isinstance(receipt.get("judgment"), dict)
            ):
                raise SemanticAuditAggregateError("judgment receipt differs")
            judgments.append(receipt["judgment"])
            receipt_hashes.append(receipt["receipt_sha256"])
            ordered_receipts.update(bytes.fromhex(receipt["receipt_sha256"]))
            counts["judgments"] += 1
            for attempt in receipt.get("attempts", []):
                outcome = attempt.get("outcome") if isinstance(attempt, dict) else None
                if isinstance(outcome, str):
                    counts[f"attempt::{outcome}"] += 1
        aggregate = aggregate_judgments(candidate, judgments)
        decision = {
            "schema": DECISION_SCHEMA,
            "candidate_identity_sha256": identity,
            "source_content_sha256": candidate["source_content_sha256"],
            "provenance_sha256": candidate["provenance_sha256"],
            "source": candidate["source"],
            "ordered_judgment_receipt_sha256s": receipt_hashes,
            "aggregate": aggregate,
            "source_text_persisted": False,
            "training_ready": False,
        }
        decision["decision_sha256"] = canonical_sha256(decision)
        decisions.append(decision)
        disposition = aggregate["disposition"]
        counts["candidates"] += 1
        counts[f"disposition::{disposition}"] += 1
        counts[f"quality_median::{aggregate['quality_score_median']}"] += 1
        counts[f"english_median::{aggregate['english_score_median']}"] += 1
        counts[f"difficulty_median::{aggregate['difficulty_median']}"] += 1
        if aggregate["curriculum_phase"] is not None:
            counts[f"curriculum_phase::{aggregate['curriculum_phase']}"] += 1
        for risk in aggregate["blocking_risks"]:
            counts[f"blocking_risk::{risk}"] += 1
        for concept in aggregate["concepts_taught_consensus"]:
            counts[f"concept::{concept}"] += 1
    output_root.mkdir(parents=True)
    decision_path = output_root / "decisions.jsonl"
    _write_jsonl(decision_path, decisions)
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_three_perspective_semantic_audit",
        "source": {
            "population_receipt_sha256": population["receipt_sha256"],
            "candidate_file_sha256": sha256_file(candidates_path),
        },
        "model": expected_model,
        "rubric_sha256": RUBRIC_SHA256,
        "judgments_per_candidate": len(PERSPECTIVES),
        "logical_shards": logical_shards,
        "ordered_shard_summary_receipts_sha256": canonical_sha256(summaries),
        "ordered_judgment_receipts_sha256": ordered_receipts.hexdigest(),
        "counts": dict(sorted(counts.items())),
        "decisions": {
            "path": decision_path.name,
            "rows": len(decisions),
            "bytes": decision_path.stat().st_size,
            "sha256": sha256_file(decision_path),
            "ordered_decision_digests_sha256": canonical_sha256(
                [row["decision_sha256"] for row in decisions]
            ),
        },
        "complete_candidate_coverage": True,
        "complete_three_perspective_coverage": True,
        "source_text_persisted": False,
        "semantic_admission_decision_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--population-receipt", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--logical-shards", type=int, required=True)
    args = parser.parse_args()
    result = build_aggregate(
        args.candidates,
        args.population_receipt,
        args.judgments_root,
        args.output_root,
        expected_model=args.expected_model,
        logical_shards=args.logical_shards,
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
