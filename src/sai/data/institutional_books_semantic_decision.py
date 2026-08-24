"""Freeze conservative, text-free decisions from the book semantic screen."""

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
from sai.data.institutional_books_compiler_aggregate import (
    _load_json,
    _validate_population,
    _validate_receipt,
    triage_route,
)
from sai.data.institutional_books_quarantine_manifest import _load_aggregate
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-institutional-books-semantic-decision-receipt-v1"
RECORD_SCHEMA = "sai-institutional-books-semantic-decision-v1"
POLICY_SCHEMA = "sai-institutional-books-conservative-semantic-policy-v1"
ALLOWED_RIGHTS_CODES = ("cc-zero", "pd", "pdus")
TECHNICAL_GENRES = (
    "science",
    "mathematics",
    "engineering",
    "medicine",
    "law",
    "reference",
    "textbook_manual",
)
POLICY = {
    "schema": POLICY_SCHEMA,
    "allowed_rights_codes": list(ALLOWED_RIGHTS_CODES),
    "minimum_confidence_ppm": 900_000,
    "required_verdict": "retain",
    "required_route": "representation_verification",
    "required_current_language": "english",
    "minimum_overall_quality": 4,
    "minimum_ocr_quality": 4,
    "minimum_peak_epistemic_quality": 4,
    "technical_minimum_factual_reliability": 4,
    "maximum_active_risks": 0,
    "promotion_target": "independent_semantic_and_representation_verification",
    "promotion_is_training_admission": False,
}
POLICY_SHA256 = canonical_sha256(POLICY)


class InstitutionalBooksSemanticDecisionError(RuntimeError):
    """Book aggregate, judgment, or decision custody differs."""


def classify_judgment(judgment: dict[str, Any]) -> tuple[str, list[str]]:
    """Apply the frozen conservative policy to one validated judgment."""

    quality = judgment.get("quality")
    risks = judgment.get("risks")
    rights = judgment.get("rights_evidence")
    if (
        not isinstance(quality, dict)
        or not isinstance(risks, dict)
        or not isinstance(rights, dict)
    ):
        raise InstitutionalBooksSemanticDecisionError("book semantic judgment differs")
    route = triage_route(judgment)
    if route == "quarantine":
        return "quarantine", ["compiler_reject"]
    reasons = []
    if rights.get("status_code") not in ALLOWED_RIGHTS_CODES:
        reasons.append("rights_code_not_allowed")
    if risks.get("rights_evidence_incomplete"):
        reasons.append("rights_evidence_incomplete")
    if reasons:
        return "rights_hold", sorted(set(reasons))
    active_risks = sorted(key for key, value in risks.items() if value)
    if active_risks:
        return "risk_hold", [f"active_risk:{key}" for key in active_risks]
    if judgment.get("current_language") != "english":
        return "translation_hold", ["current_language_not_english"]
    if judgment.get("verdict") != "retain":
        reasons.append("verdict_not_retain")
    if route != "representation_verification":
        reasons.append("route_not_representation_verification")
    if judgment.get("confidence_ppm", 0) < POLICY["minimum_confidence_ppm"]:
        reasons.append("confidence_below_floor")
    if quality.get("overall_quality", 0) < POLICY["minimum_overall_quality"]:
        reasons.append("overall_quality_below_floor")
    if quality.get("ocr_quality", 0) < POLICY["minimum_ocr_quality"]:
        reasons.append("ocr_quality_below_floor")
    epistemic = max(
        quality.get("knowledge_density", 0),
        quality.get("literary_value", 0),
        quality.get("historical_value", 0),
    )
    if epistemic < POLICY["minimum_peak_epistemic_quality"]:
        reasons.append("epistemic_quality_below_floor")
    if (
        judgment.get("genre") in TECHNICAL_GENRES
        and quality.get("factual_reliability", 0)
        < POLICY["technical_minimum_factual_reliability"]
    ):
        reasons.append("technical_factual_reliability_below_floor")
    if reasons:
        return "quality_hold", sorted(set(reasons))
    return "independent_verification", []


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise InstitutionalBooksSemanticDecisionError("decision output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial.{uuid.uuid4().hex}"
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
        )
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_decisions(
    population_root: Path,
    judgments_root: Path,
    aggregate_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Replay complete semantic evidence and emit no source text."""

    if output_root.exists() or output_root.is_symlink():
        raise InstitutionalBooksSemanticDecisionError("decision root exists")
    aggregate = _load_aggregate(aggregate_path)
    try:
        candidates, population = _validate_population(population_root)
    except RuntimeError as error:
        raise InstitutionalBooksSemanticDecisionError(
            "book population differs"
        ) from error
    if aggregate.get("population", {}).get("rows") != len(candidates):
        raise InstitutionalBooksSemanticDecisionError(
            "book aggregate population differs"
        )
    expected = {
        judgments_root / f"{candidate['candidate_identity_sha256']}.book-compiler.json"
        for candidate in candidates
    }
    if set(judgments_root.glob("*.book-compiler.json")) != expected:
        raise InstitutionalBooksSemanticDecisionError(
            "book judgment population differs"
        )
    records = []
    counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for candidate in candidates:
        identity = candidate["candidate_identity_sha256"]
        try:
            receipt = _validate_receipt(
                _load_json(
                    judgments_root / f"{identity}.book-compiler.json",
                    "book compiler receipt",
                ),
                candidate,
            )
        except RuntimeError as error:
            raise InstitutionalBooksSemanticDecisionError(
                "book judgment differs"
            ) from error
        judgment = receipt["judgment"]
        disposition, reasons = classify_judgment(judgment)
        token_count = candidate["measurements"]["token_count_o200k_base_gen"]
        if isinstance(token_count, bool) or not isinstance(token_count, int):
            raise InstitutionalBooksSemanticDecisionError("book token count differs")
        record = {
            "schema": RECORD_SCHEMA,
            "candidate_identity_sha256": identity,
            "source_book_id": candidate["source"]["barcode_src"],
            "source_content_sha256": candidate["source_content_sha256"],
            "source_provenance_sha256": candidate["provenance_sha256"],
            "compiler_receipt_sha256": receipt["receipt_sha256"],
            "compiler_judgment_sha256": judgment["judgment_sha256"],
            "policy_sha256": POLICY_SHA256,
            "disposition": disposition,
            "reasons": reasons,
            "genre": judgment["genre"],
            "domains": judgment["domains"],
            "curriculum_band": judgment["curriculum_band"],
            "quality": judgment["quality"],
            "confidence_ppm": judgment["confidence_ppm"],
            "token_count_o200k_base_gen": token_count,
            "independent_verification_complete": False,
            "source_text_persisted": False,
            "training_ready": False,
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)
        counts[disposition] += 1
        tokens[disposition] += token_count
    if len(records) != len(candidates):
        raise InstitutionalBooksSemanticDecisionError("book decision coverage differs")
    records.sort(key=lambda row: row["candidate_identity_sha256"])
    output_root.mkdir(parents=True)
    try:
        manifest_path = output_root / "decisions.jsonl"
        _atomic_jsonl(manifest_path, records)
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_conservative_book_semantic_decision",
            "aggregate": {
                "receipt_sha256": aggregate["receipt_sha256"],
                "file_sha256": sha256_file(aggregate_path),
            },
            "population": {
                "receipt_sha256": population["receipt_sha256"],
                "rows": len(candidates),
            },
            "policy": POLICY,
            "policy_sha256": POLICY_SHA256,
            "counts": dict(sorted(counts.items())),
            "tokens": dict(sorted(tokens.items())),
            "manifest": {
                "path": manifest_path.name,
                "rows": len(records),
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in records]
                ),
            },
            "promoted_rows_are_training_admissions": False,
            "source_text_persisted": False,
            "independent_verification_complete": False,
            "benchmark_decontamination_complete": False,
            "global_semantic_deduplication_complete": False,
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
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_decisions(
        args.population_root,
        args.judgments_root,
        args.aggregate,
        args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
