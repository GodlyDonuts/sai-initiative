"""Aggregate the complete Institutional Books pilot without exposing excerpts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.book_compiler_labeling import RUBRIC_SHA256
from sai.data.institutional_books_pilot import RECEIPT_SCHEMA as POPULATION_SCHEMA
from sai.data.institutional_books_semantic_population import (
    SCHEMA as SEMANTIC_POPULATION_SCHEMA,
)
from sai.data.nous_book_compiler_worker import (
    RECEIPT_SCHEMA,
    SUMMARY_SCHEMA,
    _load_book_jsonl,
)
from sai.data.nous_label_worker import DEFAULT_MODEL
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-institutional-books-compiler-aggregate-v1"
DEFAULT_LOGICAL_SHARDS = 10_000
INDEPENDENT_POPULATION_SCHEMA = (
    "sai-institutional-books-independent-candidate-population-v1"
)


class InstitutionalBooksAggregateError(RuntimeError):
    """The book population, compiler receipt, or shard custody differs."""


def triage_route(judgment: dict[str, Any]) -> str:
    """Route a model judgment conservatively without treating it as admission."""

    risks = judgment.get("risks")
    quality = judgment.get("quality")
    if not isinstance(risks, dict) or not isinstance(quality, dict):
        raise InstitutionalBooksAggregateError("book judgment route differs")
    if judgment.get("verdict") == "reject":
        return "quarantine"
    if risks.get("rights_evidence_incomplete"):
        return "rights_hold"
    if risks.get("duplicate_or_near_duplicate_edition"):
        return "deduplication_review"
    if risks.get("ocr_damage") or quality.get("ocr_quality", 0) < 3:
        return "cleanup_review"
    if judgment.get("current_language") != "english":
        return "translation_review"
    if risks.get("factual_unreliability"):
        return "factual_grounding_review"
    if risks.get("outdated_or_harmful_claims"):
        return "historical_context_transformation"
    epistemic_score = max(
        quality.get("knowledge_density", 0),
        quality.get("literary_value", 0),
        quality.get("historical_value", 0),
    )
    if quality.get("overall_quality", 0) >= 3 and epistemic_score >= 3:
        return "representation_verification"
    return "quality_review"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise InstitutionalBooksAggregateError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InstitutionalBooksAggregateError(f"{label} cannot be decoded") from error
    if not isinstance(value, dict):
        raise InstitutionalBooksAggregateError(f"{label} differs")
    return value


def _validate_receipt(
    receipt: dict[str, Any],
    candidate: dict[str, Any],
    expected_model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    judgment = receipt.get("judgment")
    if not isinstance(judgment, dict):
        raise InstitutionalBooksAggregateError("book compiler judgment differs")
    judgment_unsigned = {
        key: value for key, value in judgment.items() if key != "judgment_sha256"
    }
    rights = candidate["bibliographic"]["rights_evidence"]
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("candidate_identity_sha256")
        != candidate["candidate_identity_sha256"]
        or receipt.get("requested_model") != expected_model
        or receipt.get("rubric_sha256") != RUBRIC_SHA256
        or receipt.get("request_reasoning_effort") != "low"
        or receipt.get("api_key_persisted") is not False
        or receipt.get("tools_enabled") is not False
        or receipt.get("raw_source_is_training_data") is not False
        or receipt.get("training_ready") is not False
        or judgment.get("schema") != "sai-institutional-book-compiler-judgment-v2"
        or judgment.get("judgment_sha256") != canonical_sha256(judgment_unsigned)
        or judgment.get("candidate_identity_sha256")
        != candidate["candidate_identity_sha256"]
        or judgment.get("rubric_sha256") != RUBRIC_SHA256
        or judgment.get("source_id") != candidate["source"]["barcode_src"]
        or judgment.get("rights_evidence") != rights
        or judgment.get("rights_are_model_inferred") is not False
        or judgment.get("raw_archive_source_is_training_ready") is not False
    ):
        raise InstitutionalBooksAggregateError("book compiler receipt differs")
    triage_route(judgment)
    return receipt


def _validate_population(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt_path = root / "receipt.json"
    receipt = _load_json(receipt_path, "book population receipt")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    output = receipt.get("output")
    pilot = receipt.get("schema") == POPULATION_SCHEMA
    semantic = receipt.get("schema") == SEMANTIC_POPULATION_SCHEMA
    independent = receipt.get("schema") == INDEPENDENT_POPULATION_SCHEMA
    common_valid = (
        receipt.get("receipt_sha256") == canonical_sha256(unsigned)
        and receipt.get("training_ready") is False
        and receipt.get("four_b_training_authorized") is False
        and isinstance(output, dict)
    )
    pilot_valid = (
        pilot
        and receipt.get("status") == "complete"
        and receipt.get("book_text_downloaded") is True
    )
    semantic_valid = (
        semantic
        and receipt.get("status")
        == "complete_nontraining_private_semantic_candidate_population"
        and receipt.get("source_text_private") is True
        and receipt.get("source_text_publishable") is False
        and receipt.get("semantic_admission_complete") is False
    )
    independent_valid = (
        independent
        and receipt.get("status")
        == "complete_nontraining_private_independent_book_candidate_population"
        and receipt.get("source_text_private") is True
        and receipt.get("source_text_publishable") is False
        and receipt.get("independent_verification_complete") is False
    )
    if not common_valid or not (pilot_valid or semantic_valid or independent_valid):
        raise InstitutionalBooksAggregateError("book population receipt differs")
    output_path = Path(str(output.get("path")))
    path = output_path if output_path.is_absolute() else root / output_path
    if (
        not path.resolve().is_relative_to(root.resolve())
        or path.parent.resolve() != root.resolve()
        or path.name != "candidates.jsonl"
        or path.stat().st_size != output.get("bytes")
        or sha256_file(path) != output.get("sha256")
    ):
        raise InstitutionalBooksAggregateError("book population bytes differ")
    expected_rows = (
        receipt.get("statistics", {}).get("candidate_rows")
        if pilot
        else output.get("rows")
    )
    candidates = (
        []
        if independent and expected_rows == 0 and path.stat().st_size == 0
        else _load_book_jsonl(path)
    )
    if len(candidates) != expected_rows:
        raise InstitutionalBooksAggregateError("book population coverage differs")
    return candidates, receipt


def build_aggregate(
    population_root: Path,
    judgments_root: Path,
    output_path: Path,
    logical_shards: int = DEFAULT_LOGICAL_SHARDS,
    expected_model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Replay every candidate, receipt, and nonempty shard into one summary."""

    if output_path.exists() or output_path.is_symlink():
        raise InstitutionalBooksAggregateError("book aggregate output differs")
    if (
        isinstance(logical_shards, bool)
        or not isinstance(logical_shards, int)
        or logical_shards <= 0
    ):
        raise InstitutionalBooksAggregateError("book logical shards differ")
    if not isinstance(expected_model, str) or not expected_model:
        raise InstitutionalBooksAggregateError("book expected model differs")
    candidates, population = _validate_population(population_root)
    expected_receipts = {
        judgments_root / f"{candidate['candidate_identity_sha256']}.book-compiler.json"
        for candidate in candidates
    }
    if set(judgments_root.glob("*.book-compiler.json")) != expected_receipts:
        raise InstitutionalBooksAggregateError("book receipt population differs")
    nonempty_shards = sorted(
        {
            int(candidate["candidate_identity_sha256"], 16) % logical_shards
            for candidate in candidates
        }
    )
    expected_summaries = {
        judgments_root / f"shard_{index:05d}.summary.json" for index in nonempty_shards
    }
    if set(judgments_root.glob("shard_*.summary.json")) != expected_summaries:
        raise InstitutionalBooksAggregateError("book shard population differs")
    receipts = []
    receipt_hashes = []
    streamed = 0
    for candidate in candidates:
        path = judgments_root / (
            f"{candidate['candidate_identity_sha256']}.book-compiler.json"
        )
        receipt = _validate_receipt(
            _load_json(path, "book compiler receipt"),
            candidate,
            expected_model,
        )
        receipts.append(receipt)
        receipt_hashes.append(receipt["receipt_sha256"])
        streamed += receipt.get("request_stream_transport") is True
    summary_hashes = []
    for index in nonempty_shards:
        summary = _load_json(
            judgments_root / f"shard_{index:05d}.summary.json", "book shard summary"
        )
        expected = sum(
            int(candidate["candidate_identity_sha256"], 16) % logical_shards == index
            for candidate in candidates
        )
        created = summary.get("created_judgments")
        preexisting = summary.get("preexisting_judgments")
        unsigned = {
            key: value for key, value in summary.items() if key != "receipt_sha256"
        }
        if (
            summary.get("schema") != SUMMARY_SCHEMA
            or summary.get("status") != "complete"
            or summary.get("receipt_sha256") != canonical_sha256(unsigned)
            or summary.get("model") != expected_model
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
            raise InstitutionalBooksAggregateError("book shard summary differs")
        summary_hashes.append(summary["receipt_sha256"])

    counters: dict[str, Counter[str]] = defaultdict(Counter)
    total_usage = Counter()
    explicit_edge_count = 0
    unique_edges = set()
    unique_concepts = set()
    for receipt in receipts:
        judgment = receipt["judgment"]
        counters["verdict"][judgment["verdict"]] += 1
        counters["triage_route"][triage_route(judgment)] += 1
        counters["current_language"][judgment["current_language"]] += 1
        counters["genre"][judgment["genre"]] += 1
        counters["style"][judgment["style"]] += 1
        counters["curriculum_band"][judgment["curriculum_band"]] += 1
        counters["translation_type"][judgment["translation_type"]] += 1
        counters["rights_status"][judgment["rights_evidence"]["status_code"]] += 1
        for domain in judgment["domains"]:
            counters["domain"][domain] += 1
        for representation in judgment["recommended_representations"]:
            counters["recommended_representation"][representation] += 1
        for risk, value in judgment["risks"].items():
            counters["risk"][risk] += value
        for concept in judgment["concepts"]:
            unique_concepts.add(concept)
        for prerequisite in judgment["prerequisites"]:
            unique_concepts.add(prerequisite)
        for edge in judgment["concept_edges"]:
            explicit_edge_count += 1
            unique_edges.add(
                (edge["prerequisite"], edge["dependent"], edge["relation"])
            )
        total_usage.update(receipt.get("usage", {}))
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_book_compiler_aggregate",
        "population": {
            "root_name": population_root.name,
            "receipt_file_sha256": sha256_file(population_root / "receipt.json"),
            "receipt_sha256": population["receipt_sha256"],
            "candidate_file_sha256": population["output"]["sha256"],
            "rows": len(candidates),
        },
        "logical_shards": logical_shards,
        "model": expected_model,
        "nonempty_shards": len(nonempty_shards),
        "ordered_shard_summaries_sha256": canonical_sha256(summary_hashes),
        "ordered_compiler_receipts_sha256": canonical_sha256(receipt_hashes),
        "stream_transport_receipts": streamed,
        "nonstream_transport_receipts": len(receipts) - streamed,
        "counts": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(counters.items())
        },
        "explicit_prerequisite_edge_claims": explicit_edge_count,
        "unique_explicit_prerequisite_edge_claims": len(unique_edges),
        "unique_model_concept_labels": len(unique_concepts),
        "usage": dict(sorted(total_usage.items())),
        "source_text_persisted": False,
        "evidence_quotes_persisted": False,
        "model_judgments_are_verified_admissions": False,
        "semantic_edges_verified": False,
        "rights_are_model_inferred": False,
        "independent_representation_verification_complete": False,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-shards", type=int, default=DEFAULT_LOGICAL_SHARDS)
    parser.add_argument("--expected-model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    result = build_aggregate(
        args.population_root,
        args.judgments_root,
        args.output,
        args.logical_shards,
        args.expected_model,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
