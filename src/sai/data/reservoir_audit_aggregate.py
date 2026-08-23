"""Aggregate exact Hermes judgments for a sealed reservoir audit population."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create, normalize_candidate
from sai.data.data_compiler_labeling import (
    JUDGMENT_SCHEMA,
    RISK_KEYS,
    RUBRIC_SHA256,
    SCORE_KEYS,
    validate_normalized_judgment,
)
from sai.data.nous_compiler_worker import COMPILER_REASONING_EFFORT, RECEIPT_SCHEMA
from sai.data.nous_label_worker import DEFAULT_MODEL
from sai.data.reservoir_audit_population import (
    LINEAGE_SCHEMA,
)
from sai.data.reservoir_audit_population import (
    SCHEMA as POPULATION_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-reservoir-hermes-audit-aggregate-v1"


class ReservoirAuditAggregateError(RuntimeError):
    """Population evidence, compiler custody, or aggregate differs."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise ReservoirAuditAggregateError("audit input is missing or unsafe")
    try:
        rows = [json.loads(line) for line in path.open()]
    except (OSError, json.JSONDecodeError) as error:
        raise ReservoirAuditAggregateError("audit input cannot be decoded") from error
    if not rows:
        raise ReservoirAuditAggregateError("audit input is empty")
    return rows


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_population(
    population_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates_path = population_root / "candidates.jsonl"
    lineage_path = population_root / "lineage.jsonl"
    receipt_path = population_root / "receipt.json"
    candidates = [normalize_candidate(row) for row in _load_jsonl(candidates_path)]
    lineage = _load_jsonl(lineage_path)
    receipt_rows = _load_jsonl(receipt_path)
    if len(receipt_rows) != 1:
        raise ReservoirAuditAggregateError("population receipt is duplicated")
    receipt = receipt_rows[0]
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if (
        receipt.get("schema") != POPULATION_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("hermes_judgments_complete") is not False
        or receipt.get("training_ready") is not False
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned_receipt)
        or receipt.get("population", {}).get("path") != candidates_path.name
        or receipt.get("population", {}).get("rows") != len(candidates)
        or receipt.get("population", {}).get("bytes") != candidates_path.stat().st_size
        or receipt.get("population", {}).get("sha256") != sha256_file(candidates_path)
        or receipt.get("population", {}).get("ordered_identities_sha256")
        != canonical_sha256(identities)
        or receipt.get("lineage", {}).get("path") != lineage_path.name
        or receipt.get("lineage", {}).get("rows") != len(lineage)
        or receipt.get("lineage", {}).get("bytes") != lineage_path.stat().st_size
        or receipt.get("lineage", {}).get("sha256") != sha256_file(lineage_path)
        or receipt.get("lineage", {}).get("ordered_rows_sha256")
        != canonical_sha256(lineage)
        or len(candidates) != len(lineage)
        or len(identities) != len(set(identities))
    ):
        raise ReservoirAuditAggregateError("population receipt differs")
    for ordinal, (candidate, source) in enumerate(
        zip(candidates, lineage, strict=True)
    ):
        unsigned_lineage = {
            key: value for key, value in source.items() if key != "lineage_sha256"
        }
        if (
            source.get("schema") != LINEAGE_SCHEMA
            or source.get("ordinal") != ordinal
            or source.get("candidate_identity_sha256")
            != candidate["candidate_identity_sha256"]
            or source.get("repository") != candidate["source"]["dataset"]
            or source.get("revision") != candidate["source"]["revision"]
            or source.get("license") != candidate["source"]["license"]
            or source.get("excerpt_sha256") != candidate["source_content_sha256"]
            or source.get("excerpt_bytes") != len(candidate["text"].encode())
            or source.get("raw_source_is_training_ready") is not False
            or source.get("lineage_sha256") != canonical_sha256(unsigned_lineage)
        ):
            raise ReservoirAuditAggregateError("population lineage differs")
    return candidates, lineage, receipt


def _validate_compiler_receipt(
    receipt: Any, candidate: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ReservoirAuditAggregateError("compiler receipt differs")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("candidate_identity_sha256")
        != candidate["candidate_identity_sha256"]
        or receipt.get("rubric_sha256") != RUBRIC_SHA256
        or receipt.get("requested_model") != DEFAULT_MODEL
        or receipt.get("request_reasoning_effort") != COMPILER_REASONING_EFFORT
        or receipt.get("api_key_persisted") is not False
        or receipt.get("tools_enabled") is not False
        or receipt.get("raw_source_is_training_data") is not False
        or receipt.get("training_ready") is not False
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise ReservoirAuditAggregateError("compiler receipt identity differs")
    judgment = validate_normalized_judgment(receipt.get("judgment"), candidate)
    if (
        judgment.get("schema") != JUDGMENT_SCHEMA
        or judgment.get("judgment_sha256")
        != canonical_sha256(
            {key: value for key, value in judgment.items() if key != "judgment_sha256"}
        )
        or receipt.get("raw_model_json_sha256") is None
        or not _valid_sha256(receipt.get("raw_model_json_sha256"))
    ):
        raise ReservoirAuditAggregateError("compiler judgment custody differs")
    attempt_hashes = receipt.get("attempt_request_sha256s")
    attempts = receipt.get("attempts")
    if (
        not isinstance(attempt_hashes, list)
        or not attempt_hashes
        or any(not _valid_sha256(value) for value in attempt_hashes)
        or not isinstance(attempts, list)
        or len(attempts) != len(attempt_hashes)
        or receipt.get("successful_request_sha256") != attempt_hashes[-1]
    ):
        raise ReservoirAuditAggregateError("compiler attempt custody differs")
    return receipt


def _nested_counts(
    lineage: list[dict[str, Any]], judgments: list[dict[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for source, judgment in zip(lineage, judgments, strict=True):
        counts[source[field]][judgment["verdict"]] += 1
    return {
        key: dict(sorted(counter.items())) for key, counter in sorted(counts.items())
    }


def summarize(
    lineage: list[dict[str, Any]], receipts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Summarize model judgments without promoting them to verified data."""

    if not lineage or len(lineage) != len(receipts):
        raise ReservoirAuditAggregateError("aggregate inputs differ")
    judgments = [receipt["judgment"] for receipt in receipts]
    counter_fields = {
        "verdict": Counter(),
        "epistemic_functions": Counter(),
        "domains": Counter(),
        "curriculum_phase": Counter(),
        "source_language": Counter(),
        "translation_disposition": Counter(),
        "preservation_policy": Counter(),
        "recommended_representations": Counter(),
        "style": Counter(),
        "likely_origin": Counter(),
        "grounding_type": Counter(),
        "difficulty": Counter(),
        "prerequisite_burden": Counter(),
        "risks": Counter(),
        "cross_domain_bridges": Counter(),
    }
    score_sums = Counter()
    bridge_rows = 0
    potential_translation_rows = 0
    for judgment in judgments:
        for key in (
            "verdict",
            "curriculum_phase",
            "source_language",
            "translation_disposition",
            "preservation_policy",
            "style",
            "likely_origin",
            "grounding_type",
            "difficulty",
            "prerequisite_burden",
        ):
            counter_fields[key][str(judgment[key])] += 1
        for key in (
            "epistemic_functions",
            "domains",
            "recommended_representations",
            "cross_domain_bridges",
        ):
            counter_fields[key].update(judgment[key])
        counter_fields["risks"].update(
            key for key in RISK_KEYS if judgment["risks"][key]
        )
        score_sums.update(judgment["scores"])
        bridge_rows += bool(judgment["cross_domain_bridges"])
        potential_translation_rows += (
            judgment["verdict"] != "reject"
            and judgment["source_language"] != "english"
            and judgment["translation_priority"] > 0
        )
    usage = Counter()
    attempt_outcomes = Counter()
    repaired_rows = 0
    for receipt in receipts:
        for key, value in receipt["usage"].items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] += value
        attempt_outcomes.update(attempt["outcome"] for attempt in receipt["attempts"])
        repaired_rows += len(receipt["attempts"]) > 1
    return {
        "rows": len(judgments),
        "by_source_verdict": _nested_counts(lineage, judgments, "source_id"),
        "by_stratum_verdict": _nested_counts(lineage, judgments, "stratum"),
        "counts": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(counter_fields.items())
        },
        "mean_scores_milli": {
            key: (score_sums[key] * 1000) // len(judgments) for key in SCORE_KEYS
        },
        "rows_with_cross_domain_bridges": bridge_rows,
        "potential_translation_rows": potential_translation_rows,
        "usage": dict(sorted(usage.items())),
        "attempt_outcomes": dict(sorted(attempt_outcomes.items())),
        "rows_requiring_repair": repaired_rows,
        "model_judgments_are_verified_admissions": False,
    }


def build_aggregate(
    population_root: Path, judgments_root: Path, output_path: Path
) -> dict[str, Any]:
    """Replay every candidate and compiler receipt, then publish safe statistics."""

    if output_path.exists() or output_path.is_symlink():
        raise ReservoirAuditAggregateError("aggregate output already exists")
    candidates, lineage, population_receipt = _load_population(population_root)
    receipts = []
    receipt_hashes = []
    for candidate in candidates:
        identity = candidate["candidate_identity_sha256"]
        path = judgments_root / f"{identity}.compiler.json"
        rows = _load_jsonl(path)
        if len(rows) != 1:
            raise ReservoirAuditAggregateError("compiler receipt is duplicated")
        receipt = _validate_compiler_receipt(rows[0], candidate)
        receipts.append(receipt)
        receipt_hashes.append(receipt["receipt_sha256"])
    expected_paths = {
        judgments_root / f"{candidate['candidate_identity_sha256']}.compiler.json"
        for candidate in candidates
    }
    actual_paths = set(judgments_root.glob("*.compiler.json"))
    if actual_paths != expected_paths:
        raise ReservoirAuditAggregateError("compiler receipt population differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "population_receipt_sha256": population_receipt["receipt_sha256"],
        "population_file_sha256": sha256_file(population_root / "candidates.jsonl"),
        "lineage_file_sha256": sha256_file(population_root / "lineage.jsonl"),
        "rubric_sha256": RUBRIC_SHA256,
        "requested_model": DEFAULT_MODEL,
        "ordered_compiler_receipts_sha256": canonical_sha256(receipt_hashes),
        "summary": summarize(lineage, receipts),
        "coverage_first_not_statistical_acceptance_estimate": True,
        "independent_factual_verification_complete": False,
        "cross_source_deduplication_complete": False,
        "benchmark_decontamination_complete": False,
        "training_ready": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_aggregate(args.population_root, args.judgments_root, args.output)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
