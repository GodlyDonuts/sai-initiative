"""Evaluate the frozen OpenCoder code-web promotion screen without new calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.data_compiler_labeling import RUBRIC_SHA256
from sai.data.nous_compiler_worker import (
    SUMMARY_SCHEMA,
    _resume_completed_shard,
)
from sai.data.nous_label_worker import DEFAULT_MODEL
from sai.data.reservoir_audit_aggregate import (
    _load_jsonl,
    _triage_route,
    _validate_compiler_receipt,
    load_population,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-opencoder-code-web-promotion-screen-v1"
LOGICAL_SHARDS = 128
SCREEN_SHARDS = tuple((*range(64, 72), *range(96, 104)))
EXPECTED_SCREEN_ROWS = 276
PINNED_POPULATION_FILE_SHA256 = (
    "3cf1a97021a22f8a2dbab932c0bbf58ac724bd49b03c679aa61d447126e46182"
)
PINNED_POPULATION_RECEIPT_SHA256 = (
    "53abfd09fb2bc71b17dba5b922c1eaa2c7752cb216654e1557b442701937e7c9"
)
MINIMUM_REPRESENTATION_MILLI = 300
MAXIMUM_QUARANTINE_MILLI = 250
MINIMUM_COMPUTER_SCIENCE_MILLI = 600
MINIMUM_EDUCATIONAL_VALUE_MILLI = 2500
MINIMUM_TECHNICAL_DEPTH_MILLI = 2500


class OpenCoderPromotionScreenError(RuntimeError):
    """The frozen screen population, receipt custody, or decision differs."""


def _assigned(identity: str, shard_index: int) -> bool:
    return int(identity, 16) % LOGICAL_SHARDS == shard_index


def _metric_gates(
    *,
    rows: int,
    representation_rows: int,
    quarantine_rows: int,
    computer_science_rows: int,
    educational_value_sum: int,
    technical_depth_sum: int,
) -> dict[str, dict[str, Any]]:
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                representation_rows,
                quarantine_rows,
                computer_science_rows,
                educational_value_sum,
                technical_depth_sum,
            )
        )
        or representation_rows > rows
        or quarantine_rows > rows
        or computer_science_rows > rows
    ):
        raise OpenCoderPromotionScreenError("promotion metric geometry differs")

    def fraction_gate(
        numerator: int, threshold_milli: int, comparison: str
    ) -> dict[str, Any]:
        passed = (
            numerator * 1000 >= rows * threshold_milli
            if comparison == "minimum"
            else numerator * 1000 <= rows * threshold_milli
        )
        return {
            "numerator": numerator,
            "denominator": rows,
            "observed_milli": (numerator * 1000) // rows,
            "threshold_milli": threshold_milli,
            "comparison": comparison,
            "passed": passed,
        }

    def mean_gate(total: int, threshold_milli: int) -> dict[str, Any]:
        return {
            "score_sum": total,
            "rows": rows,
            "observed_mean_milli": (total * 1000) // rows,
            "threshold_mean_milli": threshold_milli,
            "comparison": "minimum",
            "passed": total * 1000 >= rows * threshold_milli,
        }

    return {
        "representation_verification": fraction_gate(
            representation_rows, MINIMUM_REPRESENTATION_MILLI, "minimum"
        ),
        "quarantine": fraction_gate(
            quarantine_rows, MAXIMUM_QUARANTINE_MILLI, "maximum"
        ),
        "computer_science": fraction_gate(
            computer_science_rows, MINIMUM_COMPUTER_SCIENCE_MILLI, "minimum"
        ),
        "educational_value": mean_gate(
            educational_value_sum, MINIMUM_EDUCATIONAL_VALUE_MILLI
        ),
        "technical_depth": mean_gate(
            technical_depth_sum, MINIMUM_TECHNICAL_DEPTH_MILLI
        ),
    }


def summarize_screen(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the frozen gate using integer comparisons only."""

    if len(receipts) != EXPECTED_SCREEN_ROWS:
        raise OpenCoderPromotionScreenError("promotion screen row count differs")
    routes = Counter()
    verdicts = Counter()
    domains = Counter()
    computer_science_rows = 0
    educational_value_sum = 0
    technical_depth_sum = 0
    usage = Counter()
    attempts = Counter()
    for receipt in receipts:
        judgment = receipt["judgment"]
        routes[_triage_route(judgment)] += 1
        verdicts[judgment["verdict"]] += 1
        domains.update(judgment["domains"])
        computer_science_rows += "computer_science" in judgment["domains"]
        educational_value_sum += judgment["scores"]["educational_value"]
        technical_depth_sum += judgment["scores"]["technical_depth"]
        usage.update(
            {
                key: value
                for key, value in receipt["usage"].items()
                if isinstance(value, int) and not isinstance(value, bool)
            }
        )
        attempts.update(row["outcome"] for row in receipt["attempts"])
    gates = _metric_gates(
        rows=len(receipts),
        representation_rows=routes["representation_verification"],
        quarantine_rows=routes["quarantine"],
        computer_science_rows=computer_science_rows,
        educational_value_sum=educational_value_sum,
        technical_depth_sum=technical_depth_sum,
    )
    passed = all(value["passed"] for value in gates.values())
    return {
        "rows": len(receipts),
        "conservative_triage_routes": dict(sorted(routes.items())),
        "verdicts": dict(sorted(verdicts.items())),
        "domains": dict(sorted(domains.items())),
        "usage": dict(sorted(usage.items())),
        "attempt_outcomes": dict(sorted(attempts.items())),
        "gates": gates,
        "all_gates_passed": passed,
        "decision": (
            "promote_full_2048_row_audit"
            if passed
            else "stop_full_audit_and_reallocate_hermes_capacity"
        ),
    }


def build_screen(
    population_root: Path, judgments_root: Path, output_path: Path
) -> dict[str, Any]:
    """Replay the exact selected shards and publish a source-text-free decision."""

    if output_path.exists() or output_path.is_symlink():
        raise OpenCoderPromotionScreenError("promotion screen output already exists")
    candidates, _lineage, population = load_population(population_root)
    candidates_file = population_root / "candidates.jsonl"
    if (
        sha256_file(candidates_file) != PINNED_POPULATION_FILE_SHA256
        or population.get("receipt_sha256") != PINNED_POPULATION_RECEIPT_SHA256
    ):
        raise OpenCoderPromotionScreenError("pinned OpenCoder population differs")
    by_shard = {
        shard_index: [
            row
            for row in candidates
            if _assigned(row["candidate_identity_sha256"], shard_index)
        ]
        for shard_index in SCREEN_SHARDS
    }
    selected = [row for shard_index in SCREEN_SHARDS for row in by_shard[shard_index]]
    if (
        len(selected) != EXPECTED_SCREEN_ROWS
        or len({row["candidate_identity_sha256"] for row in selected})
        != EXPECTED_SCREEN_ROWS
    ):
        raise OpenCoderPromotionScreenError("frozen screen identity coverage differs")

    summary_receipts = []
    receipts = []
    for shard_index in SCREEN_SHARDS:
        summary = _resume_completed_shard(
            judgments_root / f"shard_{shard_index:05d}.summary.json",
            judgments_root,
            by_shard[shard_index],
            model=DEFAULT_MODEL,
            logical_shards=LOGICAL_SHARDS,
            shard_index=shard_index,
            summary_schema=SUMMARY_SCHEMA,
            rubric_sha256=RUBRIC_SHA256,
            output_suffix="compiler",
        )
        if summary is None:
            raise OpenCoderPromotionScreenError("promotion screen shard is incomplete")
        summary_receipts.append(summary["receipt_sha256"])
        for candidate in by_shard[shard_index]:
            identity = candidate["candidate_identity_sha256"]
            rows = _load_jsonl(judgments_root / f"{identity}.compiler.json")
            if len(rows) != 1:
                raise OpenCoderPromotionScreenError("compiler receipt is duplicated")
            receipts.append(_validate_compiler_receipt(rows[0], candidate))

    result = summarize_screen(receipts)
    payload = {
        "schema": SCHEMA,
        "status": "complete_source_safe_nontraining_promotion_screen",
        "population": {
            "root_name": population_root.name,
            "candidate_file_sha256": PINNED_POPULATION_FILE_SHA256,
            "population_receipt_sha256": PINNED_POPULATION_RECEIPT_SHA256,
        },
        "logical_shards": LOGICAL_SHARDS,
        "screen_shards": list(SCREEN_SHARDS),
        "screen_rows": EXPECTED_SCREEN_ROWS,
        "ordered_candidate_identities_sha256": canonical_sha256(
            [row["candidate_identity_sha256"] for row in selected]
        ),
        "ordered_shard_summary_receipts_sha256": canonical_sha256(summary_receipts),
        "ordered_compiler_receipts_sha256": canonical_sha256(
            [row["receipt_sha256"] for row in receipts]
        ),
        "result": result,
        "screen_was_frozen_before_results": True,
        "teacher_judgments_are_verified_admissions": False,
        "publication_contains_source_text": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_screen(args.population_root, args.judgments_root, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "screen_rows": result["screen_rows"],
                "decision": result["result"]["decision"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
