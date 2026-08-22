"""Adjudicate two complete blinded FineMath reviews with a frozen precision gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.authored_curriculum import _read_regular_bytes
from sai.data.finemath_filter_ladder import (
    CANDIDATE_SCHEMA,
    KEY_SCHEMA,
    STRATA,
    validate_ladder,
)
from sai.data.finemath_review_workspace import REVIEW_ROW_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-finemath-human-review-adjudication-v1"
MINIMUM_CONSENSUS_ACCEPT_PPM = 900_000
MINIMUM_WILSON_95_LCB_PPM = 800_000
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_REVIEW_KEYS = {
    "schema",
    "reviewer_id",
    "review_identity_sha256",
    "quality_decision",
    "mathematical_correctness",
    "instructional_structure",
    "self_contained",
    "english_clarity_ppm",
    "defects",
    "evidence_quotes",
}
_DEFECTS = {
    "answer_only",
    "duplicated_boilerplate",
    "incoherent_prose",
    "incorrect_math",
    "incomplete_solution",
    "low_value_repetition",
    "non_english_or_garbled",
    "relies_on_missing_context",
    "source_noise",
    "unsafe_or_advertising",
}


class FineMathReviewAdjudicationError(RuntimeError):
    """The blinded reviews, hidden key, candidate output, or decision differs."""


def _jsonl(path: Path, *, maximum_bytes: int) -> tuple[list[dict[str, Any]], bytes]:
    encoded = _read_regular_bytes(path, maximum_bytes=maximum_bytes)
    try:
        rows = [json.loads(line) for line in encoded.decode().splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FineMathReviewAdjudicationError("review artifact differs") from error
    if not rows:
        raise FineMathReviewAdjudicationError("review artifact differs")
    return rows, encoded


def _count(text: str, quote: str) -> int:
    return text.count(quote)


def _validate_review(
    path: Path,
    *,
    expected_sha256: str,
    packet: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], bytes]:
    if not _HEX64.fullmatch(expected_sha256) or sha256_file(path) != expected_sha256:
        raise FineMathReviewAdjudicationError("review file hash differs")
    rows, encoded = _jsonl(path, maximum_bytes=8 << 20)
    if len(rows) != len(packet):
        raise FineMathReviewAdjudicationError("review row population differs")
    reviewer_ids: set[str] = set()
    for row, source in zip(rows, packet, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != _REVIEW_KEYS
            or row["schema"] != REVIEW_ROW_SCHEMA
            or not isinstance(row["reviewer_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{3,64}", row["reviewer_id"])
            or row["review_identity_sha256"] != source["review_identity_sha256"]
            or row["quality_decision"] not in {"accept", "reject", "uncertain"}
            or row["mathematical_correctness"]
            not in {"correct", "incorrect", "uncertain"}
            or row["instructional_structure"]
            not in {"explanatory", "answer_only", "incoherent", "uncertain"}
            or not isinstance(row["self_contained"], bool)
            or isinstance(row["english_clarity_ppm"], bool)
            or not isinstance(row["english_clarity_ppm"], int)
            or not 0 <= row["english_clarity_ppm"] <= 1_000_000
            or not isinstance(row["defects"], list)
            or row["defects"] != sorted(set(row["defects"]))
            or any(value not in _DEFECTS for value in row["defects"])
            or not isinstance(row["evidence_quotes"], list)
            or not row["evidence_quotes"]
            or len(row["evidence_quotes"]) != len(set(row["evidence_quotes"]))
        ):
            raise FineMathReviewAdjudicationError("review row differs")
        reviewer_ids.add(row["reviewer_id"])
        text = source["text"]
        if any(
            not isinstance(quote, str) or len(quote) < 12 or _count(text, quote) != 1
            for quote in row["evidence_quotes"]
        ):
            raise FineMathReviewAdjudicationError("review evidence differs")
        if row["quality_decision"] == "accept" and (
            row["mathematical_correctness"] != "correct"
            or row["instructional_structure"] != "explanatory"
            or row["self_contained"] is not True
            or row["english_clarity_ppm"] < 800_000
            or row["defects"]
        ):
            raise FineMathReviewAdjudicationError("accepted review row differs")
        if row["quality_decision"] == "reject" and not row["defects"]:
            raise FineMathReviewAdjudicationError("rejected review row differs")
    if len(reviewer_ids) != 1:
        raise FineMathReviewAdjudicationError("reviewer identity differs")
    return reviewer_ids.pop(), rows, encoded


def _wilson_lcb_ppm(accepted: int, total: int) -> int:
    if total <= 0 or accepted < 0 or accepted > total:
        raise FineMathReviewAdjudicationError("review precision geometry differs")
    z = 1.959963984540054
    rate = accepted / total
    z2 = z * z
    center = rate + z2 / (2 * total)
    radius = z * math.sqrt((rate * (1 - rate) + z2 / (4 * total)) / total)
    lower = (center - radius) / (1 + z2 / total)
    return max(0, min(1_000_000, math.floor(lower * 1_000_000)))


def _descriptor(
    path: Path, rows: list[dict[str, Any]], encoded: bytes
) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "ordered_rows_sha256": canonical_sha256(rows),
    }


def _write_staged(path: Path, encoded: bytes) -> Path:
    stage = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with stage.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return stage


def adjudicate(
    *,
    ladder_receipt: Path,
    expected_ladder_receipt_sha256: str,
    review_a: Path,
    expected_review_a_sha256: str,
    review_b: Path,
    expected_review_b_sha256: str,
    selected_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    if (
        not _HEX64.fullmatch(expected_ladder_receipt_sha256)
        or sha256_file(ladder_receipt) != expected_ladder_receipt_sha256
    ):
        raise FineMathReviewAdjudicationError("ladder receipt hash differs")
    if selected_output.parent.resolve() != receipt_output.parent.resolve():
        raise FineMathReviewAdjudicationError(
            "adjudication outputs must share one parent"
        )
    if any(
        path.exists() or path.is_symlink() for path in (selected_output, receipt_output)
    ):
        raise FineMathReviewAdjudicationError("adjudication output already exists")
    try:
        ladder = validate_ladder(ladder_receipt)
    except Exception as error:
        raise FineMathReviewAdjudicationError(
            "FineMath ladder validation failed"
        ) from error
    packet, packet_encoded = _jsonl(
        Path(ladder["blind_review_output"]["path"]), maximum_bytes=32 << 20
    )
    reviewer_a, rows_a, encoded_a = _validate_review(
        review_a, expected_sha256=expected_review_a_sha256, packet=packet
    )
    reviewer_b, rows_b, encoded_b = _validate_review(
        review_b, expected_sha256=expected_review_b_sha256, packet=packet
    )
    if reviewer_a == reviewer_b:
        raise FineMathReviewAdjudicationError("independent reviewer identities differ")

    # The hidden score key is opened only after both complete blind reviews pass.
    key_rows, key_encoded = _jsonl(
        Path(ladder["review_key_output"]["path"]), maximum_bytes=2 << 20
    )
    if len(key_rows) != len(packet):
        raise FineMathReviewAdjudicationError("hidden review key differs")
    by_stratum: dict[str, Counter[str]] = {name: Counter() for name in STRATA}
    agreements = 0
    for source, key, left, right in zip(packet, key_rows, rows_a, rows_b, strict=True):
        if (
            not isinstance(key, dict)
            or set(key)
            != {
                "schema",
                "review_identity_sha256",
                "selection_rank_sha256",
                "stratum",
                "language_score_ppm",
            }
            or key["schema"] != KEY_SCHEMA
            or key["review_identity_sha256"] != source["review_identity_sha256"]
            or key["stratum"] not in STRATA
        ):
            raise FineMathReviewAdjudicationError("hidden review key differs")
        left_decision = left["quality_decision"]
        right_decision = right["quality_decision"]
        if left_decision == right_decision:
            agreements += 1
        consensus = (
            "accept" if left_decision == right_decision == "accept" else "not_accept"
        )
        by_stratum[key["stratum"]].update([consensus])

    metrics: dict[str, dict[str, Any]] = {}
    qualified: dict[str, bool] = {}
    for stratum in STRATA:
        total = sum(by_stratum[stratum].values())
        accepted = by_stratum[stratum]["accept"]
        accept_ppm = accepted * 1_000_000 // total
        lcb = _wilson_lcb_ppm(accepted, total)
        passed = (
            accept_ppm >= MINIMUM_CONSENSUS_ACCEPT_PPM
            and lcb >= MINIMUM_WILSON_95_LCB_PPM
        )
        metrics[stratum] = {
            "rows": total,
            "consensus_accepted": accepted,
            "consensus_accept_ppm": accept_ppm,
            "wilson_95_lcb_ppm": lcb,
            "passed": passed,
        }
        qualified[stratum] = passed

    options = [
        (0, STRATA),
        (900_000, ("0p90_to_0p95", "at_least_0p95")),
        (950_000, ("at_least_0p95",)),
    ]
    selected_floor = next(
        (floor for floor, strata in options if all(qualified[name] for name in strata)),
        None,
    )
    candidate_rows, _ = _jsonl(
        Path(ladder["candidate_output"]["path"]), maximum_bytes=32 << 20
    )
    if any(
        not isinstance(row, dict)
        or row.get("schema") != CANDIDATE_SCHEMA
        or not isinstance(row.get("language_score_ppm"), int)
        for row in candidate_rows
    ):
        raise FineMathReviewAdjudicationError("candidate population differs")
    selected_rows = (
        []
        if selected_floor is None
        else [
            row for row in candidate_rows if row["language_score_ppm"] >= selected_floor
        ]
    )
    selected_encoded = b"".join(
        (
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode()
        for row in selected_rows
    )
    consensus_floor = MINIMUM_CONSENSUS_ACCEPT_PPM
    lcb_floor = MINIMUM_WILSON_95_LCB_PPM
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "threshold_selected_candidate_not_admitted"
            if selected_floor is not None
            else "finemath_rejected_by_human_precision_gate"
        ),
        "ladder_receipt": {
            "file_sha256": expected_ladder_receipt_sha256,
            "receipt_sha256": ladder["receipt_sha256"],
            "blind_packet_sha256": hashlib.sha256(packet_encoded).hexdigest(),
            "hidden_key_sha256": hashlib.sha256(key_encoded).hexdigest(),
        },
        "reviews": [
            {
                "reviewer_id": reviewer_a,
                "file_sha256": hashlib.sha256(encoded_a).hexdigest(),
                "rows": len(rows_a),
            },
            {
                "reviewer_id": reviewer_b,
                "file_sha256": hashlib.sha256(encoded_b).hexdigest(),
                "rows": len(rows_b),
            },
        ],
        "policy": {
            "consensus_accept_requires_two_accept_labels": True,
            "minimum_consensus_accept_ppm_per_included_stratum": consensus_floor,
            "minimum_wilson_95_lcb_ppm_per_included_stratum": lcb_floor,
            "select_lowest_passing_language_floor": True,
            "candidate_floors_ppm": [0, 900_000, 950_000],
        },
        "review_agreement_ppm": agreements * 1_000_000 // len(packet),
        "stratum_metrics": metrics,
        "selected_minimum_language_score_ppm": selected_floor,
        "selected_output": _descriptor(
            selected_output, selected_rows, selected_encoded
        ),
        "limitations": [
            "selected_rows_require_global_near_deduplication",
            "selected_rows_require_benchmark_decontamination",
            "selected_rows_require_semantic_prerequisite_placement",
            "selected_rows_require_final_provenance_and_license_replay",
            "adjudication_authorizes_no_training",
        ],
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_encoded = json.dumps(payload, sort_keys=True, indent=2).encode() + b"\n"
    stages: list[Path] = []
    try:
        selected_output.parent.mkdir(parents=True, exist_ok=True)
        stages.append(_write_staged(selected_output, selected_encoded))
        stages.append(_write_staged(receipt_output, receipt_encoded))
        for stage, output in zip(
            stages, (selected_output, receipt_output), strict=True
        ):
            os.replace(stage, output)
    except BaseException:
        for stage in stages:
            stage.unlink(missing_ok=True)
        raise
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder-receipt", type=Path, required=True)
    parser.add_argument("--expected-ladder-receipt-sha256", required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--expected-review-a-sha256", required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--expected-review-b-sha256", required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    payload = adjudicate(**vars(parser.parse_args(argv)))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_minimum_language_score_ppm": payload[
                    "selected_minimum_language_score_ppm"
                ],
                "selected_rows": payload["selected_output"]["rows"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
