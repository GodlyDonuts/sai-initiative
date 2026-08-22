from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sai.data.finemath_review_adjudication as module
from sai.data.finemath_review_adjudication import (
    FineMathReviewAdjudicationError,
    adjudicate,
)


def _encoded(rows: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, reject_high: bool = False
) -> dict:
    packet = []
    keys = []
    candidates = []
    strata = ("below_0p90", "0p90_to_0p95", "at_least_0p95")
    scores = (500_000, 925_000, 975_000)
    for stratum, score in zip(strata, scores, strict=True):
        for index in range(64):
            identity = hashlib.sha256(f"{stratum}-{index}".encode()).hexdigest()
            text = f"A complete mathematical explanation for {stratum} example {index}."
            packet.append(
                {
                    "schema": "sai-finemath-language-ladder-blind-review-v1",
                    "review_identity_sha256": identity,
                    "source_url": "https://example.invalid",
                    "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "text": text,
                }
            )
            keys.append(
                {
                    "schema": "sai-finemath-language-ladder-key-v1",
                    "review_identity_sha256": identity,
                    "selection_rank_sha256": hashlib.sha256(
                        identity.encode()
                    ).hexdigest(),
                    "stratum": stratum,
                    "language_score_ppm": score,
                }
            )
            candidates.append(
                {
                    "schema": "sai-finemath-language-ladder-candidate-v1",
                    "identity_sha256": identity,
                    "language_score_ppm": score,
                }
            )
    paths = {}
    for name, rows in (("packet", packet), ("key", keys), ("candidate", candidates)):
        path = tmp_path / f"{name}.jsonl"
        path.write_bytes(_encoded(rows))
        paths[name] = path
    ladder_receipt = tmp_path / "ladder.json"
    ladder_receipt.write_text("{}\n")
    ladder = {
        "receipt_sha256": "a" * 64,
        "blind_review_output": {"path": str(paths["packet"])},
        "review_key_output": {"path": str(paths["key"])},
        "candidate_output": {"path": str(paths["candidate"])},
    }
    monkeypatch.setattr(module, "validate_ladder", lambda _: ladder)

    review_paths = []
    for reviewer in ("reviewer-a", "reviewer-b"):
        rows = []
        for packet_row, key in zip(packet, keys, strict=True):
            reject = reject_high and key["stratum"] == "at_least_0p95"
            rows.append(
                {
                    "schema": "sai-finemath-human-quality-review-v1",
                    "reviewer_id": reviewer,
                    "review_identity_sha256": packet_row["review_identity_sha256"],
                    "quality_decision": "reject" if reject else "accept",
                    "mathematical_correctness": "incorrect" if reject else "correct",
                    "instructional_structure": (
                        "incoherent" if reject else "explanatory"
                    ),
                    "self_contained": not reject,
                    "english_clarity_ppm": 100_000 if reject else 950_000,
                    "defects": ["incorrect_math"] if reject else [],
                    "evidence_quotes": ["complete mathematical explanation"],
                }
            )
        path = tmp_path / f"{reviewer}.jsonl"
        path.write_bytes(_encoded(rows))
        review_paths.append(path)
    return {
        "ladder_receipt": ladder_receipt,
        "expected_ladder_receipt_sha256": hashlib.sha256(
            ladder_receipt.read_bytes()
        ).hexdigest(),
        "review_a": review_paths[0],
        "expected_review_a_sha256": hashlib.sha256(
            review_paths[0].read_bytes()
        ).hexdigest(),
        "review_b": review_paths[1],
        "expected_review_b_sha256": hashlib.sha256(
            review_paths[1].read_bytes()
        ).hexdigest(),
        "selected_output": tmp_path / "selected.jsonl",
        "receipt_output": tmp_path / "decision.json",
    }


def test_selects_lowest_floor_only_after_two_complete_reviews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    payload = adjudicate(**inputs)
    assert payload["status"] == "threshold_selected_candidate_not_admitted"
    assert payload["selected_minimum_language_score_ppm"] == 0
    assert payload["selected_output"]["rows"] == 192
    assert all(row["passed"] for row in payload["stratum_metrics"].values())
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False


def test_rejects_source_when_no_floor_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch, reject_high=True)
    payload = adjudicate(**inputs)
    assert payload["status"] == "finemath_rejected_by_human_precision_gate"
    assert payload["selected_minimum_language_score_ppm"] is None
    assert payload["selected_output"]["rows"] == 0


def test_rejects_same_reviewer_and_review_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in inputs["review_b"].read_text().splitlines()]
    for row in rows:
        row["reviewer_id"] = "reviewer-a"
    inputs["review_b"].write_bytes(_encoded(rows))
    inputs["expected_review_b_sha256"] = hashlib.sha256(
        inputs["review_b"].read_bytes()
    ).hexdigest()
    with pytest.raises(FineMathReviewAdjudicationError, match="independent reviewer"):
        adjudicate(**inputs)
