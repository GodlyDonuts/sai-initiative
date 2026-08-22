from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.authored_review_compile import (
    DRAFT_SCHEMA,
    AuthoredReviewCompileError,
    compile_review,
    validate_compiled_review,
)
from sai.data.token_stream import canonical_sha256

ROOT = Path(__file__).parents[1]
ARTIFACT = ROOT / "artifacts" / "authored-curriculum-sources-r1"


def _draft(tmp_path: Path) -> tuple[Path, list[dict]]:
    packet = [
        json.loads(line)
        for line in (ARTIFACT / "authored-curriculum-blind-review.jsonl")
        .read_text()
        .splitlines()
    ]
    rows = [
        {
            "schema": DRAFT_SCHEMA,
            "review_identity_sha256": source["review_identity_sha256"],
            "instructional_quality_ppm": 900_000,
            "assumed_prior_concepts": [],
            "taught_concepts": [],
            "defects": [],
            "admission_recommendation": "revise",
        }
        for source in packet
    ]
    quote = next(
        line
        for line in packet[0]["text"].splitlines()
        if len(line) >= 24 and packet[0]["text"].count(line) == 1
    )
    rows[0]["taught_concepts"] = [
        {
            "concept_id": "code.literal",
            "confidence_ppm": 900_000,
            "evidence_quotes": [quote],
        }
    ]
    rows[0]["admission_recommendation"] = "admit"
    path = tmp_path / "draft.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path, packet


def _inputs(tmp_path: Path) -> dict[str, Path]:
    draft, _ = _draft(tmp_path)
    identity = tmp_path / "reviewer-identity.json"
    identity.write_text('{"reviewer":"candidate-test-reviewer"}\n')
    return {
        "candidate": ARTIFACT / "authored-curriculum-candidates.jsonl",
        "candidate_receipt": ARTIFACT / "authored-curriculum-receipt.json",
        "review_packet": ARTIFACT / "authored-curriculum-blind-review.jsonl",
        "review_key": ARTIFACT / "authored-curriculum-review-key.jsonl",
        "review_packet_receipt": ARTIFACT / "authored-curriculum-review-receipt.json",
        "concept_list": ROOT
        / "docs"
        / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json",
        "annotation_policy": ROOT / "docs" / "SAI_SEMANTIC_ANNOTATION_POLICY.json",
        "reviewer_identity": identity,
        "draft": draft,
        "output": tmp_path / "compiled.jsonl",
        "receipt_output": tmp_path / "compiled-receipt.json",
    }


def test_compile_review_resolves_unique_quotes_and_remains_candidate_only(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    payload = compile_review(**inputs)
    assert payload["status"] == "compiled_candidate_review"
    assert payload["audit_qualified"] is False
    assert payload["human_review_completed"] is False
    assert payload["training_authorized"] is False
    rows = [json.loads(line) for line in inputs["output"].read_text().splitlines()]
    source = json.loads(inputs["review_packet"].read_text().splitlines()[0])
    span = rows[0]["taught_concepts"][0]["evidence_spans"][0]
    quote = json.loads(inputs["draft"].read_text().splitlines()[0])["taught_concepts"][
        0
    ]["evidence_quotes"][0]
    assert source["text"][span["start"] : span["end"]] == quote
    assert validate_compiled_review(**inputs) == payload


@pytest.mark.parametrize("replacement", ["missing quote text", "the"])
def test_compile_review_rejects_missing_or_short_quote(
    tmp_path: Path, replacement: str
) -> None:
    inputs = _inputs(tmp_path)
    rows = [json.loads(line) for line in inputs["draft"].read_text().splitlines()]
    rows[0]["taught_concepts"][0]["evidence_quotes"] = [replacement]
    inputs["draft"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    with pytest.raises(AuthoredReviewCompileError, match="quote"):
        compile_review(**inputs)


def test_compile_review_rejects_ambiguous_quote(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    packet = [
        json.loads(line) for line in inputs["review_packet"].read_text().splitlines()
    ]
    rows = [json.loads(line) for line in inputs["draft"].read_text().splitlines()]
    words = packet[0]["text"].split()
    quote = next(
        " ".join(words[index : index + 4])
        for index in range(len(words) - 3)
        if len(" ".join(words[index : index + 4])) >= 16
        and packet[0]["text"].count(" ".join(words[index : index + 4])) > 1
    )
    rows[0]["taught_concepts"][0]["evidence_quotes"] = [quote]
    inputs["draft"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    with pytest.raises(AuthoredReviewCompileError, match="ambiguous"):
        compile_review(**inputs)


def test_compile_review_rejects_assumed_and_taught_overlap(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    rows = [json.loads(line) for line in inputs["draft"].read_text().splitlines()]
    rows[0]["assumed_prior_concepts"] = ["code.literal"]
    inputs["draft"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    with pytest.raises(AuthoredReviewCompileError, match="roles"):
        compile_review(**inputs)


def test_compile_review_rejects_blank_admitted_row(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    rows = [json.loads(line) for line in inputs["draft"].read_text().splitlines()]
    rows[1]["admission_recommendation"] = "admit"
    inputs["draft"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    with pytest.raises(AuthoredReviewCompileError, match="no taught concept"):
        compile_review(**inputs)


def test_validate_compiled_review_rejects_resigned_output_tamper(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    compile_review(**inputs)
    inputs["output"].chmod(0o644)
    rows = [json.loads(line) for line in inputs["output"].read_text().splitlines()]
    rows[0]["instructional_quality_ppm"] -= 1
    inputs["output"].write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    receipt = json.loads(inputs["receipt_output"].read_text())
    receipt["compiled_reviews"]["sha256"] = hashlib.sha256(
        inputs["output"].read_bytes()
    ).hexdigest()
    receipt["compiled_reviews"]["bytes"] = inputs["output"].stat().st_size
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    inputs["receipt_output"].chmod(0o644)
    inputs["receipt_output"].write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )
    with pytest.raises(AuthoredReviewCompileError, match="replay differs"):
        validate_compiled_review(**inputs)
