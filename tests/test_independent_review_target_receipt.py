import hashlib
import json

from sai.data.independent_review_compare import ReviewLane
from sai.data.independent_review_target_receipt import build_target_receipt
from sai.data.token_stream import canonical_sha256, sha256_file


def _candidate(row_id: str) -> dict:
    text = (f"Document {row_id}. " * 20).strip()
    row = {
        "schema": "sai-agent-data-candidate-v1",
        "text": text,
        "source": {
            "dataset": "test/data",
            "revision": "a" * 40,
            "row_id": row_id,
            "license": "test",
            "source_type": "reference",
        },
        "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "provenance_sha256": "b" * 64,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def test_seals_exact_target_and_missing_coverage(tmp_path, monkeypatch):
    population = tmp_path / "population"
    population.mkdir()
    rows = [_candidate("one"), _candidate("two")]
    candidates = population / "candidates.jsonl"
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = {
        "schema": "sai-independent-review-population-receipt-v1",
        "status": "complete_nontraining_independent_review_population",
        "population": {"rows": 2, "sha256": sha256_file(candidates)},
        "selected_descriptors": [
            {
                "candidate_identity_sha256": rows[0]["candidate_identity_sha256"],
                "lane": "a",
                "stratum": "clean_retain",
            },
            {
                "candidate_identity_sha256": rows[1]["candidate_identity_sha256"],
                "lane": "a",
                "stratum": "nonretain",
            },
        ],
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (population / "receipt.json").write_text(json.dumps(receipt))
    reviews = tmp_path / "reviews"
    reviews.mkdir()
    review = {"receipt_sha256": "a" * 64}
    review_path = (
        reviews
        / f"{rows[0]['candidate_identity_sha256']}.independent-review.json"
    )
    review_path.write_text(json.dumps(review))

    monkeypatch.setattr(
        "sai.data.independent_review_target_receipt._validate_review_receipt",
        lambda value, candidate, lane: {"verdict": "retain"},
    )
    result = build_target_receipt(
        population,
        ReviewLane("review", "model", "endpoint", reviews),
        {"clean_retain", "nonretain"},
        tmp_path / "output.json",
    )

    assert result["counts"] == {
        "target_rows": 2,
        "covered_rows": 1,
        "missing_or_failed_rows": 1,
    }
    assert result["missing_rows_require_adjudication"] is True
    assert result["training_ready"] is False
