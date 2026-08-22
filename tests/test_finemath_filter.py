from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.finemath_audit import audit_shard
from sai.data.finemath_filter import (
    FineMathFilterError,
    build_filtered_candidate,
    validate_filtered_candidate,
)
from sai.data.token_stream import sha256_file


def _long(text: str) -> str:
    return (text + " ") * 45


def _source(path: Path) -> None:
    texts = [
        _long(
            "Let x = 2 and define y = x + 3. Because the equation is linear, "
            "we can solve each step; therefore y = 5. This proof and example "
            "use $x+y=7$ and \\frac{y}{x}."
        ),
        _long(
            "Let n = 4. Because n is even, we can pair terms; therefore the "
            "equation n = 2k has a solution. This theorem and proof use $k=2$."
        ),
        _long(
            "Write my paper with an essay writing service. Let x = 2; because "
            "x is positive, therefore $x>0$ and this proof uses \\frac{x}{1}."
        ),
        _long(
            "This is a long general explanation because examples are useful. "
            "Therefore we can define an idea and explain each step carefully."
        ),
        _long("$x=2$ \\frac{x}{2} equation theorem proof value result."),
    ]
    rows = []
    for index, text in enumerate(texts):
        rows.append(
            {
                "url": f"https://example{index}.org/math",
                "fetch_time": index,
                "content_mime_type": "text/html",
                "warc_filename": "sample.warc.gz",
                "warc_record_offset": index,
                "warc_record_length": len(text),
                "text": text,
                "token_count": len(text.split()),
                "char_count": len(text),
                "metadata": json.dumps({"found_math": index != 3}),
                "score": 5.0,
                "int_score": 4 if index == 1 else 5,
                "crawl": "CC-MAIN-TEST",
                "snapshot_type": "latest",
                "language": "en",
                "language_score": 0.99,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def _build(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict]:
    source = tmp_path / "shard.parquet"
    audit_sample = tmp_path / "audit-sample.jsonl"
    audit_receipt = tmp_path / "audit-receipt.json"
    _source(source)
    audit_shard(
        source,
        revision="1" * 40,
        source_file="finemath-4plus/train-00000-of-00064.parquet",
        expected_bytes=source.stat().st_size,
        expected_sha256=sha256_file(source),
        sample_size=5,
        sample_output=audit_sample,
        receipt_output=audit_receipt,
    )
    accepted = tmp_path / "accepted.jsonl"
    review = tmp_path / "review.jsonl"
    receipt = tmp_path / "filter-receipt.json"
    payload = build_filtered_candidate(
        audit_receipt,
        accepted,
        review,
        receipt,
        review_per_decision=1,
    )
    return source, accepted, review, receipt, payload


def test_builds_and_replays_conservative_candidate(tmp_path: Path) -> None:
    _, accepted, review, receipt, payload = _build(tmp_path)
    assert payload["status"] == "filtered_candidate_not_admitted"
    assert payload["summary"]["rows"] == 5
    assert payload["summary"]["accepted_rows"] == 1
    assert payload["summary"]["rejected_rows"] == 4
    assert payload["summary"]["rejection_reason_counts"] == {
        "found_math_absent": 1,
        "insufficient_distinct_math_signals": 1,
        "insufficient_explanatory_structure": 1,
        "risk_pattern:essay_service": 1,
        "upstream_score_below_5": 1,
    }
    accepted_rows = [json.loads(line) for line in accepted.read_text().splitlines()]
    assert len(accepted_rows) == 1
    assert accepted_rows[0]["source"]["license"] == "ODC-By-1.0"
    assert accepted_rows[0]["limitations"] == [
        "candidate_not_benchmark_decontaminated",
        "candidate_not_near_deduplicated",
        "candidate_not_human_quality_approved",
    ]
    review_rows = [json.loads(line) for line in review.read_text().splitlines()]
    assert {row["decision"] for row in review_rows} == {"accepted", "rejected"}
    assert validate_filtered_candidate(receipt) == payload
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False


def test_rejects_tamper_overwrite_and_bad_review_geometry(tmp_path: Path) -> None:
    _, accepted, review, receipt, _ = _build(tmp_path)
    with pytest.raises(FineMathFilterError, match="already exists"):
        build_filtered_candidate(
            tmp_path / "audit-receipt.json",
            accepted,
            review,
            receipt,
            review_per_decision=1,
        )
    accepted.write_text(accepted.read_text() + "{}\n")
    with pytest.raises(FineMathFilterError, match="replay differs"):
        validate_filtered_candidate(receipt)
    with pytest.raises(FineMathFilterError, match="review geometry differs"):
        build_filtered_candidate(
            tmp_path / "audit-receipt.json",
            tmp_path / "new-accepted.jsonl",
            tmp_path / "new-review.jsonl",
            tmp_path / "new-receipt.json",
            review_per_decision=0,
        )


def test_preserves_zero_acceptance_as_terminal_evidence(tmp_path: Path) -> None:
    source = tmp_path / "all-rejected.parquet"
    _source(source)
    table = pq.read_table(source)
    columns = {name: table[name].to_pylist() for name in table.column_names}
    columns["int_score"] = [4] * table.num_rows
    pq.write_table(pa.Table.from_pydict(columns), source)
    audit_receipt = tmp_path / "audit-receipt.json"
    audit_shard(
        source,
        revision="1" * 40,
        source_file="finemath-4plus/train-00000-of-00064.parquet",
        expected_bytes=source.stat().st_size,
        expected_sha256=sha256_file(source),
        sample_size=5,
        sample_output=tmp_path / "audit-sample.jsonl",
        receipt_output=audit_receipt,
    )
    accepted = tmp_path / "accepted.jsonl"
    review = tmp_path / "review.jsonl"
    receipt = tmp_path / "receipt.json"
    payload = build_filtered_candidate(
        audit_receipt,
        accepted,
        review,
        receipt,
        review_per_decision=1,
    )
    assert payload["status"] == "filter_empty_no_candidate"
    assert payload["summary"]["accepted_rows"] == 0
    assert payload["summary"]["review_rows_by_decision"] == {
        "accepted": 0,
        "rejected": 1,
    }
    assert accepted.read_bytes() == b""
    assert validate_filtered_candidate(receipt) == payload
