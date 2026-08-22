from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.finemath_audit import audit_shard
from sai.data.finemath_filter_ladder import (
    FineMathLadderError,
    build_ladder,
    validate_ladder,
)
from sai.data.token_stream import sha256_file


def _text(index: int) -> str:
    return (
        f"Let x = {index + 2} and define y = x + 3. Because this equation is "
        "linear, we can solve each step; therefore the solution follows. This "
        "theorem, proof, and example use $x+y=7$ and \\frac{y}{x}. "
    ) * 45


def _source(path: Path, *, high_score_row: bool = True) -> None:
    language_scores = [0.85, 0.92, 0.96, 0.99]
    rows = []
    for index, language_score in enumerate(language_scores):
        text = _text(index)
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
                "metadata": json.dumps({"found_math": True}),
                "score": 5.0,
                "int_score": 4 if index == 3 else 5,
                "crawl": "CC-MAIN-TEST",
                "snapshot_type": "latest",
                "language": "en",
                "language_score": (
                    0.92 if index == 2 and not high_score_row else language_score
                ),
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def _audit(tmp_path: Path, *, high_score_row: bool = True) -> Path:
    source = tmp_path / "source.parquet"
    _source(source, high_score_row=high_score_row)
    receipt = tmp_path / "audit-receipt.json"
    audit_shard(
        source,
        revision="1" * 40,
        source_file="finemath-4plus/train-00000-of-00064.parquet",
        expected_bytes=source.stat().st_size,
        expected_sha256=sha256_file(source),
        sample_size=4,
        sample_output=tmp_path / "audit-sample.jsonl",
        receipt_output=receipt,
    )
    return receipt


def _build(tmp_path: Path) -> tuple[dict, dict[str, Path]]:
    receipt = _audit(tmp_path)
    paths = {
        "candidate": tmp_path / "candidates.jsonl",
        "blind": tmp_path / "blind-review.jsonl",
        "key": tmp_path / "review-key.jsonl",
        "receipt": tmp_path / "ladder-receipt.json",
    }
    payload = build_ladder(
        receipt,
        paths["candidate"],
        paths["blind"],
        paths["key"],
        paths["receipt"],
        review_per_stratum=1,
    )
    return payload, paths


def test_builds_balanced_blinded_nonoverlapping_ladder(tmp_path: Path) -> None:
    payload, paths = _build(tmp_path)
    assert payload["summary"]["base_candidate_rows"] == 3
    assert payload["summary"]["candidate_rows_by_stratum"] == {
        "below_0p90": 1,
        "0p90_to_0p95": 1,
        "at_least_0p95": 1,
    }
    assert payload["summary"]["blind_review_rows"] == 3
    blind = [json.loads(line) for line in paths["blind"].read_text().splitlines()]
    key = [json.loads(line) for line in paths["key"].read_text().splitlines()]
    assert all(
        "stratum" not in row and "language_score_ppm" not in row for row in blind
    )
    assert [row["review_identity_sha256"] for row in blind] == [
        row["review_identity_sha256"] for row in key
    ]
    assert {row["stratum"] for row in key} == {
        "below_0p90",
        "0p90_to_0p95",
        "at_least_0p95",
    }
    assert validate_ladder(paths["receipt"]) == payload
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False


def test_rejects_tamper_overwrite_and_incomplete_stratum(tmp_path: Path) -> None:
    _, paths = _build(tmp_path)
    paths["blind"].write_text(paths["blind"].read_text() + "{}\n")
    with pytest.raises(FineMathLadderError, match="blind_review_output differs"):
        validate_ladder(paths["receipt"])
    with pytest.raises(FineMathLadderError, match="already exists"):
        build_ladder(
            tmp_path / "audit-receipt.json",
            paths["candidate"],
            paths["blind"],
            paths["key"],
            paths["receipt"],
            review_per_stratum=1,
        )

    other = tmp_path / "incomplete"
    other.mkdir()
    receipt = _audit(other, high_score_row=False)
    with pytest.raises(FineMathLadderError, match="stratum is incomplete"):
        build_ladder(
            receipt,
            other / "candidates.jsonl",
            other / "blind.jsonl",
            other / "key.jsonl",
            other / "receipt.json",
            review_per_stratum=1,
        )
