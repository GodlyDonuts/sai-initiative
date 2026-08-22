from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.finemath_audit import FineMathAuditError, audit_shard, validate_audit
from sai.data.token_stream import sha256_file


def _source(path: Path) -> None:
    texts = [
        "A self-contained explanation of addition and multiplication. " * 12,
        "Write my paper with the best essay writing service. " * 12,
        "Casino betting poker sportsbook probability claims. " * 12,
        "Homework answer key from Course Hero with worked algebra. " * 12,
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
                "metadata": json.dumps({"found_math": index in {0, 3}}),
                "score": 4.5,
                "int_score": 4 if index else 5,
                "crawl": "CC-MAIN-TEST",
                "snapshot_type": "latest",
                "language": "en",
                "language_score": 0.99,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)


def _audit(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    source = tmp_path / "shard.parquet"
    sample = tmp_path / "sample.jsonl"
    receipt = tmp_path / "receipt.json"
    _source(source)
    payload = audit_shard(
        source,
        revision="1" * 40,
        source_file="finemath-4plus/train-00000-of-00064.parquet",
        expected_bytes=source.stat().st_size,
        expected_sha256=sha256_file(source),
        sample_size=4,
        sample_output=sample,
        receipt_output=receipt,
    )
    return source, sample, receipt, payload


def test_audits_and_replays_exact_shard(tmp_path: Path) -> None:
    _, sample, receipt, payload = _audit(tmp_path)
    assert validate_audit(receipt) == payload
    assert payload["summary"]["rows"] == 4
    assert payload["summary"]["score_counts"] == {"4": 3, "5": 1}
    assert payload["summary"]["found_math_rows"] == 2
    assert payload["summary"]["risk_signal_lower_bounds"] == {
        "answer_key_or_homework_site": 1,
        "casino_or_betting": 1,
        "essay_service": 1,
        "malformed_url": 0,
        "seo_or_marketing": 0,
        "under_80_words": 1,
    }
    rows = [json.loads(line) for line in sample.read_text().splitlines()]
    assert len(rows) == 4
    assert {row["row_index"] for row in rows} == {0, 1, 2, 3}
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False


def test_rejects_tamper_overwrite_and_unsafe_inputs(tmp_path: Path) -> None:
    source, sample, receipt, _ = _audit(tmp_path)
    with pytest.raises(FineMathAuditError, match="already exists"):
        audit_shard(
            source,
            revision="1" * 40,
            source_file="finemath-4plus/train-00000-of-00064.parquet",
            expected_bytes=source.stat().st_size,
            expected_sha256=sha256_file(source),
            sample_size=4,
            sample_output=sample,
            receipt_output=receipt,
        )
    sample.write_text(sample.read_text() + "{}\n")
    with pytest.raises(FineMathAuditError, match="sample differs"):
        validate_audit(receipt)

    target = tmp_path / "target.parquet"
    _source(target)
    link = tmp_path / "linked.parquet"
    link.symlink_to(target)
    with pytest.raises(FineMathAuditError, match="unsafe"):
        audit_shard(
            link,
            revision="1" * 40,
            source_file="finemath-4plus/train-00001-of-00064.parquet",
            expected_bytes=target.stat().st_size,
            expected_sha256=sha256_file(target),
            sample_size=1,
            sample_output=tmp_path / "other-sample.jsonl",
            receipt_output=tmp_path / "other-receipt.json",
        )
