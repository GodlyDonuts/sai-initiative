from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.stack_edu_audit import (
    REVISION,
    StackEduAuditError,
    audit_shard,
    validate_audit,
)


def _source(tmp_path: Path) -> Path:
    rows = []
    for index in range(80):
        accepted = index < 40
        rows.append(
            {
                "blob_id": hashlib.sha1(f"blob:{index}".encode()).hexdigest(),
                "language": "Python",
                "repo_name": f"owner/repo-{index // 4}",
                "path": f"/src/module_{index}.py",
                "src_encoding": "UTF-8",
                "length_bytes": 256 + index,
                "score": 4.25 if accepted else 3.25,
                "int_score": 4 if accepted else 3,
                "detected_licenses": ["MIT"] if accepted else [],
                "license_type": "permissive" if accepted else "no_license",
            }
        )
    source = tmp_path / "python.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    return source


def _build(tmp_path: Path) -> tuple[dict, Path, Path, Path]:
    source = _source(tmp_path)
    sample = tmp_path / "sample.jsonl"
    receipt = tmp_path / "receipt.json"
    payload = audit_shard(
        source,
        source_file="Python/train-00000-of-00005.parquet",
        expected_bytes=source.stat().st_size,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sample_output=sample,
        receipt_output=receipt,
    )
    return payload, source, sample, receipt


def test_audits_metadata_without_admitting_code(tmp_path: Path) -> None:
    payload, _source_path, sample, receipt = _build(tmp_path)
    assert payload["source"]["revision"] == REVISION
    assert payload["status"] == "metadata_audited_content_not_acquired"
    assert payload["summary"]["rows"] == 80
    assert payload["summary"]["candidate_rows"] == 40
    assert payload["summary"]["permissive_rows"] == 40
    assert payload["summary"]["permissive_rows_without_detected_license"] == 0
    assert payload["summary"]["no_license_rows"] == 40
    assert payload["review_sample"]["rows"] == 64
    rows = [json.loads(line) for line in sample.read_text().splitlines()]
    assert sum(row["selected_by_metadata_policy"] for row in rows) == 32
    assert all("content" not in row and "text" not in row for row in rows)
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    assert validate_audit(receipt) == payload


def test_rejects_unallowlisted_license_and_source_tamper(tmp_path: Path) -> None:
    source = _source(tmp_path)
    table = pq.read_table(source).to_pylist()
    table[0]["detected_licenses"] = ["GPL-3.0-only"]
    pq.write_table(pa.Table.from_pylist(table), source)
    sample = tmp_path / "sample.jsonl"
    receipt = tmp_path / "receipt.json"
    payload = audit_shard(
        source,
        source_file="Python/train-00000-of-00005.parquet",
        expected_bytes=source.stat().st_size,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sample_output=sample,
        receipt_output=receipt,
    )
    assert payload["summary"]["candidate_rows"] == 39

    with source.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(StackEduAuditError, match="source differs"):
        validate_audit(receipt)


def test_counts_but_does_not_admit_permissive_rows_without_license(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    table = pq.read_table(source).to_pylist()
    table[0]["detected_licenses"] = []
    pq.write_table(pa.Table.from_pylist(table), source)
    payload = audit_shard(
        source,
        source_file="Python/train-00000-of-00005.parquet",
        expected_bytes=source.stat().st_size,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sample_output=tmp_path / "sample.jsonl",
        receipt_output=tmp_path / "receipt.json",
    )
    assert payload["summary"]["permissive_rows_without_detected_license"] == 1
    assert payload["summary"]["candidate_rows"] == 39


def test_rejects_resigned_receipt_and_sample_tamper(tmp_path: Path) -> None:
    _payload, _source_path, sample, receipt = _build(tmp_path)
    receipt_payload = json.loads(receipt.read_text())
    receipt_payload["policy"]["minimum_length_bytes"] = 1
    unsigned = {
        key: value for key, value in receipt_payload.items() if key != "receipt_sha256"
    }
    receipt_payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt.write_text(json.dumps(receipt_payload, sort_keys=True) + "\n")
    with pytest.raises(StackEduAuditError, match="audit replay differs"):
        validate_audit(receipt)

    second = tmp_path / "second"
    second.mkdir()
    _payload, _source_path, sample, receipt = _build(second)
    sample.write_text(sample.read_text().replace('"int_score":4', '"int_score":5', 1))
    with pytest.raises(StackEduAuditError, match="audit replay differs"):
        validate_audit(receipt)


def test_rejects_wrong_columns_revision_and_existing_outputs(tmp_path: Path) -> None:
    source = _source(tmp_path)
    table = pq.read_table(source).drop(["score"])
    pq.write_table(table, source)
    with pytest.raises(StackEduAuditError, match="columns"):
        audit_shard(
            source,
            source_file="Python/train-00000-of-00005.parquet",
            expected_bytes=source.stat().st_size,
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            sample_output=tmp_path / "sample.jsonl",
            receipt_output=tmp_path / "receipt.json",
        )

    source = _source(tmp_path)
    sample = tmp_path / "existing.jsonl"
    sample.write_text("occupied\n")
    with pytest.raises(StackEduAuditError, match="boundary"):
        audit_shard(
            source,
            source_file="Python/train-00000-of-00005.parquet",
            expected_bytes=source.stat().st_size,
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            sample_output=sample,
            receipt_output=tmp_path / "new-receipt.json",
        )
    with pytest.raises(StackEduAuditError, match="source file"):
        audit_shard(
            source,
            source_file="Python/train-00000-of-00000.parquet",
            expected_bytes=source.stat().st_size,
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            sample_output=tmp_path / "new-sample.jsonl",
            receipt_output=tmp_path / "another-receipt.json",
        )
