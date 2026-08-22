from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.stack_edu_aggregate import (
    StackEduAggregateError,
    aggregate_audits,
    validate_aggregate,
)
from sai.data.stack_edu_audit import audit_shard


def _shard(root: Path, index: int, count: int = 2) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = root / f"train-{index:05d}-of-{count:05d}.parquet"
    rows = []
    for offset in range(8):
        identity = index * 8 + offset
        selected = offset < 4
        rows.append(
            {
                "blob_id": hashlib.sha1(f"blob:{identity}".encode()).hexdigest(),
                "language": "Python",
                "repo_name": f"owner/repo-{identity}",
                "path": f"/src/module_{identity}.py",
                "src_encoding": "UTF-8",
                "length_bytes": 256 + identity,
                "score": 4.25 if selected else 3.25,
                "int_score": 4 if selected else 3,
                "detected_licenses": ["MIT"] if selected else [],
                "license_type": "permissive" if selected else "no_license",
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), source)
    sample = root / "sample.jsonl"
    receipt = root / "receipt.json"
    audit_shard(
        source,
        source_file=f"Python/train-{index:05d}-of-{count:05d}.parquet",
        expected_bytes=source.stat().st_size,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sample_output=sample,
        receipt_output=receipt,
    )
    return receipt


def test_aggregates_complete_language_population_and_replays(tmp_path: Path) -> None:
    receipts = [_shard(tmp_path / str(index), index) for index in range(2)]
    output = tmp_path / "aggregate.json"
    payload = aggregate_audits(receipts, output)
    assert payload["language"] == "Python"
    assert payload["shard_count"] == 2
    assert payload["summary"]["rows"] == 16
    assert payload["summary"]["candidate_rows"] == 8
    assert payload["summary"]["candidate_fraction_ppm"] == 500_000
    assert payload["summary"]["cross_shard_duplicate_identity_check_complete"] is False
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    assert validate_aggregate(output) == payload


def test_rejects_missing_duplicate_tampered_and_existing_output(tmp_path: Path) -> None:
    first = _shard(tmp_path / "first", 0)
    second = _shard(tmp_path / "second", 1)
    with pytest.raises(StackEduAggregateError, match="incomplete"):
        aggregate_audits([first], tmp_path / "missing.json")
    with pytest.raises(StackEduAggregateError, match="duplicated"):
        aggregate_audits([first, first], tmp_path / "duplicate.json")

    output = tmp_path / "aggregate.json"
    aggregate_audits([first, second], output)
    with pytest.raises(StackEduAggregateError, match="unsafe"):
        aggregate_audits([first, second], output)

    payload = json.loads(output.read_text())
    payload["summary"]["candidate_rows"] += 1
    output.write_text(json.dumps(payload))
    with pytest.raises(StackEduAggregateError, match="hash differs"):
        validate_aggregate(output)


def test_rejects_mixed_geometry(tmp_path: Path) -> None:
    first = _shard(tmp_path / "first", 0)
    third = _shard(tmp_path / "third", 1, count=3)
    with pytest.raises(StackEduAggregateError, match="geometry differs"):
        aggregate_audits([first, third], tmp_path / "aggregate.json")
