from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.stack_edu_aggregate import aggregate_audits
from sai.data.stack_edu_audit import audit_shard
from sai.data.stack_edu_candidates import (
    StackEduCandidateError,
    aggregate_candidates,
    extract_candidates,
    validate_candidate_aggregate,
    validate_shard,
)


def _row(
    identity: str,
    *,
    repo: str,
    path: str,
    selected: bool = True,
    length: int = 256,
) -> dict:
    return {
        "blob_id": hashlib.sha1(identity.encode()).hexdigest(),
        "language": "Python",
        "repo_name": repo,
        "path": path,
        "src_encoding": "UTF-8",
        "length_bytes": length,
        "score": 4.25 if selected else 3.25,
        "int_score": 4 if selected else 3,
        "detected_licenses": ["MIT"] if selected else [],
        "license_type": "permissive" if selected else "no_license",
    }


def _audit(root: Path, index: int, rows: list[dict], count: int = 2) -> Path:
    root.mkdir(parents=True)
    source = root / f"train-{index:05d}-of-{count:05d}.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    receipt = root / "audit.json"
    audit_shard(
        source,
        source_file=f"Python/train-{index:05d}-of-{count:05d}.parquet",
        expected_bytes=source.stat().st_size,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sample_output=root / "sample.jsonl",
        receipt_output=receipt,
    )
    return receipt


def _population(tmp_path: Path):
    first = _audit(
        tmp_path / "first",
        0,
        [
            _row("shared", repo="owner/a", path="/a.py", length=300),
            _row("path-a", repo="owner/shared", path="/same.py", length=301),
            _row("rejected", repo="owner/no", path="/no.py", selected=False),
        ],
    )
    second = _audit(
        tmp_path / "second",
        1,
        [
            _row("shared", repo="owner/b", path="/b.py", length=300),
            _row("path-b", repo="owner/shared", path="/same.py", length=302),
            _row("unique", repo="owner/c", path="/c.py", length=303),
        ],
    )
    metadata = tmp_path / "metadata-aggregate.json"
    aggregate_audits([first, second], metadata)
    shard_receipts = []
    for name, audit in (("first", first), ("second", second)):
        root = tmp_path / f"{name}-candidates"
        root.mkdir()
        receipt = root / "receipt.json"
        extract_candidates(audit, root / "candidates.jsonl", receipt)
        shard_receipts.append(receipt)
    return metadata, shard_receipts


def test_freezes_complete_candidates_and_cross_shard_deduplicates(
    tmp_path: Path,
) -> None:
    metadata, receipts = _population(tmp_path)
    root = tmp_path / "aggregate"
    root.mkdir()
    receipt = root / "receipt.json"
    payload = aggregate_candidates(
        metadata,
        receipts,
        root / "candidates.jsonl",
        receipt,
    )
    assert payload["summary"] == {
        "input_candidate_rows": 5,
        "input_declared_content_bytes": 1_506,
        "unique_blob_rows": 4,
        "unique_blob_declared_content_bytes": 1_206,
        "duplicate_blob_rows": 1,
        "duplicate_blob_groups": 1,
        "unique_repo_paths": 4,
        "duplicate_repo_path_rows": 1,
        "duplicate_repo_path_groups": 1,
        "cross_shard_duplicate_identity_check_complete": True,
        "deduplication_key": "upstream_blob_id_sha1",
        "canonical_occurrence": "lowest_source_shard_then_source_row_index",
    }
    rows = [
        json.loads(line)
        for line in (root / "candidates.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 4
    assert rows[0]["repo_name"] == "owner/a"
    assert len({row["blob_id"] for row in rows}) == 4
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    assert validate_candidate_aggregate(receipt) == payload


def test_rejects_candidate_tamper_and_duplicate_receipt(
    tmp_path: Path,
) -> None:
    metadata, receipts = _population(tmp_path)
    shard = json.loads(receipts[0].read_text())
    candidates = Path(shard["candidates"]["path"])
    candidates.write_text(
        candidates.read_text().replace('"int_score":4', '"int_score":5', 1)
    )
    with pytest.raises(StackEduCandidateError, match="replay differs"):
        validate_shard(receipts[0])

    other = tmp_path / "other"
    other.mkdir()
    metadata, receipts = _population(other)
    root = other / "bad-aggregate"
    root.mkdir()
    with pytest.raises(StackEduCandidateError, match="duplicated"):
        aggregate_candidates(
            metadata,
            [receipts[0], receipts[0]],
            root / "candidates.jsonl",
            root / "receipt.json",
        )

    second = json.loads(receipts[1].read_text())
    source = Path(second["source"]["path"])
    table = pq.read_table(source).to_pylist()
    table[0]["length_bytes"] = 999
    pq.write_table(pa.Table.from_pylist(table), source)
    with pytest.raises(StackEduCandidateError, match="source audit differs"):
        validate_shard(receipts[1])


def test_rejects_cross_shard_blob_geometry_drift(tmp_path: Path) -> None:
    first = _audit(
        tmp_path / "first",
        0,
        [_row("shared", repo="owner/a", path="/a.py", length=300)],
    )
    second = _audit(
        tmp_path / "second",
        1,
        [_row("shared", repo="owner/b", path="/b.py", length=301)],
    )
    metadata = tmp_path / "metadata.json"
    aggregate_audits([first, second], metadata)
    receipts = []
    for index, audit in enumerate((first, second)):
        root = tmp_path / f"candidates-{index}"
        root.mkdir()
        receipt = root / "receipt.json"
        extract_candidates(audit, root / "candidates.jsonl", receipt)
        receipts.append(receipt)
    output = tmp_path / "aggregate"
    output.mkdir()
    with pytest.raises(StackEduCandidateError, match="blob geometry differs"):
        aggregate_candidates(
            metadata,
            receipts,
            output / "candidates.jsonl",
            output / "receipt.json",
        )
