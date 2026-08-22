from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data.stack_edu_aggregate import aggregate_audits
from sai.data.stack_edu_audit import audit_shard
from sai.data.stack_edu_candidates import aggregate_candidates, extract_candidates
from sai.data.stack_edu_content import (
    INDEX_ROW_SCHEMA,
    StackEduContentError,
    validate_content_receipt,
    verify_content_bundle,
)
from sai.data.stack_v2_alignment import (
    ACCESS_SCHEMA,
    CURRENT_DATASET,
    CURRENT_RELEASE,
    CURRENT_REVISION,
    StackV2AlignmentError,
    align_candidates,
    freeze_current_snapshot,
    validate_alignment,
    validate_current_snapshot,
)
from sai.data.token_stream import canonical_sha256


def _stack_edu_row(
    identity: str,
    *,
    repo: str,
    path: str,
    length: int = 256,
    content: bytes | None = None,
) -> dict:
    blob_id = (
        hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if content is not None
        else hashlib.sha1(identity.encode()).hexdigest()
    )
    return {
        "blob_id": blob_id,
        "language": "Python",
        "repo_name": repo,
        "path": path,
        "src_encoding": "UTF-8",
        "length_bytes": len(content) if content is not None else length,
        "score": 4.5,
        "int_score": 4,
        "detected_licenses": ["MIT"],
        "license_type": "permissive",
    }


def _candidate_population(
    tmp_path: Path, rows: list[dict] | None = None
) -> tuple[Path, list[dict]]:
    rows = rows or [
        _stack_edu_row("kept", repo="owner/kept", path="/src/kept.py"),
        _stack_edu_row("opted-out", repo="owner/gone", path="/src/gone.py"),
        _stack_edu_row("vendor", repo="owner/vendor", path="/vendor/code.py"),
    ]
    source = tmp_path / "stack-edu.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    audit_receipt = audit_root / "receipt.json"
    audit_shard(
        source,
        source_file="Python/train-00000-of-00001.parquet",
        expected_bytes=source.stat().st_size,
        expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sample_output=audit_root / "sample.jsonl",
        receipt_output=audit_receipt,
    )
    metadata = tmp_path / "metadata.json"
    aggregate_audits([audit_receipt], metadata)
    shard_root = tmp_path / "candidate-shard"
    shard_root.mkdir()
    shard_receipt = shard_root / "receipt.json"
    extract_candidates(
        audit_receipt,
        shard_root / "candidates.jsonl",
        shard_receipt,
    )
    aggregate_root = tmp_path / "candidate-aggregate"
    aggregate_root.mkdir()
    aggregate_receipt = aggregate_root / "receipt.json"
    aggregate_candidates(
        metadata,
        [shard_receipt],
        aggregate_root / "candidates.jsonl",
        aggregate_receipt,
    )
    return aggregate_receipt, rows


def _current_row(
    parent: dict,
    *,
    vendor: bool = False,
    generated: bool = False,
    permissive: bool = True,
) -> dict:
    return {
        "blob_id": parent["blob_id"],
        "content_id": f"sha1_git:{parent['blob_id']}",
        "detected_licenses": ["MIT"] if permissive else [],
        "is_generated": generated,
        "is_vendor": vendor,
        "language": "Python",
        "license_type": "permissive" if permissive else "no_license",
        "path": parent["path"],
        "repo_name": parent["repo_name"],
        "revision_id": hashlib.sha1(
            (parent["repo_name"] + "-revision").encode()
        ).hexdigest(),
        "src_encoding": "UTF-8",
    }


def _snapshot(tmp_path: Path, rows: list[dict], *, bulk: bool = False) -> Path:
    source = tmp_path / "current.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    card = tmp_path / "README.md"
    card.write_text(
        "bigcode/the-stack-v2 v2.2.0 removed opt-outs through 2026-07-29. "
        "SoftwareHeritage requires users to update your own version of The Stack v2.\n"
    )
    access = tmp_path / "access.json"
    access_payload = {
        "schema": ACCESS_SCHEMA,
        "status": "metadata_access_terms_accepted",
        "dataset": CURRENT_DATASET,
        "revision": CURRENT_REVISION,
        "release": CURRENT_RELEASE,
        "accepted_by": "synthetic-test-reviewer",
        "accepted_at_utc": "2026-08-22T15:00:00Z",
        "dataset_card_sha256": hashlib.sha256(card.read_bytes()).hexdigest(),
        "metadata_access_authorized": True,
        "bulk_content_access_authorized": bulk,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    access_payload["receipt_sha256"] = canonical_sha256(access_payload)
    access.write_text(json.dumps(access_payload))
    receipt = tmp_path / "snapshot.json"
    freeze_current_snapshot(
        dataset_card=card,
        access_evidence=access,
        sources=[("data/Python/train-00000-of-00001.parquet", source)],
        receipt_output=receipt,
    )
    return receipt


def test_aligns_only_current_permissive_nonvendor_candidates(tmp_path: Path) -> None:
    candidates, rows = _candidate_population(tmp_path)
    snapshot = _snapshot(
        tmp_path,
        [
            _current_row(rows[0]),
            _current_row(rows[0]),
            _current_row(rows[2], vendor=True),
        ],
    )
    output_root = tmp_path / "aligned"
    output_root.mkdir()
    receipt = output_root / "receipt.json"
    payload = align_candidates(
        candidates,
        snapshot,
        output_root / "candidates.jsonl",
        receipt,
    )
    assert payload["summary"] == {
        "input_candidate_rows": 3,
        "aligned_rows": 1,
        "removed_rows": 2,
        "removed_by_reason": {
            "absent_from_current_opt_out_enacted_snapshot": 1,
            "current_vendor": 1,
        },
        "current_rows_scanned": 3,
        "current_matching_occurrences": 3,
        "current_duplicate_matching_occurrences": 1,
        "membership_key": "exact_repo_name_path_blob_id",
        "current_release_is_opt_out_authority": True,
    }
    aligned = [
        json.loads(line)
        for line in (output_root / "candidates.jsonl").read_text().splitlines()
    ]
    assert len(aligned) == 1
    assert aligned[0]["candidate"]["repo_name"] == "owner/kept"
    assert aligned[0]["current"]["is_vendor"] is False
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    assert validate_alignment(receipt) == payload


def test_snapshot_and_alignment_tamper_fail_closed(tmp_path: Path) -> None:
    candidates, rows = _candidate_population(tmp_path)
    snapshot = _snapshot(tmp_path, [_current_row(rows[0])])
    payload = json.loads(snapshot.read_text())
    payload["revision"] = "0" * 40
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    snapshot.write_text(json.dumps(payload))
    with pytest.raises(StackV2AlignmentError, match="snapshot receipt differs"):
        validate_current_snapshot(snapshot)

    snapshot.unlink()
    snapshot = _snapshot(tmp_path, [_current_row(rows[0])])
    output_root = tmp_path / "aligned"
    output_root.mkdir()
    receipt = output_root / "receipt.json"
    align_candidates(
        candidates,
        snapshot,
        output_root / "candidates.jsonl",
        receipt,
    )
    aligned = output_root / "candidates.jsonl"
    aligned.write_text(aligned.read_text().replace("owner/kept", "owner/tampered"))
    with pytest.raises(StackV2AlignmentError, match="replay differs"):
        validate_alignment(receipt)


def test_snapshot_requires_complete_current_release_shards(tmp_path: Path) -> None:
    _candidates, rows = _candidate_population(tmp_path)
    source = tmp_path / "current.parquet"
    pq.write_table(pa.Table.from_pylist([_current_row(rows[0])]), source)
    card = tmp_path / "README.md"
    card.write_text(
        "bigcode/the-stack-v2 v2.2.0 removed opt-outs through 2026-07-29. "
        "SoftwareHeritage requires users to update your own version of The Stack v2."
    )
    access = tmp_path / "access.json"
    access_payload = {
        "schema": ACCESS_SCHEMA,
        "status": "metadata_access_terms_accepted",
        "dataset": CURRENT_DATASET,
        "revision": CURRENT_REVISION,
        "release": CURRENT_RELEASE,
        "accepted_by": "synthetic-test-reviewer",
        "accepted_at_utc": "2026-08-22T15:00:00Z",
        "dataset_card_sha256": hashlib.sha256(card.read_bytes()).hexdigest(),
        "metadata_access_authorized": True,
        "bulk_content_access_authorized": False,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    access_payload["receipt_sha256"] = canonical_sha256(access_payload)
    access.write_text(json.dumps(access_payload))
    with pytest.raises(StackV2AlignmentError, match="incomplete"):
        freeze_current_snapshot(
            dataset_card=card,
            access_evidence=access,
            sources=[("data/Python/train-00000-of-00002.parquet", source)],
            receipt_output=tmp_path / "snapshot.json",
        )


def test_current_revision_is_the_pinned_opt_out_release() -> None:
    assert CURRENT_REVISION == "e565caa3a78c2423bd374333a472b049eb090e47"


def test_zero_current_matches_publish_an_explicit_empty_alignment(
    tmp_path: Path,
) -> None:
    candidates, _rows = _candidate_population(tmp_path)
    unrelated = _stack_edu_row(
        "unrelated", repo="other/repository", path="/unrelated.py"
    )
    snapshot = _snapshot(tmp_path, [_current_row(unrelated)])
    output_root = tmp_path / "empty-alignment"
    output_root.mkdir()
    receipt = output_root / "receipt.json"
    payload = align_candidates(
        candidates,
        snapshot,
        output_root / "candidates.jsonl",
        receipt,
    )
    assert payload["summary"]["aligned_rows"] == 0
    assert payload["summary"]["removed_rows"] == 3
    assert (output_root / "candidates.jsonl").read_bytes() == b""
    assert validate_alignment(receipt) == payload


def test_verifies_exact_git_blob_content_bundle_and_replays(tmp_path: Path) -> None:
    content = (b"def add(left, right):\n    return left + right\n\n" * 4) + b"# end\n"
    source_row = _stack_edu_row(
        "content",
        repo="owner/content",
        path="/src/content.py",
        content=content,
    )
    candidates, rows = _candidate_population(tmp_path, [source_row])
    snapshot = _snapshot(tmp_path, [_current_row(rows[0])], bulk=True)
    alignment_root = tmp_path / "alignment"
    alignment_root.mkdir()
    aligned = alignment_root / "candidates.jsonl"
    alignment_receipt = alignment_root / "receipt.json"
    align_candidates(candidates, snapshot, aligned, alignment_receipt)

    bundle = tmp_path / "content.bin"
    bundle.write_bytes(content)
    index = tmp_path / "content.index.jsonl"
    index_row = {
        "schema": INDEX_ROW_SCHEMA,
        "ordinal": 0,
        "repo_name": source_row["repo_name"],
        "path": source_row["path"],
        "blob_id": source_row["blob_id"],
        "offset": 0,
        "length_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "s3_bucket": "softwareheritage",
        "s3_key": f"content/{source_row['blob_id']}",
        "s3_etag": "synthetic-test-etag",
    }
    index.write_text(json.dumps(index_row, sort_keys=True) + "\n")
    for path in (
        snapshot,
        tmp_path / "access.json",
        aligned,
        alignment_receipt,
        bundle,
        index,
    ):
        path.chmod(0o444)

    receipt = tmp_path / "content.receipt.json"
    payload = verify_content_bundle(alignment_receipt, bundle, index, receipt)
    assert payload["verification"] == {
        "ordered_content_sha256": canonical_sha256(
            [hashlib.sha256(content).hexdigest()]
        ),
        "git_blob_sha1_verified_rows": 1,
        "independent_sha256_verified_rows": 1,
        "candidate_length_verified_rows": 1,
        "utf8_round_trip_verified_rows": 1,
        "gap_overlap_or_trailing_bytes": 0,
        "complete": True,
    }
    assert payload["training_authorized"] is False
    assert validate_content_receipt(receipt) == payload

    receipt.chmod(0o644)
    receipt.unlink()
    index.chmod(0o644)
    bad = dict(index_row)
    bad["sha256"] = "0" * 64
    index.write_text(json.dumps(bad) + "\n")
    index.chmod(0o444)
    with pytest.raises(StackEduContentError, match="content identity differs"):
        verify_content_bundle(alignment_receipt, bundle, index, receipt)


def test_content_verification_requires_explicit_bulk_access(tmp_path: Path) -> None:
    content = b"# a source file with enough bytes for the conservative policy\n" * 3
    source_row = _stack_edu_row(
        "content",
        repo="owner/content",
        path="/src/content.py",
        content=content,
    )
    candidates, rows = _candidate_population(tmp_path, [source_row])
    snapshot = _snapshot(tmp_path, [_current_row(rows[0])], bulk=False)
    alignment_root = tmp_path / "alignment"
    alignment_root.mkdir()
    aligned = alignment_root / "candidates.jsonl"
    alignment_receipt = alignment_root / "receipt.json"
    align_candidates(candidates, snapshot, aligned, alignment_receipt)
    bundle = tmp_path / "content.bin"
    bundle.write_bytes(content)
    index = tmp_path / "content.index.jsonl"
    index.write_text(
        json.dumps(
            {
                "schema": INDEX_ROW_SCHEMA,
                "ordinal": 0,
                "repo_name": source_row["repo_name"],
                "path": source_row["path"],
                "blob_id": source_row["blob_id"],
                "offset": 0,
                "length_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "s3_bucket": "softwareheritage",
                "s3_key": f"content/{source_row['blob_id']}",
                "s3_etag": "synthetic-test-etag",
            }
        )
        + "\n"
    )
    for path in (
        snapshot,
        tmp_path / "access.json",
        aligned,
        alignment_receipt,
        bundle,
        index,
    ):
        path.chmod(0o444)
    with pytest.raises(StackEduContentError, match="not authorized"):
        verify_content_bundle(
            alignment_receipt,
            bundle,
            index,
            tmp_path / "content.receipt.json",
        )
