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
from sai.data.stack_edu_content import INDEX_ROW_SCHEMA, verify_content_bundle
from sai.data.stack_edu_safety import (
    StackEduSafetyError,
    _scan_text,
    scan_content,
    validate_safety_receipt,
)
from sai.data.stack_v2_alignment import (
    ACCESS_SCHEMA,
    CURRENT_DATASET,
    CURRENT_RELEASE,
    CURRENT_REVISION,
    align_candidates,
    freeze_current_snapshot,
)
from sai.data.token_stream import canonical_sha256


def _git_blob(content: bytes) -> str:
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()


def _remote(files: dict[str, Path]):
    def lookup(paths: list[str]) -> dict:
        rows = {}
        for name in paths:
            path = files[name]
            content = path.read_bytes()
            is_lfs = name.endswith(".parquet")
            rows[name] = {
                "path": name,
                "size": len(content),
                "blob_id": "1" * 40 if is_lfs else _git_blob(content),
                "lfs_sha256": hashlib.sha256(content).hexdigest() if is_lfs else None,
                "lfs_size": len(content) if is_lfs else None,
                "xet_hash": "synthetic-xet" if is_lfs else None,
            }
        return {
            "resolved_commit_sha": CURRENT_REVISION,
            "queried_at_utc": "2026-08-22T16:00:00Z",
            "files": rows,
        }

    return lookup


def _content_receipt(tmp_path: Path, content: bytes) -> Path:
    blob = _git_blob(content)
    stack_row = {
        "blob_id": blob,
        "language": "Python",
        "repo_name": "owner/repository",
        "path": "/src/module.py",
        "src_encoding": "UTF-8",
        "length_bytes": len(content),
        "score": 4.5,
        "int_score": 4,
        "detected_licenses": ["MIT"],
        "license_type": "permissive",
    }
    old_source = tmp_path / "stack-edu.parquet"
    pq.write_table(pa.Table.from_pylist([stack_row]), old_source)
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    audit = audit_root / "receipt.json"
    audit_shard(
        old_source,
        source_file="Python/train-00000-of-00001.parquet",
        expected_bytes=old_source.stat().st_size,
        expected_sha256=hashlib.sha256(old_source.read_bytes()).hexdigest(),
        sample_output=audit_root / "sample.jsonl",
        receipt_output=audit,
    )
    metadata = tmp_path / "metadata.json"
    aggregate_audits([audit], metadata)
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    candidate_shard = candidate_root / "receipt.json"
    extract_candidates(audit, candidate_root / "candidates.jsonl", candidate_shard)
    aggregate_root = tmp_path / "candidate-aggregate"
    aggregate_root.mkdir()
    candidate_aggregate = aggregate_root / "receipt.json"
    aggregate_candidates(
        metadata,
        [candidate_shard],
        aggregate_root / "candidates.jsonl",
        candidate_aggregate,
    )

    current_row = {
        "blob_id": blob,
        "content_id": f"sha1_git:{blob}",
        "detected_licenses": ["MIT"],
        "is_generated": False,
        "is_vendor": False,
        "language": "Python",
        "license_type": "permissive",
        "path": stack_row["path"],
        "repo_name": stack_row["repo_name"],
        "revision_id": "2" * 40,
        "src_encoding": "UTF-8",
    }
    current = tmp_path / "current.parquet"
    pq.write_table(pa.Table.from_pylist([current_row]), current)
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
        "accepted_at_utc": "2026-08-22T16:00:00Z",
        "dataset_card_sha256": hashlib.sha256(card.read_bytes()).hexdigest(),
        "metadata_access_authorized": True,
        "bulk_content_access_authorized": True,
        "training_authorized": False,
        "four_b_training_authorized": False,
    }
    access_payload["receipt_sha256"] = canonical_sha256(access_payload)
    access.write_text(json.dumps(access_payload))
    snapshot = tmp_path / "snapshot.json"
    member = "data/Python/train-00000-of-00001.parquet"
    freeze_current_snapshot(
        dataset_card=card,
        access_evidence=access,
        sources=[(member, current)],
        receipt_output=snapshot,
        remote_lookup=_remote({"README.md": card, member: current}),
    )
    alignment_root = tmp_path / "alignment"
    alignment_root.mkdir()
    aligned = alignment_root / "candidates.jsonl"
    alignment = alignment_root / "receipt.json"
    align_candidates(candidate_aggregate, snapshot, aligned, alignment)

    bundle = tmp_path / "content.bin"
    bundle.write_bytes(content)
    index = tmp_path / "content.index.jsonl"
    index.write_text(
        json.dumps(
            {
                "schema": INDEX_ROW_SCHEMA,
                "ordinal": 0,
                "repo_name": stack_row["repo_name"],
                "path": stack_row["path"],
                "blob_id": blob,
                "offset": 0,
                "length_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "s3_bucket": "softwareheritage",
                "s3_key": f"content/{blob}",
                "s3_etag": "synthetic-etag",
            },
            sort_keys=True,
        )
        + "\n"
    )
    for path in (snapshot, access, aligned, alignment, bundle, index):
        path.chmod(0o444)
    receipt = tmp_path / "content.receipt.json"
    verify_content_bundle(alignment, bundle, index, receipt)
    return receipt


def test_bounded_scanner_separates_reject_review_and_clean() -> None:
    clean = _scan_text("def add(a, b):\n    return a + b\n", source_path="/add.py")
    assert clean["decision"] == "candidate_clean_by_bounded_scanner"

    private_key = _scan_text(
        "value = '''-----BEGIN PRIVATE KEY-----\nabc\n'''\n",
        source_path="/secret.py",
    )
    assert private_key["decision"] == "rejected_high_confidence_sensitive_or_invalid"
    assert "high_confidence_secret:private_key_pem" in private_key["reject_reasons"]

    review = _scan_text(
        "# contact maintainer@real-domain.dev\ndef value():\n    return 1\n",
        source_path="/contact.py",
    )
    assert review["decision"] == "manual_review_required"
    assert review["review_reasons"] == ["personal_email_candidate"]

    placeholder = _scan_text(
        'password = "example-placeholder-value"\n', source_path="/example.py"
    )
    assert (
        "nonplaceholder_generic_credential_assignment"
        not in placeholder["reject_reasons"]
    )


def test_safety_receipt_replays_findings_and_rejects_tamper(tmp_path: Path) -> None:
    content = (
        b"# contact maintainer@real-domain.dev\n"
        b"def add(left, right):\n    return left + right\n"
        b"# padding to satisfy the conservative source length policy\n" * 2
    )
    parent = _content_receipt(tmp_path, content)
    root = tmp_path / "safety"
    root.mkdir()
    findings = root / "findings.jsonl"
    receipt = root / "receipt.json"
    payload = scan_content(parent, findings, receipt)
    assert payload["summary"]["decision_counts"] == {"manual_review_required": 1}
    assert payload["summary"]["review_reason_counts"] == {"personal_email_candidate": 1}
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    assert validate_safety_receipt(receipt) == payload

    findings.chmod(0o644)
    findings.write_text(findings.read_text().replace("manual_review_required", "clean"))
    findings.chmod(0o444)
    with pytest.raises(StackEduSafetyError, match="replay differs"):
        validate_safety_receipt(receipt)
