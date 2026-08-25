import json
from pathlib import Path

import pytest

from sai.data.pleias_practical_admission_recovery import (
    PleiasPracticalAdmissionRecoveryError,
    cleanup_failed_primary,
    validate_primary,
)
from sai.data.token_stream import canonical_sha256


def _primary_receipt(path: Path, *, training_ready: bool = True) -> Path:
    payload = {
        "schema": "sai-pleias-practical-admission-receipt-v1",
        "status": "complete_practical_pleias_pretraining_admission",
        "global_exact_content_deduplication_complete": True,
        "known_quarantine_exclusions_applied": True,
        "training_ready": training_ready,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True))
    return path


def test_validate_primary_accepts_only_complete_signed_admission(
    tmp_path: Path,
) -> None:
    good = _primary_receipt(tmp_path / "good.json")
    assert validate_primary(good)["training_ready"] is True
    bad = _primary_receipt(tmp_path / "bad.json", training_ready=False)
    with pytest.raises(
        PleiasPracticalAdmissionRecoveryError,
        match="receipt is not complete",
    ):
        validate_primary(bad)


def test_cleanup_removes_only_exact_failed_partial_and_signs_receipt(
    tmp_path: Path,
) -> None:
    root = tmp_path / "exact-output"
    nested = root / "shards" / "shard_00000"
    nested.mkdir(parents=True)
    (nested / ".locators.partial.parquet").write_bytes(b"partial")
    destination = tmp_path / "evidence" / "receipt.json"
    result = cleanup_failed_primary(
        root, root, destination, 822232, "TIMEOUT"
    )
    assert not root.exists()
    assert result["partial_root_present"] is True
    assert result["partial_file_bytes_removed"] == 7
    assert result["recovery_admission_required"] is True
    persisted = json.loads(destination.read_text())
    receipt = persisted.pop("receipt_sha256")
    assert receipt == canonical_sha256(persisted)


def test_cleanup_records_an_absent_partial_root(tmp_path: Path) -> None:
    root = tmp_path / "exact-output"
    result = cleanup_failed_primary(
        root, root, tmp_path / "receipt.json", 822232, "FAILED"
    )
    assert result["partial_root_present"] is False
    assert result["partial_entries"] == 0


def test_cleanup_rejects_different_target_or_nonterminal_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    root.mkdir()
    with pytest.raises(PleiasPracticalAdmissionRecoveryError):
        cleanup_failed_primary(
            root, tmp_path / "other", tmp_path / "receipt.json", 822232, "FAILED"
        )
    with pytest.raises(PleiasPracticalAdmissionRecoveryError):
        cleanup_failed_primary(
            root, root, tmp_path / "receipt.json", 822232, "RUNNING"
        )
    assert root.exists()


def test_cleanup_rejects_symlink_anywhere_without_deleting(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    external = tmp_path / "external"
    external.write_text("preserve")
    (root / "unsafe").symlink_to(external)
    with pytest.raises(
        PleiasPracticalAdmissionRecoveryError,
        match="contains a symlink",
    ):
        cleanup_failed_primary(
            root, root, tmp_path / "receipt.json", 822232, "TIMEOUT"
        )
    assert root.exists()
    assert external.read_text() == "preserve"
