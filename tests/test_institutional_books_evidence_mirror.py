from __future__ import annotations

import pytest

import sai.data.institutional_books_evidence_mirror as mirror
from sai.data.institutional_books_evidence_mirror import (
    InstitutionalBooksEvidenceMirrorError,
    mirror_evidence,
)
from sai.data.token_stream import sha256_file


def test_mirror_copies_only_fixed_safe_files_and_hashes_bytes(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    (source / "safe").mkdir(parents=True)
    (source / "safe" / "receipt.json").write_text('{"training_ready":false}')
    monkeypatch.setattr(
        mirror,
        "SAFE_FILES",
        (("safe_receipt", "safe/receipt.json"),),
    )
    output = tmp_path / "durable"
    receipt = mirror_evidence(source, output)
    copied = output / "files" / "safe_receipt" / "receipt.json"
    assert copied.read_bytes() == (source / "safe" / "receipt.json").read_bytes()
    assert receipt["file_count"] == 1
    assert receipt["files"][0]["sha256"] == sha256_file(copied)
    assert receipt["full_book_text_copied"] is False


def test_mirror_fails_closed_when_allowlisted_input_is_missing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        mirror,
        "SAFE_FILES",
        (("missing", "missing/receipt.json"),),
    )
    with pytest.raises(InstitutionalBooksEvidenceMirrorError, match="file differs"):
        mirror_evidence(tmp_path / "source", tmp_path / "durable")
    assert not (tmp_path / "durable").exists()
