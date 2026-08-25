from __future__ import annotations

import json
from pathlib import Path

import pytest

import sai.data.institutional_books_quarantine_manifest as manifest
from sai.data.institutional_books_compiler_aggregate import SCHEMA as AGGREGATE_SCHEMA
from sai.data.institutional_books_quarantine_manifest import (
    InstitutionalBooksQuarantineManifestError,
    build_quarantine_manifest,
)
from sai.data.token_stream import canonical_sha256


def _inputs(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    population = tmp_path / "population"
    judgments = population / "judgments"
    judgments.mkdir(parents=True)
    candidates = [
        {
            "candidate_identity_sha256": "a" * 64,
            "source_content_sha256": "c" * 64,
            "provenance_sha256": "e" * 64,
            "source": {"barcode_src": "bad-book"},
        },
        {
            "candidate_identity_sha256": "b" * 64,
            "source_content_sha256": "d" * 64,
            "provenance_sha256": "f" * 64,
            "source": {"barcode_src": "good-book"},
        },
    ]
    population_receipt = {"receipt_sha256": "1" * 64}
    monkeypatch.setattr(
        manifest,
        "_validate_population",
        lambda _root: (candidates, population_receipt),
    )
    receipts = {}
    for candidate, verdict in zip(candidates, ("reject", "retain"), strict=True):
        identity = candidate["candidate_identity_sha256"]
        receipt = {
            "receipt_sha256": ("2" if verdict == "reject" else "3") * 64,
            "judgment": {
                "judgment_sha256": ("4" if verdict == "reject" else "5") * 64,
                "verdict": verdict,
                "risks": {"ocr_damage": verdict == "reject"},
            },
        }
        receipts[identity] = receipt
        (judgments / f"{identity}.book-compiler.json").write_text("{}")
    monkeypatch.setattr(
        manifest,
        "_validate_receipt",
        lambda _receipt, candidate: receipts[candidate["candidate_identity_sha256"]],
    )
    monkeypatch.setattr(
        manifest,
        "triage_route",
        lambda judgment: "quarantine"
        if judgment["verdict"] == "reject"
        else "representation_verification",
    )
    aggregate = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete_nontraining_book_compiler_aggregate",
        "population": {"rows": 2},
        "counts": {"triage_route": {"quarantine": 1}},
        "model_judgments_are_verified_admissions": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    aggregate["receipt_sha256"] = canonical_sha256(aggregate)
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate))
    return population, judgments, aggregate_path


def test_book_quarantine_manifest_excludes_without_source_text(
    tmp_path: Path, monkeypatch
) -> None:
    population, judgments, aggregate = _inputs(tmp_path, monkeypatch)
    output = tmp_path / "output"
    result = build_quarantine_manifest(population, judgments, aggregate, output)
    rows = [
        json.loads(line)
        for line in (output / "quarantine_exclusions.jsonl").open()
    ]
    assert len(rows) == 1
    assert rows[0]["source_book_id"] == "bad-book"
    assert rows[0]["dataset_materialization_allowed"] is False
    assert rows[0]["source_text_persisted"] is False
    assert result["quarantine_rows"] == 1


def test_book_quarantine_manifest_rejects_count_tamper(
    tmp_path: Path, monkeypatch
) -> None:
    population, judgments, aggregate = _inputs(tmp_path, monkeypatch)
    value = json.loads(aggregate.read_text())
    value["counts"]["triage_route"]["quarantine"] = 2
    value["receipt_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    aggregate.write_text(json.dumps(value))
    with pytest.raises(InstitutionalBooksQuarantineManifestError, match="coverage"):
        build_quarantine_manifest(
            population, judgments, aggregate, tmp_path / "output"
        )


def test_book_quarantine_manifest_accepts_omitted_zero_count(
    tmp_path: Path, monkeypatch
) -> None:
    population, judgments, aggregate = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(manifest, "triage_route", lambda _judgment: "cleanup_review")
    value = json.loads(aggregate.read_text())
    value["counts"]["triage_route"].pop("quarantine")
    value["receipt_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    aggregate.write_text(json.dumps(value))

    output = tmp_path / "output"
    result = build_quarantine_manifest(population, judgments, aggregate, output)

    assert result["quarantine_rows"] == 0
    assert (output / "quarantine_exclusions.jsonl").read_text() == ""


def test_book_quarantine_manifest_rejects_extra_judgment(
    tmp_path: Path, monkeypatch
) -> None:
    population, judgments, aggregate = _inputs(tmp_path, monkeypatch)
    (judgments / f"{'9' * 64}.book-compiler.json").write_text("{}")
    with pytest.raises(
        InstitutionalBooksQuarantineManifestError, match="population"
    ):
        build_quarantine_manifest(
            population, judgments, aggregate, tmp_path / "output"
        )
