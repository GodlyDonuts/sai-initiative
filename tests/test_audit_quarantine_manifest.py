from __future__ import annotations

import json
from pathlib import Path

import pytest

import sai.data.audit_quarantine_manifest as manifest
from sai.data.audit_quarantine_manifest import (
    AuditQuarantineManifestError,
    build_quarantine_manifest,
)
from sai.data.reservoir_audit_aggregate import SCHEMA as AGGREGATE_SCHEMA
from sai.data.token_stream import canonical_sha256


def _inputs(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    population = tmp_path / "population"
    judgments = population / "judgments"
    judgments.mkdir(parents=True)
    candidates = [
        {
            "candidate_identity_sha256": "a" * 64,
            "source_content_sha256": "c" * 64,
            "source": {"row_id": "bad-row"},
        },
        {
            "candidate_identity_sha256": "b" * 64,
            "source_content_sha256": "d" * 64,
            "source": {"row_id": "good-row"},
        },
    ]
    lineage = [{"source_id": "example"}, {"source_id": "example"}]
    population_receipt = {"receipt_sha256": "e" * 64}
    monkeypatch.setattr(
        manifest,
        "load_population",
        lambda _root: (candidates, lineage, population_receipt),
    )

    receipts = []
    for candidate, verdict, risks in (
        (candidates[0], "reject", {"seo_or_content_farm": True}),
        (candidates[1], "retain", {}),
    ):
        judgment = {
            "judgment_sha256": ("f" if verdict == "reject" else "1") * 64,
            "verdict": verdict,
            "risks": risks,
        }
        receipt = {
            "candidate_identity_sha256": candidate["candidate_identity_sha256"],
            "receipt_sha256": ("2" if verdict == "reject" else "3") * 64,
            "judgment": judgment,
        }
        receipts.append(receipt)
        path = judgments / (
            f"{candidate['candidate_identity_sha256']}.compiler.json"
        )
        path.write_text(json.dumps(receipt))

    by_identity = {
        candidate["candidate_identity_sha256"]: receipt
        for candidate, receipt in zip(candidates, receipts, strict=True)
    }
    monkeypatch.setattr(
        manifest,
        "_validate_compiler_receipt",
        lambda receipt, candidate: by_identity[candidate["candidate_identity_sha256"]],
    )
    monkeypatch.setattr(
        manifest,
        "_triage_route",
        lambda judgment: "quarantine"
        if judgment["verdict"] == "reject"
        else "representation_verification",
    )

    aggregate = {
        "schema": AGGREGATE_SCHEMA,
        "status": "complete",
        "training_ready": False,
        "summary": {
            "rows": 2,
            "conservative_triage_routes": {"quarantine": 1},
            "model_judgments_are_verified_admissions": False,
            "representation_verification_is_training_admission": False,
        },
    }
    aggregate["receipt_sha256"] = canonical_sha256(aggregate)
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate))
    return population, judgments, aggregate_path


def test_quarantine_manifest_bars_exact_rows_without_source_text(
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
    assert rows[0]["source_row_id"] == "bad-row"
    assert rows[0]["dataset_materialization_allowed"] is False
    assert rows[0]["source_text_persisted"] is False
    assert "seo_or_content_farm" in rows[0]["active_risks"]
    assert result["quarantine_rows"] == 1
    assert result["training_ready"] is False


def test_quarantine_manifest_rejects_aggregate_count_tamper(
    tmp_path: Path, monkeypatch
) -> None:
    population, judgments, aggregate = _inputs(tmp_path, monkeypatch)
    value = json.loads(aggregate.read_text())
    value["summary"]["conservative_triage_routes"]["quarantine"] = 2
    value["receipt_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    aggregate.write_text(json.dumps(value))
    with pytest.raises(AuditQuarantineManifestError, match="coverage"):
        build_quarantine_manifest(
            population, judgments, aggregate, tmp_path / "output"
        )
    assert not (tmp_path / "output").exists()


def test_quarantine_manifest_rejects_extra_judgment(
    tmp_path: Path, monkeypatch
) -> None:
    population, judgments, aggregate = _inputs(tmp_path, monkeypatch)
    (judgments / f"{'9' * 64}.compiler.json").write_text("{}")
    with pytest.raises(AuditQuarantineManifestError, match="population"):
        build_quarantine_manifest(
            population, judgments, aggregate, tmp_path / "output"
        )
