from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

import sai.data.prerequisite_sample as sample_module
from sai.data.curriculum import BANDS, PHASES
from sai.data.prerequisite_sample import (
    PrerequisiteSampleError,
    build_audit_population,
    validate_audit_population,
)
from sai.data.token_stream import ROW_SCHEMA, normalize_document, sha256_file


def _curriculum(tmp_path: Path) -> tuple[Path, dict]:
    output = tmp_path / "curriculum.jsonl"
    rows = []
    phases = {}
    index = 0
    for phase_index, phase in enumerate(PHASES):
        by_band = Counter()
        identities = []
        for band in BANDS:
            for item in range(3):
                row = normalize_document(
                    {
                        "schema": ROW_SCHEMA,
                        "text": f"band={band} phase={phase} item={item} "
                        + "clear instructional evidence " * 40,
                        "source": {
                            "dataset": "semantic-sample-test",
                            "row_id": str(index),
                            "license": "CC0",
                            "domain": "english",
                        },
                        "verification": {
                            "benchmark_disjoint": True,
                            "evidence_sha256": f"{index + 100:064x}",
                        },
                    }
                )
                rows.append(row)
                by_band[band] += 1
                identities.append(row["identity_sha256"])
                index += 1
        phases[phase] = {
            "index": phase_index,
            "documents": len(BANDS) * 3,
            "by_band": dict(by_band),
            "mean_difficulty": float(phase_index),
            "identity_sha256": f"{phase_index + 1:064x}",
        }
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt = tmp_path / "curriculum.receipt.json"
    receipt.write_text('{"curriculum":"test"}\n')
    payload = {
        "receipt_sha256": "a" * 64,
        "output": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "phases": phases,
    }
    return receipt, payload


def test_selects_replays_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curriculum_receipt, curriculum = _curriculum(tmp_path)
    monkeypatch.setattr(
        sample_module,
        "validate_curriculum",
        lambda path, workers: curriculum,
    )
    monkeypatch.setattr(
        sample_module,
        "document_signals",
        lambda text: {"band": next(band for band in BANDS if f"band={band}" in text)},
    )
    output = tmp_path / "sample.jsonl"
    receipt = tmp_path / "sample.receipt.json"
    payload = build_audit_population(
        curriculum_receipt,
        output,
        receipt,
        per_stratum=2,
        curriculum_workers=3,
    )
    assert validate_audit_population(receipt, curriculum_workers=3) == payload
    assert payload["selection"]["selected_documents"] == 32
    assert payload["selection"]["strata"] == 16
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert Counter((row["phase"], row["surface_band"]) for row in rows) == {
        (phase, band): 2 for phase in PHASES for band in BANDS
    }
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False

    tampered = deepcopy(payload)
    tampered["extra"] = "resigned-extension"
    tampered["receipt_sha256"] = sample_module.canonical_sha256(
        {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    )
    receipt.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(PrerequisiteSampleError, match="receipt differs"):
        validate_audit_population(receipt, curriculum_workers=3)
    receipt.write_text(json.dumps(payload) + "\n")

    with pytest.raises(PrerequisiteSampleError, match="already exists"):
        build_audit_population(curriculum_receipt, output, receipt, per_stratum=2)
    output.write_text(output.read_text() + "{}\n")
    with pytest.raises(PrerequisiteSampleError, match="replay differs"):
        validate_audit_population(receipt, curriculum_workers=3)


def test_rejects_incomplete_stratum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    curriculum_receipt, curriculum = _curriculum(tmp_path)
    monkeypatch.setattr(
        sample_module,
        "validate_curriculum",
        lambda path, workers: curriculum,
    )
    monkeypatch.setattr(
        sample_module,
        "document_signals",
        lambda text: {"band": next(band for band in BANDS if f"band={band}" in text)},
    )
    with pytest.raises(PrerequisiteSampleError, match="stratum is incomplete"):
        build_audit_population(
            curriculum_receipt,
            tmp_path / "sample.jsonl",
            tmp_path / "sample.receipt.json",
            per_stratum=4,
        )
