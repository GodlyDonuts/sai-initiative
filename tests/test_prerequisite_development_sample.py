from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

import sai.data.prerequisite_sample as sample_module
from sai.data.curriculum import BANDS, PHASES
from sai.data.prerequisite_development_sample import (
    PrerequisiteDevelopmentSampleError,
    build_development_audit_population,
    validate_development_audit_population,
)
from sai.data.token_stream import (
    ROW_SCHEMA,
    canonical_sha256,
    normalize_document,
    sha256_file,
)


def _split(tmp_path: Path) -> Path:
    development = tmp_path / "development.jsonl"
    rows = []
    phases = {}
    index = 0
    for phase_index, phase in enumerate(PHASES):
        by_band = Counter()
        for band in BANDS:
            for item in range(3):
                row = normalize_document(
                    {
                        "schema": ROW_SCHEMA,
                        "text": f"band={band} phase={phase} item={item} "
                        + "clear instructional evidence " * 40,
                        "source": {
                            "dataset": "semantic-development-sample-test",
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
                index += 1
        phases[phase] = {
            "index": phase_index,
            "documents": len(BANDS) * 3,
            "by_band": dict(by_band),
            "mean_difficulty": float(phase_index),
            "identity_sha256": f"{phase_index + 1:064x}",
        }
    development.write_text("".join(json.dumps(row) + "\n" for row in rows))
    payload = {
        "schema": "sai-curriculum-train-development-split-v1",
        "status": "qualified",
        "split_qualified": True,
        "training_authorized": False,
        "four_b_training_authorized": False,
        "checks": {
            "all_curriculum_documents_emitted_once": True,
            "both_populations_have_every_phase": True,
            "exact_identity_assignment_disjoint": True,
            "train_progression_qualified": True,
        },
        "development": {
            "path": str(development.resolve()),
            "bytes": development.stat().st_size,
            "sha256": sha256_file(development),
            "documents": len(rows),
            "identity_sha256": "a" * 64,
            "curriculum_qualified": True,
            "phases": phases,
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt = tmp_path / "split.receipt.json"
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return receipt


def test_selects_and_replays_source_disjoint_development_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _split(tmp_path)
    monkeypatch.setattr(
        sample_module,
        "document_signals",
        lambda text: {"band": next(band for band in BANDS if f"band={band}" in text)},
    )
    output = tmp_path / "sample.jsonl"
    receipt = tmp_path / "sample.receipt.json"
    payload = build_development_audit_population(split, output, receipt, per_stratum=2)
    assert validate_development_audit_population(receipt) == payload
    assert payload["selection"]["selected_documents"] == 32
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert Counter((row["phase"], row["surface_band"]) for row in rows) == {
        (phase, band): 2 for phase in PHASES for band in BANDS
    }
    assert payload["limitations"][1] == (
        "development_split_is_source_disjoint_from_training"
    )
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False

    output.write_text(output.read_text() + "{}\n")
    with pytest.raises(
        PrerequisiteDevelopmentSampleError, match="population replay differs"
    ):
        validate_development_audit_population(receipt)


def test_rejects_resigned_split_or_incomplete_stratum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split = _split(tmp_path)
    payload = json.loads(split.read_text())
    tampered = deepcopy(payload)
    tampered["development"]["identity_sha256"] = "b" * 64
    split.write_text(json.dumps(tampered) + "\n")
    with pytest.raises(PrerequisiteDevelopmentSampleError, match="self hash"):
        build_development_audit_population(
            split, tmp_path / "sample.jsonl", tmp_path / "sample.receipt.json"
        )

    split = _split(tmp_path)
    monkeypatch.setattr(
        sample_module,
        "document_signals",
        lambda text: {"band": next(band for band in BANDS if f"band={band}" in text)},
    )
    with pytest.raises(PrerequisiteDevelopmentSampleError, match="stratum"):
        build_development_audit_population(
            split,
            tmp_path / "too-many.jsonl",
            tmp_path / "too-many.receipt.json",
            per_stratum=4,
        )
