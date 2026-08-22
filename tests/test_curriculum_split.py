from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sai.data.curriculum as curriculum
import sai.data.curriculum_split as curriculum_split
from sai.data.curriculum import BANDS, PHASES, build_curriculum
from sai.data.curriculum_split import (
    CurriculumSplitError,
    build_curriculum_split,
    validate_curriculum_split,
)
from sai.data.decontamination import RECEIPT_SCHEMA
from sai.data.token_stream import (
    ROW_SCHEMA,
    canonical_sha256,
    load_curriculum_phase_contract,
    sha256_file,
)


def _row(index: int, band: str) -> dict:
    payload = {
        "schema": ROW_SCHEMA,
        "text": f"{band} curriculum split document {index}. "
        + ("clear example words " * 40),
        "source": {
            "dataset": "synthetic-curriculum-split-test",
            "row_id": f"{band}-{index}",
            "license": "CC0",
            "domain": "english",
        },
        "verification": {
            "benchmark_disjoint": True,
            "evidence_sha256": hashlib.sha256(
                f"evidence-{band}-{index}".encode()
            ).hexdigest(),
        },
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def _signals(text: str) -> dict:
    band = next(band for band in BANDS if text.startswith(band))
    return {
        "quality_accepted": True,
        "quality_reasons": [],
        "difficulty": (BANDS.index(band) + 1) / 5,
        "band": band,
    }


def _sketch(text: str) -> tuple[int, ...]:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
    return tuple(seed + offset for offset in range(8))


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    monkeypatch.setattr(curriculum, "document_signals", _signals)
    monkeypatch.setattr(curriculum, "_near_duplicate_sketch", _sketch)
    monkeypatch.setattr(curriculum_split, "document_signals", _signals)
    source = tmp_path / "admitted.jsonl"
    rows = [_row(index, band) for band in BANDS for index in range(1_000)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    decontamination = tmp_path / "decontamination.json"
    decontamination_payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "output": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
    }
    decontamination_payload["receipt_sha256"] = canonical_sha256(
        decontamination_payload
    )
    decontamination.write_text(json.dumps(decontamination_payload) + "\n")
    curriculum_receipt = tmp_path / "curriculum.receipt.json"
    build_curriculum(
        source,
        decontamination,
        tmp_path / "curriculum.jsonl",
        curriculum_receipt,
        minimum_documents_per_band=1_000,
    )
    train = tmp_path / "train.jsonl"
    development = tmp_path / "development.jsonl"
    split_receipt = tmp_path / "split.receipt.json"
    payload = build_curriculum_split(
        curriculum_receipt, train, development, split_receipt
    )
    return split_receipt, payload


def test_split_is_exact_disjoint_progressive_and_replayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, payload = _fixture(tmp_path, monkeypatch)
    assert payload["status"] == "qualified", payload
    assert payload["split_qualified"] is True
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False
    assert payload["train"]["documents"] + payload["development"]["documents"] == 4_000
    assert (
        payload["train"]["identity_sha256"] != payload["development"]["identity_sha256"]
    )
    assert list(payload["train"]["phases"]) == list(PHASES)
    assert list(payload["development"]["phases"]) == list(PHASES)
    assert all(payload["checks"].values())
    assert validate_curriculum_split(receipt) == payload
    assert validate_curriculum_split(receipt, curriculum_workers=2) == payload
    assert load_curriculum_phase_contract(
        receipt,
        [Path(payload["train"]["path"])],
        sha256_file(receipt),
    ) == [(phase, payload["train"]["phases"][phase]["documents"]) for phase in PHASES]


def test_split_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, payload = _fixture(tmp_path, monkeypatch)
    train = Path(payload["train"]["path"])
    train.write_text(train.read_text() + "{}\n")
    with pytest.raises(CurriculumSplitError, match="output differs"):
        validate_curriculum_split(receipt)


def test_split_and_development_stream_jobs_are_cpu_only_and_create_only() -> None:
    root = Path(__file__).resolve().parents[1]
    split = (root / "jobs" / "sai-split-500m-curriculum-cpu.sbatch").read_text()
    development = (
        root / "jobs" / "sai-freeze-500m-development-stream-cpu.sbatch"
    ).read_text()
    for job in (split, development):
        assert "#SBATCH --no-requeue" in job
        assert "#SBATCH --gres=" not in job
        assert 'rev-parse HEAD)" = "$EXPECTED_COMMIT"' in job
        assert (
            "sbatch "
            not in "\n".join(
                line for line in job.splitlines() if not line.startswith("#SBATCH")
            ).lower()
        )
        assert "scancel" not in job.lower()
    assert "sai.data.curriculum_split build" in split
    assert "sai.data.curriculum_split validate" in split
    assert split.count('--curriculum-workers "$SLURM_CPUS_PER_TASK"') == 2
    assert 'test ! -e "$TRAIN_CORPUS"' in split
    assert 'test ! -e "$DEVELOPMENT_CORPUS"' in split
    assert 'test ! -e "$SPLIT_RECEIPT"' in split
    assert "validate_curriculum_split" in development
    assert 'test ! -e "$DEVELOPMENT_STREAM"' in development
    assert "--prefix-sequences 1024" in development
    assert 'stream["sequences"] == 1_024' in development
