from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sai.data.curriculum as curriculum
from sai.data.curriculum import (
    BANDS,
    PHASES,
    CurriculumError,
    build_curriculum,
    document_signals,
    validate_curriculum,
)
from sai.data.decontamination import RECEIPT_SCHEMA
from sai.data.token_stream import ROW_SCHEMA, canonical_sha256, sha256_file


def _row(index: int, band: str) -> dict:
    payload = {
        "schema": ROW_SCHEMA,
        "text": f"{band} curriculum document {index}. " + ("clear example words " * 40),
        "source": {
            "dataset": "synthetic-curriculum-test",
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


def _source_receipt(source: Path, receipt: Path) -> None:
    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "output": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n")


def test_difficulty_signals_separate_prerequisites_from_specialization() -> None:
    foundation = (
        "Color is a property used to describe how things look. A yellow object "
        "reflects yellow light. A blue object reflects blue light. When yellow "
        "paint and blue paint are mixed, they usually make green paint. This "
        "example explains how familiar things combine to make something new. "
    ) * 3
    composition = (
        "Plants need light, water, and air to grow. Because leaves collect "
        "sunlight, they make food for the plant. However, roots absorb water "
        "and minerals from soil. For example, a plant kept in darkness may "
        "have water but cannot continue healthy growth. "
    ) * 4
    reasoning = (
        "Assume a population doubles every three years. Therefore its size "
        "after n periods follows the equation P(n) = P0 * 2**n. To derive the "
        "result, compare each period with evidence. If the rate changes, "
        "however, the hypothesis must be revised. "
    ) * 5
    specialization_parts = []
    for index in range(4):
        specialization_parts.append(
            "The algorithm computes the eigendecomposition of covariance matrix "
            f"A{index} in R^{{d x d}}; consequently, vector x{index} maximizes "
            "x^T A x subject to ||x||_2 = 1. We derive the gradient, implement "
            "a compiler kernel, and analyze stochastic convergence, numerical "
            "conditioning, and asymptotic complexity."
        )
    specialization = " ".join(specialization_parts)

    texts = (foundation, composition, reasoning, specialization)
    assert [document_signals(text)["band"] for text in texts] == list(BANDS)
    assert all(document_signals(text)["quality_accepted"] for text in texts)


def _patched_signals(text: str) -> dict:
    band = next(band for band in BANDS if text.startswith(band))
    return {
        "quality_accepted": True,
        "quality_reasons": [],
        "difficulty": (BANDS.index(band) + 1) / 5,
        "band": band,
    }


def _patched_sketch(text: str) -> tuple[int, ...]:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)
    return tuple(seed + offset for offset in range(8))


def _build_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict]:
    source = tmp_path / "admitted.jsonl"
    rows = [_row(index, band) for band in BANDS for index in range(10)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    decontamination = tmp_path / "decontamination.json"
    _source_receipt(source, decontamination)
    monkeypatch.setattr(curriculum, "document_signals", _patched_signals)
    monkeypatch.setattr(curriculum, "_near_duplicate_sketch", _patched_sketch)
    output = tmp_path / "curriculum.jsonl"
    receipt = tmp_path / "curriculum.receipt.json"
    payload = build_curriculum(
        source,
        decontamination,
        output,
        receipt,
        minimum_documents_per_band=10,
    )
    return output, receipt, payload


def test_curriculum_builds_all_four_progressive_phases_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, receipt, payload = _build_fixture(tmp_path, monkeypatch)

    assert payload["status"] == "qualified"
    assert payload["curriculum_qualified"] is True
    assert payload["training_authorized"] is False
    assert payload["documents"]["accepted_by_band"] == {band: 10 for band in BANDS}
    assert payload["documents"]["all_accepted_emitted_once"] is True
    assert list(payload["phases"]) == list(PHASES)
    means = [payload["phases"][phase]["mean_difficulty"] for phase in PHASES]
    assert means == sorted(means)
    assert payload["phases"]["grounding"]["by_band"]["specialization"] == 0
    assert payload["phases"]["specialization"]["by_band"]["foundation"] > 0
    assert len(output.read_text().splitlines()) == 40
    assert validate_curriculum(receipt) == payload
    assert validate_curriculum(receipt, workers=2) == payload


def test_parallel_scoring_preserves_exact_curriculum_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "admitted.jsonl"
    rows = [_row(index, band) for band in BANDS for index in range(10)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    decontamination = tmp_path / "decontamination.json"
    _source_receipt(source, decontamination)
    monkeypatch.setattr(curriculum, "document_signals", _patched_signals)
    monkeypatch.setattr(curriculum, "_near_duplicate_sketch", _patched_sketch)
    serial = tmp_path / "serial.jsonl"
    parallel = tmp_path / "parallel.jsonl"
    build_curriculum(
        source,
        decontamination,
        serial,
        tmp_path / "serial.receipt.json",
        minimum_documents_per_band=10,
        workers=1,
    )
    build_curriculum(
        source,
        decontamination,
        parallel,
        tmp_path / "parallel.receipt.json",
        minimum_documents_per_band=10,
        workers=2,
    )
    assert serial.read_bytes() == parallel.read_bytes()


def test_high_confidence_near_duplicate_sketch_is_rejected() -> None:
    base = (
        "A careful lesson introduces one idea before combining it with another. "
        "The learner sees a concrete example and then practices the relationship. "
    ) * 10
    changed = base.replace("concrete example", "specific example", 1)
    first = curriculum._near_duplicate_sketch(base)
    second = curriculum._near_duplicate_sketch(changed)
    accepted: list[tuple[int, ...]] = []
    index: dict[tuple[int, int, int], set[int]] = {}
    assert curriculum._is_near_duplicate(first, accepted, index) is False
    curriculum._add_sketch(first, accepted, index)
    assert curriculum._is_near_duplicate(second, accepted, index) is True


def test_curriculum_tamper_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, receipt, _ = _build_fixture(tmp_path, monkeypatch)
    output.write_text(output.read_text() + "{}\n")
    with pytest.raises(CurriculumError, match="output differs"):
        validate_curriculum(receipt)
