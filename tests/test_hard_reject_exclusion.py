from __future__ import annotations

import json
from pathlib import Path

import pytest

import sai.data.hard_reject_exclusion as exclusion
from sai.data.hard_reject_exclusion import (
    HardRejectExclusionError,
    build_hard_reject_exclusion,
)
from sai.data.token_stream import ROW_SCHEMA, canonical_sha256, sha256_file


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _document(row_id: str, text: str) -> dict:
    value = {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": "example/books",
            "row_id": row_id,
            "license": "CC-BY-4.0",
            "domain": "english",
        },
        "verification": {"benchmark_disjoint": True, "evidence_sha256": "e" * 64},
    }
    value["identity_sha256"] = canonical_sha256(value)
    return value


def _attribution(row_id: str, identity: str) -> dict:
    value = {
        "schema": "sai-document-attribution-manifest-v1",
        "row_id": row_id,
        "identity_sha256": identity,
        "source": {"dataset": "example/books", "row_index": 0},
        "rights_declaration": {"canonical_license": "CC-BY-4.0"},
    }
    value["record_sha256"] = canonical_sha256(value)
    return value


def _evidence(tmp_path: Path) -> tuple[Path, Path, str]:
    pilot = tmp_path / "pilot"
    judgments = pilot / "judgments"
    judgments.mkdir(parents=True)
    reject_identity = "a" * 64
    keep_identity = "b" * 64
    candidates = [
        {
            "schema": "sai-agent-data-candidate-v1",
            "candidate_identity_sha256": reject_identity,
            "source_content_sha256": "c" * 64,
            "source": {"row_id": "reject-row"},
        },
        {
            "schema": "sai-agent-data-candidate-v1",
            "candidate_identity_sha256": keep_identity,
            "source_content_sha256": "d" * 64,
            "source": {"row_id": "keep-row"},
        },
    ]
    lineage = [
        {
            "candidate_identity_sha256": reject_identity,
            "row_id": "reject-row",
            "source_id": "example_books",
            "source_content_sha256": "c" * 64,
        },
        {
            "candidate_identity_sha256": keep_identity,
            "row_id": "keep-row",
            "source_id": "example_books",
            "source_content_sha256": "d" * 64,
        },
    ]
    _write(pilot / "candidates.jsonl", candidates)
    _write(pilot / "lineage.jsonl", lineage)
    receipt = {
        "schema": "sai-bounded-pilot-compiler-population-v1",
        "status": "complete_nontraining_compiler_population",
        "population": {
            "bytes": (pilot / "candidates.jsonl").stat().st_size,
            "sha256": sha256_file(pilot / "candidates.jsonl"),
        },
        "lineage": {
            "bytes": (pilot / "lineage.jsonl").stat().st_size,
            "sha256": sha256_file(pilot / "lineage.jsonl"),
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (pilot / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    for identity, verdict in ((reject_identity, "reject"), (keep_identity, "retain")):
        judgment = {
            "schema": "sai-nous-data-compiler-receipt-v2",
            "status": "complete",
            "candidate_identity_sha256": identity,
            "judgment": {
                "candidate_identity_sha256": identity,
                "verdict": verdict,
                "curriculum_phase": verdict,
                "preservation_policy": verdict,
            },
        }
        judgment["receipt_sha256"] = canonical_sha256(judgment)
        (judgments / f"{identity}.compiler.json").write_text(
            json.dumps(judgment, sort_keys=True)
        )
    return pilot, judgments, reject_identity


def test_hard_reject_rows_are_removed_with_text_free_evidence(tmp_path: Path) -> None:
    pilot, judgments, reject_identity = _evidence(tmp_path)
    rejected = _document("reject-row", "Private low-quality worksheet content.")
    retained = _document("keep-row", "A useful explanation of orbital mechanics.")
    candidates = tmp_path / "materialized.jsonl"
    attribution = tmp_path / "attribution.jsonl"
    _write(candidates, [rejected, retained])
    _write(
        attribution,
        [
            _attribution("reject-row", rejected["identity_sha256"]),
            _attribution("keep-row", retained["identity_sha256"]),
        ],
    )
    output = tmp_path / "output" / "candidates.jsonl"
    output_attribution = tmp_path / "output" / "attribution.jsonl"
    manifest = tmp_path / "output" / "exclusions.jsonl"
    result = build_hard_reject_exclusion(
        [candidates],
        [attribution],
        pilot,
        judgments,
        output,
        output_attribution,
        manifest,
        tmp_path / "output" / "receipt.json",
    )
    assert [json.loads(line) for line in output.read_text().splitlines()] == [retained]
    assert len(output_attribution.read_text().splitlines()) == 1
    exclusion = json.loads(manifest.read_text())
    assert exclusion["pilot_candidate_identity_sha256"] == reject_identity
    assert exclusion["source_text_persisted"] is False
    assert "Private low-quality" not in manifest.read_text()
    assert result["counts"]["excluded_candidate_rows"] == 1
    assert result["counts"]["excluded_attribution_rows"] == 1
    assert result["training_ready"] is False


def test_hard_reject_requires_exact_candidate_and_attribution_coverage(
    tmp_path: Path,
) -> None:
    pilot, judgments, _ = _evidence(tmp_path)
    retained = _document("keep-row", "A useful explanation.")
    candidates = tmp_path / "materialized.jsonl"
    attribution = tmp_path / "attribution.jsonl"
    _write(candidates, [retained])
    _write(attribution, [_attribution("keep-row", retained["identity_sha256"])])
    with pytest.raises(HardRejectExclusionError, match="coverage differs"):
        build_hard_reject_exclusion(
            [candidates],
            [attribution],
            pilot,
            judgments,
            tmp_path / "output" / "candidates.jsonl",
            tmp_path / "output" / "attribution.jsonl",
            tmp_path / "output" / "exclusions.jsonl",
            tmp_path / "output" / "receipt.json",
        )


def test_hard_reject_detects_input_change_during_replay(
    tmp_path: Path, monkeypatch
) -> None:
    pilot, judgments, _ = _evidence(tmp_path)
    rejected = _document("reject-row", "A low-quality worksheet.")
    retained = _document("keep-row", "A useful explanation.")
    candidates = tmp_path / "materialized.jsonl"
    attribution = tmp_path / "attribution.jsonl"
    _write(candidates, [rejected, retained])
    _write(
        attribution,
        [
            _attribution("reject-row", rejected["identity_sha256"]),
            _attribution("keep-row", retained["identity_sha256"]),
        ],
    )
    original = exclusion.normalize_document
    changed = False

    def mutate_after_binding(value: dict) -> dict:
        nonlocal changed
        if not changed:
            with candidates.open("a") as handle:
                handle.write("\n")
            changed = True
        return original(value)

    monkeypatch.setattr(exclusion, "normalize_document", mutate_after_binding)
    with pytest.raises(HardRejectExclusionError, match="changed during replay"):
        build_hard_reject_exclusion(
            [candidates],
            [attribution],
            pilot,
            judgments,
            tmp_path / "output" / "candidates.jsonl",
            tmp_path / "output" / "attribution.jsonl",
            tmp_path / "output" / "exclusions.jsonl",
            tmp_path / "output" / "receipt.json",
        )
    assert not (tmp_path / "output" / "candidates.jsonl").exists()
