from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.bridge_transfer_screen import (
    BridgeTransferScreenError,
    _chunks,
    _token_sha256,
    build_text_sets,
)
from sai.data.grounded_bridge_verification_population import SCHEMA as POPULATION_SCHEMA
from sai.data.practical_bridge_reconcile import SCHEMA as RECONCILIATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(payload: dict) -> dict:
    value = dict(payload)
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def _stream(root: Path, name: str, rows: list[dict]) -> dict:
    path = root / name
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return {
        "path": name,
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    reconciliation = tmp_path / "reconciliation"
    reconciliation.mkdir()
    train = [
        {
            "pair_identity_sha256": "1" * 64,
            "corpus_split": "train",
            "text": "A verified bridge lesson.",
        }
    ]
    development = [
        {
            "pair_identity_sha256": "2" * 64,
            "corpus_split": "development",
            "text": "A held-out bridge question and answer.",
        }
    ]
    outputs = {
        "train": _stream(reconciliation, "train.jsonl", train),
        "development": _stream(reconciliation, "development.jsonl", development),
    }
    _write_json(
        reconciliation / "receipt.json",
        _signed(
            {
                "schema": RECONCILIATION_SCHEMA,
                "status": "complete_practical_bridge_foundation_reconciliation",
                "outputs": outputs,
                "global_exact_content_deduplication_complete": True,
                "development_source_disjoint_against_foundation_complete": True,
                "transfer_ablation_complete": False,
                "training_ready": False,
            }
        ),
    )

    population = tmp_path / "population"
    population.mkdir()
    rows = []
    for pair, suffix in (("1" * 64, "train"), ("2" * 64, "development")):
        anchor_a = f"anchor A {suffix}"
        anchor_b = f"anchor B {suffix}"
        rows.append(
            {
                "pair_identity_sha256": pair,
                "candidate_identity_sha256": hashlib.sha256(pair.encode()).hexdigest(),
                "anchor_a_text": anchor_a,
                "anchor_b_text": anchor_b,
                "anchor_a_source_content_sha256": hashlib.sha256(
                    anchor_a.encode()
                ).hexdigest(),
                "anchor_b_source_content_sha256": hashlib.sha256(
                    anchor_b.encode()
                ).hexdigest(),
            }
        )
    descriptor = _stream(population, "candidates.jsonl", rows)
    descriptor["ordered_identities_sha256"] = canonical_sha256(
        [row["candidate_identity_sha256"] for row in rows]
    )
    _write_json(
        population / "receipt.json",
        _signed(
            {
                "schema": POPULATION_SCHEMA,
                "status": "complete_nontraining_bridge_verification_population",
                "source_disjoint_pairs": True,
                "candidates": descriptor,
            }
        ),
    )
    return reconciliation, population


def test_builds_source_disjoint_equal_compute_text_sets(tmp_path: Path) -> None:
    reconciliation, population = _fixture(tmp_path)
    sets, lineage = build_text_sets(reconciliation, population)
    assert sets["connection_train"] == ["A verified bridge lesson."]
    assert len(sets["source_control_train"]) == 2
    assert sets["connection_development"] == ["A held-out bridge question and answer."]
    assert len(sets["source_development"]) == 2
    assert lineage["train_pairs"] == 1
    assert lineage["development_pairs"] == 1
    assert len(set(lineage["ordered_text_sha256"].values())) == 4


def test_rejects_overlapping_pair_splits(tmp_path: Path) -> None:
    reconciliation, population = _fixture(tmp_path)
    path = reconciliation / "development.jsonl"
    row = json.loads(path.read_text())
    row["pair_identity_sha256"] = "1" * 64
    path.write_text(json.dumps(row, sort_keys=True) + "\n")
    receipt = json.loads((reconciliation / "receipt.json").read_text())
    receipt["outputs"]["development"]["bytes"] = path.stat().st_size
    receipt["outputs"]["development"]["sha256"] = sha256_file(path)
    receipt.pop("receipt_sha256")
    _write_json(reconciliation / "receipt.json", _signed(receipt))
    with pytest.raises(BridgeTransferScreenError, match="pair split"):
        build_text_sets(reconciliation, population)


def test_token_hash_and_chunk_geometry_are_exact() -> None:
    tokens = list(range(1_025))
    chunks, used = _chunks(tokens, 1_024)
    assert used == 1_024
    assert len(chunks) == 2
    assert _token_sha256(tokens[:used]) == _token_sha256(chunks[0] + chunks[1])
