from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.data_yield_ledger import DataYieldLedgerError, build_ledger
from sai.data.token_stream import canonical_sha256, sha256_file


def _seal(path: Path, payload: dict) -> dict:
    payload = dict(payload)
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    return payload


def _reservoir(tmp_path: Path) -> Path:
    root = tmp_path / "reservoir"
    root.mkdir()
    manifest = root / "manifest.jsonl"
    manifest.write_text('{"row":1}\n')
    receipt = root / "receipt.json"
    _seal(
        receipt,
        {
            "schema": "sai-source-reservoir-receipt-v1",
            "selected_bytes": 1234,
            "selected_files": 1,
            "manifest": {
                "path": manifest.name,
                "bytes": manifest.stat().st_size,
                "sha256": sha256_file(manifest),
            },
            "training_ready": False,
        },
    )
    return receipt


def _audit(tmp_path: Path) -> Path:
    root = tmp_path / "audit"
    root.mkdir()
    population = root / "candidates.jsonl"
    lineage = root / "lineage.jsonl"
    population.write_text('{"candidate":1}\n')
    lineage.write_text('{"lineage":1}\n')
    _seal(
        root / "receipt.json",
        {
            "schema": "sai-reservoir-audit-population-receipt-v1",
            "population": {
                "path": population.name,
                "rows": 1,
                "bytes": population.stat().st_size,
                "sha256": sha256_file(population),
            },
            "lineage": {
                "path": lineage.name,
                "rows": 1,
                "bytes": lineage.stat().st_size,
                "sha256": sha256_file(lineage),
            },
            "by_source": {"source": 1},
            "fully_verified_compressed_parent_bytes": 99,
            "training_ready": False,
        },
    )
    return root


def test_ledger_separates_candidate_volume_from_ready_bytes(tmp_path: Path) -> None:
    output = tmp_path / "ledger.json"
    payload = build_ledger([_reservoir(tmp_path)], [_audit(tmp_path)], [], output)
    assert payload["reservoir_candidates"]["referenced_candidate_bytes_sum"] == 1234
    assert payload["audit_populations"]["population_rows_sum"] == 1
    assert payload["training_ready"]["exact_bytes"] == 0
    assert payload["claims"]["raw_reservoir_bytes_are_not_training_ready_bytes"]


def test_ledger_rejects_tampered_receipt(tmp_path: Path) -> None:
    receipt = _reservoir(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["selected_bytes"] += 1
    receipt.write_text(json.dumps(payload))
    with pytest.raises(DataYieldLedgerError, match="receipt hash differs"):
        build_ledger([receipt], [], [], tmp_path / "ledger.json")
