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


def _pilot(tmp_path: Path) -> Path:
    root = tmp_path / "pilot"
    root.mkdir()
    raw = root / "raw.jsonl"
    decontaminated = root / "decontaminated.jsonl"
    near_deduplicated = root / "near-deduplicated.jsonl"
    attribution = root / "attribution.jsonl"
    raw.write_text('{"raw":1}\n')
    decontaminated.write_text('{"clean":1}\n')
    near_deduplicated.write_text('{"deduplicated":1}\n')
    attribution.write_text('{"attribution":1}\n')
    decontamination_receipt = _seal(
        root / "decontamination-receipt.json",
        {"schema": "test-decontamination", "output": {"documents": 1}},
    )
    near_duplicate_receipt = _seal(
        root / "near-duplicate-receipt.json",
        {"schema": "test-near-duplicate", "output": {"documents": 1}},
    )
    attribution_receipt = _seal(
        root / "attribution-receipt.json",
        {"schema": "test-attribution", "output": {"records": 1}},
    )
    _seal(
        root / "receipt.json",
        {
            "schema": "sai-common-pile-streaming-pilot-v1",
            "source_id": "common_pile_test",
            "raw_population": {
                "path": raw.name,
                "rows": 1,
                "bytes": raw.stat().st_size,
                "sha256": sha256_file(raw),
            },
            "decontamination": {
                "receipt_path": "decontamination-receipt.json",
                "receipt_file_sha256": sha256_file(
                    root / "decontamination-receipt.json"
                ),
                "receipt_sha256": decontamination_receipt["receipt_sha256"],
                "output_path": decontaminated.name,
                "output_documents": 1,
                "output_bytes": decontaminated.stat().st_size,
                "output_sha256": sha256_file(decontaminated),
            },
            "near_duplicate_filter": {
                "receipt_path": "near-duplicate-receipt.json",
                "receipt_file_sha256": sha256_file(
                    root / "near-duplicate-receipt.json"
                ),
                "receipt_sha256": near_duplicate_receipt["receipt_sha256"],
                "output_path": near_deduplicated.name,
                "output_documents": 1,
                "output_bytes": near_deduplicated.stat().st_size,
                "output_sha256": sha256_file(near_deduplicated),
                "documents_dropped": 0,
            },
            "attribution_manifest": {
                "receipt_path": "attribution-receipt.json",
                "receipt_file_sha256": sha256_file(
                    root / "attribution-receipt.json"
                ),
                "receipt_sha256": attribution_receipt["receipt_sha256"],
                "output_path": attribution.name,
                "output_bytes": attribution.stat().st_size,
                "output_sha256": sha256_file(attribution),
                "records": 1,
                "obligation_counts": {
                    "attribution_required": 1,
                    "share_alike_required": 0,
                },
            },
            "rights_declaration_lineage_replay_complete": True,
            "global_cross_source_near_duplicate_filter_complete": False,
            "rights_verification_complete": False,
            "representation_verification_complete": False,
            "training_ready": False,
        },
    )
    return root


def _rights(tmp_path: Path) -> Path:
    path = tmp_path / "rights.json"
    _seal(
        path,
        {
            "schema": "sai-reservoir-rights-inventory-v2",
            "source_rows": [
                {
                    "source_id": "source",
                    "bytes": 1234,
                    "rights_work_route": (
                        "recognized_declaration_obligations_required"
                    ),
                }
            ],
            "summary": {"sources": 1, "physical_candidate_bytes": 1234},
            "training_ready": False,
        },
    )
    return path


def test_ledger_separates_candidate_volume_from_ready_bytes(tmp_path: Path) -> None:
    output = tmp_path / "ledger.json"
    payload = build_ledger(
        [_reservoir(tmp_path)],
        [_audit(tmp_path)],
        [_pilot(tmp_path)],
        output,
        rights_inventory_path=_rights(tmp_path),
    )
    assert payload["reservoir_candidates"]["referenced_candidate_bytes_sum"] == 1234
    assert payload["audit_populations"]["population_rows_sum"] == 1
    assert payload["bounded_source_pilots"]["near_deduplicated_rows_sum"] == 1
    assert payload["rights_routing"]["physical_candidate_bytes"] == 1234
    assert payload["training_ready"]["exact_bytes"] == 0
    assert payload["claims"]["raw_reservoir_bytes_are_not_training_ready_bytes"]


def test_ledger_rejects_tampered_receipt(tmp_path: Path) -> None:
    receipt = _reservoir(tmp_path)
    payload = json.loads(receipt.read_text())
    payload["selected_bytes"] += 1
    receipt.write_text(json.dumps(payload))
    with pytest.raises(DataYieldLedgerError, match="receipt hash differs"):
        build_ledger([receipt], [], [], tmp_path / "ledger.json")


def test_ledger_rejects_tampered_nested_pilot_receipt(tmp_path: Path) -> None:
    reservoir = _reservoir(tmp_path)
    pilot = _pilot(tmp_path)
    nested = pilot / "near-duplicate-receipt.json"
    payload = json.loads(nested.read_text())
    payload["output"]["documents"] = 2
    nested.write_text(json.dumps(payload))
    with pytest.raises(DataYieldLedgerError, match="receipt hash differs"):
        build_ledger([reservoir], [], [pilot], tmp_path / "ledger.json")


def test_ledger_rejects_rights_bytes_that_do_not_cover_reservoir(
    tmp_path: Path,
) -> None:
    reservoir = _reservoir(tmp_path)
    rights = _rights(tmp_path)
    payload = json.loads(rights.read_text())
    payload.pop("receipt_sha256")
    payload["source_rows"][0]["bytes"] = 1233
    payload["summary"]["physical_candidate_bytes"] = 1233
    _seal(rights, payload)
    with pytest.raises(DataYieldLedgerError, match="rights and reservoir bytes differ"):
        build_ledger(
            [reservoir],
            [],
            [],
            tmp_path / "ledger.json",
            rights_inventory_path=rights,
        )


def test_ledger_rejects_repeated_audit_population(tmp_path: Path) -> None:
    reservoir = _reservoir(tmp_path)
    audit = _audit(tmp_path)
    with pytest.raises(DataYieldLedgerError, match="repeats an audit population"):
        build_ledger(
            [reservoir],
            [audit, audit],
            [],
            tmp_path / "ledger.json",
        )
