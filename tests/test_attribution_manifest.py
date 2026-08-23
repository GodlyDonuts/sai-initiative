from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.attribution_manifest import AttributionManifestError, build_manifest
from sai.data.decontamination import RAW_SCHEMA
from sai.data.token_stream import ROW_SCHEMA, canonical_sha256


def _raw() -> dict:
    return {
        "schema": RAW_SCHEMA,
        "text": "A grounded technical explanation with enough source context.",
        "source": {
            "dataset": "common-pile/example",
            "revision": "a" * 40,
            "source_file": "part-0000.json.gz",
            "row_index": 17,
            "license": "CC-BY-SA-4.0",
            "declared_license": "CC-BY-SA-4.0",
            "domain": "technical",
        },
    }


def _retained(raw: dict, *, license_name: str = "CC-BY-SA-4.0") -> dict:
    source = raw["source"]
    row_id = canonical_sha256(
        {
            "dataset": source["dataset"],
            "revision": source["revision"],
            "source_file": source["source_file"],
            "row_index": source["row_index"],
        }
    )
    payload = {
        "schema": ROW_SCHEMA,
        "text": raw["text"],
        "source": {
            "dataset": source["dataset"],
            "row_id": row_id,
            "license": license_name,
            "domain": source["domain"],
        },
        "verification": {"benchmark_disjoint": True, "evidence_sha256": "f" * 64},
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def _write(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n")


def test_manifest_replays_exact_lineage_and_obligations(tmp_path: Path) -> None:
    raw_row = _raw()
    raw = tmp_path / "raw.jsonl"
    retained = tmp_path / "retained.jsonl"
    output = tmp_path / "attribution.jsonl"
    receipt = tmp_path / "receipt.json"
    _write(raw, raw_row)
    _write(retained, _retained(raw_row))
    result = build_manifest(raw, retained, output, receipt)
    record = json.loads(output.read_text())
    assert "text" not in record
    assert record["source"]["revision"] == "a" * 40
    assert record["source"]["row_index"] == 17
    assert record["rights_declaration"]["attribution_required"] is True
    assert record["rights_declaration"]["share_alike_required"] is True
    assert result["exact_retained_document_coverage"] is True
    assert result["external_source_provenance_verified"] is False
    assert result["training_ready"] is False


def test_manifest_rejects_changed_retained_license(tmp_path: Path) -> None:
    raw_row = _raw()
    raw = tmp_path / "raw.jsonl"
    retained = tmp_path / "retained.jsonl"
    _write(raw, raw_row)
    _write(retained, _retained(raw_row, license_name="MIT"))
    with pytest.raises(
        AttributionManifestError, match="provenance or license differs"
    ):
        build_manifest(
            raw,
            retained,
            tmp_path / "attribution.jsonl",
            tmp_path / "receipt.json",
        )
