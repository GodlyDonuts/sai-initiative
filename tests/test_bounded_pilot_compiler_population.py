from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.agent_labeling import normalize_candidate
from sai.data.attribution_manifest import SCHEMA as ATTRIBUTION_SCHEMA
from sai.data.bounded_pilot_compiler_population import (
    BoundedPilotCompilerPopulationError,
    build_population,
)
from sai.data.common_pile_streaming_pilot import SCHEMA as PILOT_SCHEMA
from sai.data.cross_source_pilot_duplicates import SCHEMA as CROSS_SOURCE_SCHEMA
from sai.data.token_stream import ROW_SCHEMA, canonical_sha256, sha256_file


def _seal(path: Path, payload: dict) -> dict:
    payload = dict(payload)
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    return payload


def _document(source_id: str, index: int) -> dict:
    dataset = f"common-pile/{source_id}"
    row_id = canonical_sha256({"source_id": source_id, "index": index})
    payload = {
        "schema": ROW_SCHEMA,
        "text": (
            f"{source_id} document {index} teaches a grounded historical concept "
            "with enough distinct context for strict compiler validation. " * 4
        ),
        "source": {
            "dataset": dataset,
            "row_id": row_id,
            "license": "CC-BY-4.0",
            "domain": "english",
        },
        "verification": {
            "benchmark_disjoint": True,
            "evidence_sha256": canonical_sha256(
                {"source_id": source_id, "index": index, "clean": True}
            ),
        },
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def _attribution(document: dict, source_id: str, index: int) -> dict:
    record = {
        "schema": ATTRIBUTION_SCHEMA,
        "identity_sha256": document["identity_sha256"],
        "row_id": document["source"]["row_id"],
        "source": {
            "dataset": document["source"]["dataset"],
            "revision": "a" * 40,
            "source_file": f"part-{index:04d}.json.gz",
            "row_index": index,
            "domain": "english",
        },
        "rights_declaration": {
            "declared_license": "CC BY 4.0",
            "canonical_license": "CC-BY-4.0",
            "classification_sha256": "b" * 64,
            "attribution_required": True,
            "share_alike_required": False,
            "rights_hold": False,
        },
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def _pilot(tmp_path: Path, source_id: str, document: dict, index: int) -> Path:
    root = tmp_path / source_id
    root.mkdir()
    attribution = root / "attribution.jsonl"
    attribution.write_text(json.dumps(_attribution(document, source_id, index)) + "\n")
    _seal(
        root / "receipt.json",
        {
            "schema": PILOT_SCHEMA,
            "status": "complete_nontraining_pilot",
            "source_id": source_id,
            "attribution_manifest": {
                "output_path": attribution.name,
                "output_bytes": attribution.stat().st_size,
                "output_sha256": sha256_file(attribution),
                "records": 1,
            },
            "training_ready": False,
        },
    )
    return root


def _cross_source(tmp_path: Path, pilots: list[Path], documents: list[dict]) -> Path:
    root = tmp_path / "cross"
    root.mkdir()
    output = root / "deduplicated.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in documents))
    bindings = []
    for pilot in pilots:
        receipt = json.loads((pilot / "receipt.json").read_text())
        bindings.append(
            {
                "source_id": receipt["source_id"],
                "receipt_sha256": receipt["receipt_sha256"],
            }
        )
    _seal(
        root / "receipt.json",
        {
            "schema": CROSS_SOURCE_SCHEMA,
            "status": "complete_nontraining_cross_source_sample",
            "pilot_bindings": bindings,
            "duplicate_filter": {
                "input_documents": len(documents),
                "output_documents": len(documents),
                "documents_dropped": 0,
                "duplicate_groups": 0,
                "cross_source_duplicate_groups": 0,
                "output_path": output.name,
                "output_bytes": output.stat().st_size,
                "output_sha256": sha256_file(output),
            },
            "bounded_cross_source_pilot_sample_complete": True,
            "full_pilot_population_cross_source_deduplication_complete": True,
            "training_ready": False,
        },
    )
    return root


def _inputs(tmp_path: Path) -> tuple[list[Path], Path]:
    rows = [
        _document("common_pile_pressbooks", 0),
        _document("common_pile_public_domain_review", 1),
    ]
    pilots = [
        _pilot(tmp_path, "common_pile_pressbooks", rows[0], 0),
        _pilot(tmp_path, "common_pile_public_domain_review", rows[1], 1),
    ]
    return pilots, _cross_source(tmp_path, pilots, rows)


def test_builds_exact_attribution_bound_compiler_population(tmp_path: Path) -> None:
    pilots, cross_source = _inputs(tmp_path)
    output = tmp_path / "compiler"
    receipt = build_population(pilots, cross_source, output)
    candidates = [
        normalize_candidate(json.loads(line))
        for line in (output / "candidates.jsonl").read_text().splitlines()
    ]
    lineage = [
        json.loads(line) for line in (output / "lineage.jsonl").read_text().splitlines()
    ]
    assert len(candidates) == len(lineage) == 2
    assert receipt["by_source"] == {
        "common_pile_pressbooks": 1,
        "common_pile_public_domain_review": 1,
    }
    assert {row["source"]["source_type"] for row in candidates} == {
        "textbook",
        "educational_web",
    }
    assert all("text" not in row for row in lineage)
    assert receipt["compiler_judgments_complete"] is False
    assert receipt["training_ready"] is False


def test_rejects_cross_source_receipt_with_wrong_pilot_binding(
    tmp_path: Path,
) -> None:
    pilots, cross_source = _inputs(tmp_path)
    receipt_path = cross_source / "receipt.json"
    payload = json.loads(receipt_path.read_text())
    payload.pop("receipt_sha256")
    payload["pilot_bindings"][0]["receipt_sha256"] = "f" * 64
    _seal(receipt_path, payload)
    with pytest.raises(
        BoundedPilotCompilerPopulationError, match="pilot bindings differ"
    ):
        build_population(pilots, cross_source, tmp_path / "compiler")
