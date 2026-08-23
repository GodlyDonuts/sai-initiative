from __future__ import annotations

import json
from pathlib import Path

from sai.data.cross_source_pilot_duplicates import build_sample
from sai.data.token_stream import ROW_SCHEMA, canonical_sha256, sha256_file


def _document(text: str, source_id: str, row_id: str) -> dict:
    payload = {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": source_id,
            "row_id": row_id,
            "license": "CC0-1.0",
            "domain": "english",
        },
        "verification": {"benchmark_disjoint": True, "evidence_sha256": "f" * 64},
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def _pilot(root: Path, source_id: str, documents: list[dict]) -> Path:
    root.mkdir()
    output = root / "bounded.jsonl"
    output.write_text("".join(json.dumps(row) + "\n" for row in documents))
    receipt = {
        "schema": "sai-common-pile-streaming-pilot-v1",
        "source_id": source_id,
        "near_duplicate_filter": {
            "output_path": output.name,
            "output_bytes": output.stat().st_size,
            "output_sha256": sha256_file(output),
            "output_documents": len(documents),
        },
        "bounded_pilot_near_duplicate_filter_complete": True,
        "global_cross_source_near_duplicate_filter_complete": False,
        "training_ready": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    (root / "receipt.json").write_text(json.dumps(receipt))
    return root


def test_cross_source_sample_finds_duplicate_family(tmp_path: Path) -> None:
    base = " ".join(f"token{index}" for index in range(120))
    near = " ".join(
        f"changed{index}" if index < 5 else f"token{index}" for index in range(120)
    )
    unique = " ".join(f"unique{index}" for index in range(120))
    left = _pilot(
        tmp_path / "left",
        "common_pile_left",
        [_document(base, "left/repository", "left-0")],
    )
    right = _pilot(
        tmp_path / "right",
        "common_pile_right",
        [
            _document(near, "right/repository", "right-0"),
            _document(unique, "right/repository", "right-1"),
        ],
    )
    result = build_sample([right, left], tmp_path / "result", maximum_rows=4)
    assert result["selection"]["input_documents"] == 3
    assert result["duplicate_filter"]["duplicate_groups"] == 1
    assert result["duplicate_filter"]["cross_source_duplicate_groups"] == 1
    assert result["duplicate_filter"]["documents_dropped"] == 1
    assert result[
        "full_pilot_population_cross_source_deduplication_complete"
    ] is True
    assert result["full_reservoir_cross_source_deduplication_complete"] is False
    assert result["training_ready"] is False
