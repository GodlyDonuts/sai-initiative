from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.attribution_manifest import build_manifest
from sai.data.decontamination import RAW_SCHEMA
from sai.data.source_lineage import SourceLineageError, source_row_id
from sai.data.structural_segmentation import build_segments, segment_text
from sai.data.token_stream import ROW_SCHEMA, canonical_sha256


def _raw(text: str, row_index: int) -> dict:
    return {
        "schema": RAW_SCHEMA,
        "text": text,
        "source": {
            "dataset": "example/long-forms",
            "revision": "a" * 40,
            "source_file": "part-000.parquet",
            "row_index": row_index,
            "license": "CC-BY-4.0",
            "declared_license": "CC-BY-4.0",
            "domain": "science",
        },
    }


def _retained(raw: dict) -> dict:
    source = raw["source"]
    payload = {
        "schema": ROW_SCHEMA,
        "text": raw["text"],
        "source": {
            "dataset": source["dataset"],
            "row_id": source_row_id(source, raw["text"]),
            "license": source["license"],
            "domain": source["domain"],
        },
        "verification": {"benchmark_disjoint": True, "evidence_sha256": "f" * 64},
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def test_segment_text_is_lossless_unicode_and_byte_bounded() -> None:
    text = "\n\n".join(
        f"Section {index}. Café λ explains a distinct scientific idea in detail."
        for index in range(12)
    )
    segments = segment_text(text, minimum_bytes=24, maximum_bytes=96)
    assert len(segments) > 1
    assert "".join(segment["text"] for segment in segments) == text
    assert all(len(segment["text"].encode()) <= 96 for segment in segments)
    assert any(segment["end_boundary"] == "paragraph" for segment in segments)
    assert [segment["utf8_start"] for segment in segments[1:]] == [
        segment["utf8_end"] for segment in segments[:-1]
    ]


def test_segmenter_seals_text_free_lineage_and_unique_child_ids(
    tmp_path: Path,
) -> None:
    long_text = "\n\n".join(
        f"Chapter {index}. This is grounded source material with exact wording."
        for index in range(18)
    )
    short_text = "A short but retained source document."
    source = tmp_path / "raw.jsonl"
    source.write_text(
        "\n".join(json.dumps(row) for row in (_raw(long_text, 3), _raw(short_text, 4)))
        + "\n"
    )
    output = tmp_path / "segmented.jsonl"
    lineage = tmp_path / "lineage.jsonl"
    receipt = tmp_path / "receipt.json"
    result = build_segments(
        source,
        output,
        lineage,
        receipt,
        minimum_bytes=24,
        maximum_bytes=128,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    segmented = [row for row in rows if "segment" in row["source"]]
    assert result["counts"]["input_documents"] == 2
    assert result["counts"]["segmented_documents"] == 1
    assert result["training_ready"] is False
    assert "grounded source material" not in lineage.read_text()
    assert "grounded source material" not in receipt.read_text()
    assert "".join(row["text"] for row in segmented) == long_text
    identities = [source_row_id(row["source"], row["text"]) for row in segmented]
    assert len(identities) == len(set(identities))
    assert all(
        row["source"]["segment"]["parent_text_sha256"]
        == hashlib.sha256(long_text.encode()).hexdigest()
        for row in segmented
    )
    assert rows[-1] == _raw(short_text, 4)


def test_segment_lineage_survives_attribution_replay(tmp_path: Path) -> None:
    text = "\n\n".join(
        f"Part {index}. A sufficiently detailed factual paragraph for replay."
        for index in range(10)
    )
    source = tmp_path / "raw.jsonl"
    source.write_text(json.dumps(_raw(text, 8)) + "\n")
    segmented_path = tmp_path / "segmented.jsonl"
    build_segments(
        source,
        segmented_path,
        tmp_path / "lineage.jsonl",
        tmp_path / "segment-receipt.json",
        minimum_bytes=24,
        maximum_bytes=112,
    )
    segmented = [json.loads(line) for line in segmented_path.read_text().splitlines()]
    retained_path = tmp_path / "retained.jsonl"
    retained_path.write_text(
        "\n".join(json.dumps(_retained(row)) for row in segmented) + "\n"
    )
    manifest = tmp_path / "attribution.jsonl"
    result = build_manifest(
        segmented_path,
        retained_path,
        manifest,
        tmp_path / "attribution-receipt.json",
    )
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert result["output"]["records"] == len(segmented)
    assert [record["source"]["segment"]["index"] for record in records] == list(
        range(len(segmented))
    )
    assert all("text" not in record for record in records)


def test_tampered_segment_text_fails_lineage_validation() -> None:
    raw = _raw("A" * 300, 12)
    parent = canonical_sha256(
        {
            "dataset": raw["source"]["dataset"],
            "revision": raw["source"]["revision"],
            "source_file": raw["source"]["source_file"],
            "row_index": raw["source"]["row_index"],
        }
    )
    raw["source"]["segment"] = {
        "schema": "sai-structural-segment-lineage-v1",
        "parent_row_id": parent,
        "parent_text_sha256": hashlib.sha256(("A" * 600).encode()).hexdigest(),
        "index": 0,
        "count": 2,
        "utf8_start": 0,
        "utf8_end": 300,
        "segment_text_sha256": "0" * 64,
    }
    with pytest.raises(SourceLineageError, match="structural segment"):
        source_row_id(raw["source"], raw["text"])
