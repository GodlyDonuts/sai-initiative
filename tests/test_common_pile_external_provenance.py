from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from sai.data.attribution_manifest import build_manifest as build_attribution
from sai.data.common_pile_external_provenance import (
    CommonPileExternalProvenanceError,
    build_manifest,
)
from sai.data.common_pile_streaming_pilot import SCHEMA as PILOT_SCHEMA
from sai.data.decontamination import RAW_SCHEMA
from sai.data.token_stream import ROW_SCHEMA, canonical_sha256, sha256_file


def _seal(path: Path, payload: dict) -> dict:
    payload = dict(payload)
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))
    return payload


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    text = (
        "A carefully attributed educational chapter explains a foundational idea "
        "with enough context for a robust provenance replay. " * 4
    )
    parent_row = {
        "id": "https://example.org/books/chapter-1/",
        "source": "pressbooks",
        "created": "2025-01-01",
        "metadata": {
            "url": "https://example.org/books/chapter-1/",
            "book_url": "https://example.org/books/",
            "license": "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/",
            "title": "A Grounded Book",
            "author": "A. Author",
            "provenance": "pressbooks",
        },
        "text": text,
    }
    parent = tmp_path / "pressbooks.json.gz"
    with gzip.open(parent, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(parent_row) + "\n")
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    raw_row = {
        "schema": RAW_SCHEMA,
        "text": text,
        "source": {
            "dataset": "common-pile/pressbooks_filtered",
            "revision": "a" * 40,
            "source_file": parent.name,
            "row_index": 0,
            "license": "CC-BY-4.0",
            "declared_license": parent_row["metadata"]["license"],
            "domain": "english",
        },
    }
    raw = pilot / "raw.jsonl"
    raw.write_text(json.dumps(raw_row) + "\n")
    row_id = canonical_sha256(
        {
            "dataset": raw_row["source"]["dataset"],
            "revision": raw_row["source"]["revision"],
            "source_file": raw_row["source"]["source_file"],
            "row_index": 0,
        }
    )
    retained_row = {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": raw_row["source"]["dataset"],
            "row_id": row_id,
            "license": "CC-BY-4.0",
            "domain": "english",
        },
        "verification": {
            "benchmark_disjoint": True,
            "evidence_sha256": "f" * 64,
        },
    }
    retained_row["identity_sha256"] = canonical_sha256(retained_row)
    retained = pilot / "retained.jsonl"
    retained.write_text(json.dumps(retained_row) + "\n")
    attribution = pilot / "attribution.jsonl"
    attribution_receipt = pilot / "attribution_receipt.json"
    attribution_result = build_attribution(
        raw, retained, attribution, attribution_receipt
    )
    _seal(
        pilot / "receipt.json",
        {
            "schema": PILOT_SCHEMA,
            "status": "complete_nontraining_pilot",
            "source_id": "common_pile_pressbooks",
            "parent": {
                "source_id": "common_pile_pressbooks",
                "repository": raw_row["source"]["dataset"],
                "revision": raw_row["source"]["revision"],
                "path": parent.name,
                "bytes": parent.stat().st_size,
                "sha256": sha256_file(parent),
                "manifest_license": (
                    "common_pile_source_specific_public_domain_or_open_license"
                ),
            },
            "raw_population": {
                "path": raw.name,
                "rows": 1,
                "bytes": raw.stat().st_size,
                "sha256": sha256_file(raw),
            },
            "attribution_manifest": {
                "output_path": attribution.name,
                "output_bytes": attribution.stat().st_size,
                "output_sha256": sha256_file(attribution),
                "records": attribution_result["output"]["records"],
            },
            "training_ready": False,
        },
    )
    return pilot, parent


def test_replays_text_free_source_metadata_and_urls(tmp_path: Path) -> None:
    pilot, parent = _fixture(tmp_path)
    result = build_manifest(
        pilot,
        tmp_path / "provenance",
        token="test-token",
        download_function=lambda _parent, _token, _temp: parent,
    )
    record = json.loads(
        (tmp_path / "provenance" / "external_provenance_manifest.jsonl").read_text()
    )
    assert result["output"]["rows"] == 1
    assert record["source_urls"] == [
        "https://example.org/books/",
        "https://example.org/books/chapter-1/",
    ]
    assert record["source_metadata"]["metadata.title"] == "A Grounded Book"
    assert "text" not in json.dumps(record)
    assert result["source_text_persisted"] is False
    assert result["rights_provenance_verified"] is False


def test_rejects_parent_hash_mismatch(tmp_path: Path) -> None:
    pilot, parent = _fixture(tmp_path)
    receipt_path = pilot / "receipt.json"
    payload = json.loads(receipt_path.read_text())
    payload.pop("receipt_sha256")
    payload["parent"]["sha256"] = "0" * 64
    _seal(receipt_path, payload)
    with pytest.raises(CommonPileExternalProvenanceError, match="parent differs"):
        build_manifest(
            pilot,
            tmp_path / "provenance",
            token="test-token",
            download_function=lambda _parent, _token, _temp: parent,
        )
