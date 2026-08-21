from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.split import DocumentSplitError, split
from sai.data.token_stream import ROW_SCHEMA


def row(index: int) -> dict:
    return {
        "schema": ROW_SCHEMA,
        "text": f"document-{index}",
        "source": {
            "dataset": "split-test",
            "row_id": str(index),
            "license": "CC0-1.0",
            "domain": "english",
        },
        "verification": {
            "benchmark_disjoint": True,
            "evidence_sha256": hashlib.sha256(f"evidence-{index}".encode()).hexdigest(),
        },
    }


def test_split_is_deterministic_disjoint_and_explicitly_mechanics_only(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row(index)) + "\n" for index in range(1_000)))
    reports = []
    for prefix in ("first", "second"):
        reports.append(
            split(
                source,
                tmp_path / f"{prefix}.train",
                tmp_path / f"{prefix}.dev",
                tmp_path / f"{prefix}.receipt",
            )
        )
    assert reports[0]["train"]["sha256"] == reports[1]["train"]["sha256"]
    assert reports[0]["development"]["sha256"] == reports[1]["development"]["sha256"]
    assert (
        reports[0]["train"]["documents"] + reports[0]["development"]["documents"]
        == 1_000
    )
    assert reports[0]["near_duplicate_cluster_split_qualified"] is False
    assert reports[0]["scientific_promotion_allowed"] is False


def test_split_rejects_existing_output_or_empty_population(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row(0)) + "\n")
    with pytest.raises(DocumentSplitError, match="empty population"):
        split(source, tmp_path / "train", tmp_path / "dev", tmp_path / "receipt")
