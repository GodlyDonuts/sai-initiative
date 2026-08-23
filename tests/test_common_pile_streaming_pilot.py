from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from sai.data.common_pile_streaming_pilot import (
    select_bottom_k,
    select_parent,
    write_raw_population,
)
from sai.data.decontamination import RAW_SCHEMA
from sai.data.token_stream import sha256_file


def _manifest_row(path: str, size: int) -> dict:
    return {
        "source_id": "common_pile_stackexchange",
        "repository": "common-pile/stackexchange_filtered",
        "revision": "a" * 40,
        "path": path,
        "physical_bytes": size,
        "sha256": "1" * 64,
        "license": "CC-BY-SA-4.0",
    }


def test_parent_selection_prefers_smallest_audit_disjoint_parent() -> None:
    rows = [_manifest_row("old.json.gz", 10), _manifest_row("new.json.gz", 20)]
    selected = select_parent(
        rows,
        "common_pile_stackexchange",
        {("common-pile/stackexchange_filtered", "old.json.gz")},
    )
    assert selected["path"] == "new.json.gz"
    assert selected["parent_disjoint_from_audits"] is True


def test_bottom_k_excludes_audits_and_replays_raw_rows(tmp_path: Path) -> None:
    compressed = tmp_path / "source.json.gz"
    rows = [
        {
            "id": f"row-{index}",
            "text": (f"Document {index} carries independently useful content. " * 8),
            "metadata": {"license": "CC-BY-SA-4.0"},
        }
        for index in range(8)
    ]
    with gzip.open(compressed, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    parent = {
        "source_id": "common_pile_stackexchange",
        "repository": "common-pile/stackexchange_filtered",
        "revision": "a" * 40,
        "path": compressed.name,
        "bytes": compressed.stat().st_size,
        "sha256": sha256_file(compressed),
        "manifest_license": "CC-BY-SA-4.0",
        "domain": "technical",
    }
    excluded_hash = hashlib.sha256(rows[1]["text"].strip().encode()).hexdigest()
    selected, counters = select_bottom_k(
        compressed,
        parent,
        maximum_rows=4,
        excluded_lines=frozenset({1}),
        excluded_content_sha256s=frozenset({excluded_hash}),
    )
    assert len(selected) == 4
    assert counters["audit_excluded_rows"] == 2
    output = tmp_path / "raw.jsonl"
    receipt = write_raw_population(compressed, parent, selected, output)
    written = [json.loads(line) for line in output.read_text().splitlines()]
    assert receipt["rows"] == 4
    assert all(row["schema"] == RAW_SCHEMA for row in written)
    assert all(row["source"]["domain"] == "technical" for row in written)
    assert all(
        row["source"]["license"] == "CC-BY-SA-4.0"
        and row["source"]["declared_license"] == "CC-BY-SA-4.0"
        for row in written
    )
