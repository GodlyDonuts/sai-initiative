from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.agent_candidate_population import (
    AgentCandidatePopulationError,
    build_population,
)
from sai.data.agent_labeling import normalize_candidate
from sai.data.token_stream import canonical_sha256, sha256_file


def _document(index: int) -> dict:
    text = (
        f"Lesson {index} introduces counting with concrete objects and a worked "
        "example. "
        "The learner first names each quantity, then combines the groups, and finally "
        "checks the result by counting every object exactly once. This explanation is "
        "written in plain English and does not require algebra or specialized notation."
    )
    row = {
        "schema": "sai-pretraining-document-v1",
        "text": text,
        "source": {
            "dataset": "example/foundation",
            "row_id": f"row-{index}",
            "license": "CC-BY-4.0",
            "domain": "english",
        },
        "verification": {
            "benchmark_disjoint": True,
            "evidence_sha256": hashlib.sha256(f"evidence-{index}".encode()).hexdigest(),
        },
    }
    row["identity_sha256"] = canonical_sha256(row)
    return row


def _write_source(path: Path, rows: int = 10) -> None:
    path.write_text(
        "".join(json.dumps(_document(index)) + "\n" for index in range(rows))
    )


def test_builds_stable_provenance_bound_population(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)
    output = tmp_path / "candidates.jsonl"
    receipt_path = tmp_path / "receipt.json"
    receipt = build_population(
        source,
        output,
        receipt_path,
        source_revision="example-revision-v1",
        source_type="textbook",
        sample_size=4,
        seed=20260822,
    )
    rows = [
        normalize_candidate(json.loads(line))
        for line in output.read_text().splitlines()
    ]
    assert len(rows) == 4
    assert len({row["candidate_identity_sha256"] for row in rows}) == 4
    assert receipt["source"]["rows"] == 10
    assert receipt["population"]["sha256"] == sha256_file(output)
    assert receipt["training_ready"] is False
    assert json.loads(receipt_path.read_text()) == receipt

    second = tmp_path / "second.jsonl"
    second_receipt = tmp_path / "second.receipt.json"
    repeated = build_population(
        source,
        second,
        second_receipt,
        source_revision="example-revision-v1",
        source_type="textbook",
        sample_size=4,
        seed=20260822,
    )
    assert second.read_bytes() == output.read_bytes()
    assert repeated["selection"] == receipt["selection"]


def test_rejects_duplicates_unsafe_output_and_invalid_geometry(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    row = _document(0)
    source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(AgentCandidatePopulationError, match="duplicated"):
        build_population(
            source,
            tmp_path / "out.jsonl",
            tmp_path / "receipt.json",
            source_revision="revision",
            source_type="textbook",
            sample_size=2,
            seed=1,
        )

    _write_source(source, rows=3)
    output = tmp_path / "existing.jsonl"
    output.write_text("occupied")
    with pytest.raises(AgentCandidatePopulationError, match="already exists"):
        build_population(
            source,
            output,
            tmp_path / "receipt.json",
            source_revision="revision",
            source_type="textbook",
            sample_size=1,
            seed=1,
        )
    with pytest.raises(AgentCandidatePopulationError, match="arguments differ"):
        build_population(
            source,
            tmp_path / "new.jsonl",
            tmp_path / "new.receipt.json",
            source_revision="revision",
            source_type="unknown",
            sample_size=1,
            seed=1,
        )
