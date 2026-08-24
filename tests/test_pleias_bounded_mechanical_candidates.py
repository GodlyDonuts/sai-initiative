import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sai.data import pleias_bounded_mechanical_candidates as candidates
from sai.data.pleias_bounded_mechanical_candidates import (
    PleiasBoundedMechanicalCandidatesError,
    aggregate,
    evaluate_row,
    license_allowed,
    run_shard,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(payload):
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _policy(path: Path):
    groups = []
    for collection, language, route in (
        (
            "Books",
            "English",
            "priority_direct_representation_verification",
        ),
        ("Cleanup", "English", "priority_cleanup_then_verification"),
        ("FrenchBooks", "French", "translation_value_review"),
    ):
        row = {
            "collection": collection,
            "language": language,
            "work_route": route,
            "automatic_training_admission": False,
        }
        row["row_sha256"] = canonical_sha256(row)
        groups.append(row)
    payload = _signed(
        {
            "schema": "sai-pleias-quality-core-policy-v1",
            "status": "complete_nontraining_pleias_quality_core_work_policy",
            "groups": groups,
            "ordered_group_rows_sha256": canonical_sha256(
                [row["row_sha256"] for row in groups]
            ),
            "automatic_training_admission": False,
            "training_ready": False,
        }
    )
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def _row(identifier, text, license_name="Public Domain", collection="Books"):
    return {
        "identifier": identifier,
        "collection": collection,
        "open_type": "Open Culture",
        "license": license_name,
        "language": "English",
        "word_count": max(64, len(text.split())),
        "token_count": max(96, len(text.split()) * 2),
        "text": text,
    }


def _source(tmp_path: Path, rows):
    source = tmp_path / "parent.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source, compression="zstd")
    manifest_row = {
        "source_id": "pleias_common_corpus",
        "source_path": "data/parent.parquet",
        "source_repository": "PleIAs/common_corpus",
        "source_revision": "a" * 40,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "raw_source_is_training_ready": False,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(manifest_row, sort_keys=True) + "\n")
    return source, manifest


def test_license_allowlist_is_fail_closed():
    for value in (
        "Public Domain",
        "CC0",
        "CC-By",
        "CC BY 4.0",
        "CC-BY-SA-4.0",
        "MIT",
        "Apache-2.0",
    ):
        assert license_allowed(value)
    for value in (
        "CC-BY-NC",
        "CC-BY-ND",
        "Various open science",
        "LicenseRef-unknown, MIT",
        "",
        None,
    ):
        assert not license_allowed(value)


def test_row_gate_requires_direct_group_rights_and_real_context(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy = _policy(policy_path)
    routes = candidates._routes(policy)
    prose = "A coherent discussion of astronomy and observation. " * 30
    assert evaluate_row(_row("ok", prose), routes)[0] == "pass_mechanical_gate"
    assert (
        evaluate_row(_row("rights", prose, "Various open science"), routes)[0]
        == "hold_rights"
    )
    assert (
        evaluate_row(_row("cleanup", prose, collection="Cleanup"), routes)[0]
        == "hold_group_route"
    )
    answer_key = "\n".join(f"{number}. A" for number in range(1, 200))
    row = _row("answer-key", answer_key)
    row["word_count"] = 160
    assert evaluate_row(row, routes)[0] == "hold_hard_reject"


def test_builds_bounded_nontraining_shard_and_aggregate(tmp_path, monkeypatch):
    prose = "A coherent discussion of astronomy and observation. " * 30
    other = "A careful explanation of ecological systems and feedback. " * 30
    source, manifest = _source(
        tmp_path,
        [
            _row("one", prose),
            _row("two", other),
            _row("unknown-rights", prose, "Various open science"),
        ],
    )
    policy_path = tmp_path / "policy.json"
    _policy(policy_path)
    monkeypatch.setattr(candidates, "_download", lambda *_args: source)
    output = tmp_path / "output" / "shard_00000"
    cap = len(prose.encode()) + 8
    result = run_shard(
        manifest,
        policy_path,
        output,
        1,
        0,
        "token",
        1_000_000,
        cap,
        tmp_path,
    )
    assert result["counts"]["source_rows"] == 3
    assert result["counts"]["selected_candidates"] == 1
    assert result["counts"]["pass_over_shard_byte_cap"] == 1
    assert result["counts"]["hold_rights"] == 1
    assert result["selected"]["text_utf8_bytes"] <= cap
    assert result["training_ready"] is False
    row = pq.read_table(output / "candidates.parquet").to_pylist()[0]
    assert row["identifier"] == "one"
    assert row["training_ready"] is False
    aggregate_path = tmp_path / "aggregate.json"
    combined = aggregate(
        manifest,
        policy_path,
        tmp_path / "output",
        1,
        cap,
        aggregate_path,
    )
    assert combined["complete_source_parent_coverage"] is True
    assert combined["totals"]["selected_rows"] == 1
    assert combined["semantic_admission_complete"] is False
    assert combined["training_ready"] is False
    changed = json.loads(policy_path.read_text())
    changed["groups"][0]["work_route"] = "hold_high_blocking_signal"
    changed["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "receipt_sha256"}
    )
    policy_path.write_text(json.dumps(changed, sort_keys=True) + "\n")
    with pytest.raises(PleiasBoundedMechanicalCandidatesError, match="group route"):
        candidates._routes(changed)
