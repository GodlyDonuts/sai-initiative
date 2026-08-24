import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from sai.data import pleias_production_descriptor_census as census
from sai.data.pleias_production_descriptor_census import (
    aggregate,
    bottom_k_word_shingles,
    descriptor,
    normalized_text,
    run_shard,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _signed(value):
    value["receipt_sha256"] = canonical_sha256(value)
    return value


def _policy(path: Path):
    row = {
        "collection": "Books",
        "language": "English",
        "work_route": "priority_direct_representation_verification",
        "automatic_training_admission": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    value = _signed(
        {
            "schema": "sai-pleias-quality-core-policy-v1",
            "status": "complete_nontraining_pleias_quality_core_work_policy",
            "groups": [row],
            "ordered_group_rows_sha256": canonical_sha256([row["row_sha256"]]),
            "automatic_training_admission": False,
            "training_ready": False,
        }
    )
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return value


def _decision(path: Path, stratum: str):
    row = {
        "stratum": stratum,
        "decision": "advance_to_full_candidate_decontamination",
        "automatic_training_admission": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    value = _signed(
        {
            "schema": "sai-pleias-semantic-stratum-decision-v1",
            "status": "complete_nontraining_pleias_semantic_stratum_decision",
            "decisions": [row],
            "ordered_decisions_sha256": canonical_sha256([row["row_sha256"]]),
            "advanced_strata": [stratum],
            "training_ready": False,
        }
    )
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return value


def _row(identifier: str, text: str):
    return {
        "identifier": identifier,
        "collection": "Books",
        "open_type": "Open Culture",
        "license": "Public Domain",
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
    return source, manifest, manifest_row


def test_normalization_and_near_signature_ignore_surface_variation():
    first = "Alpha  beta\nGAMMA delta epsilon zeta eta theta"
    second = "alpha beta gamma delta epsilon zeta eta theta"
    assert normalized_text(first) == normalized_text(second)
    assert bottom_k_word_shingles(first) == bottom_k_word_shingles(second)
    assert len(bottom_k_word_shingles(first, size=3)) == 3


def test_descriptor_contains_no_text_and_hashes_identity():
    text = "A coherent account of astronomy and observation. " * 30
    row = _row("one", text)
    parent = {"source_path": "data/one.parquet", "sha256": "b" * 64}
    value = descriptor(row, parent, 7)
    assert "text" not in value
    assert value["text_utf8_bytes"] == len(text.encode())
    assert value["content_sha256"] != value["normalized_content_sha256"]
    unsigned = {key: item for key, item in value.items() if key != "descriptor_sha256"}
    assert value["descriptor_sha256"] == canonical_sha256(unsigned)
    assert value["training_ready"] is False


def test_full_parent_census_and_aggregate_are_nontraining(tmp_path, monkeypatch):
    prose = " ".join(
        f"Chapter {index} explains astronomy observation measurement telescope "
        "orbit spectrum evidence and inference."
        for index in range(30)
    )
    answer_key = "\n".join(f"{index}. A" for index in range(1, 200))
    source, manifest, _parent = _source(
        tmp_path, [_row("good", prose), _row("bad", answer_key)]
    )
    policy_path = tmp_path / "policy.json"
    _policy(policy_path)
    stratum = "Books::Open Culture::512to4095"
    decision_path = tmp_path / "decision.json"
    _decision(decision_path, stratum)
    monkeypatch.setattr(census, "_download", lambda *_args: source)
    shard_root = tmp_path / "census" / "shard_00000"
    result = run_shard(
        manifest,
        policy_path,
        decision_path,
        shard_root,
        1,
        0,
        "token",
        tmp_path,
    )
    assert result["counts"]["source_rows"] == 2
    assert result["counts"]["production_candidate_descriptors"] == 1
    assert result["source_text_persisted"] is False
    rows = pq.read_table(shard_root / "candidate_descriptors.parquet").to_pylist()
    assert len(rows) == 1
    assert "text" not in rows[0]
    output = tmp_path / "aggregate.json"
    combined = aggregate(
        manifest,
        policy_path,
        decision_path,
        tmp_path / "census",
        1,
        output,
    )
    assert combined["complete_source_parent_coverage"] is True
    assert combined["totals"]["production_candidate_descriptors"] == 1
    assert combined["global_near_deduplication_complete"] is False
    assert combined["training_ready"] is False
