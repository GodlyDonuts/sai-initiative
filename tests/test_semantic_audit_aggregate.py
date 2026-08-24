import hashlib
import json

import pytest

from sai.data.agent_candidate_population import SCHEMA as POPULATION_SCHEMA
from sai.data.agent_labeling import (
    PERSPECTIVES,
    RUBRIC_SHA256,
    normalize_model_judgment,
)
from sai.data.nous_label_worker import SCHEMA as JUDGMENT_SCHEMA
from sai.data.semantic_audit_aggregate import (
    SCHEMA,
    SemanticAuditAggregateError,
    build_aggregate,
)
from sai.data.token_stream import canonical_sha256, sha256_file


def _candidate() -> dict:
    text = "A grounded worked example explains addition with counted objects. " * 5
    row = {
        "schema": "sai-agent-data-candidate-v1",
        "text": text,
        "source": {
            "dataset": "example/math",
            "revision": "v1",
            "row_id": "row-1",
            "license": "CC-BY-4.0",
            "source_type": "educational_web",
        },
        "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "provenance_sha256": "a" * 64,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def _raw() -> dict:
    return {
        "verdict": "retain",
        "quality_score": 4,
        "english_score": 4,
        "domains": ["math"],
        "difficulty": 0,
        "prerequisite_burden": 0,
        "curriculum_phase": "grounding",
        "pedagogical_role": "worked_example",
        "concepts_taught": ["addition"],
        "prerequisites_assumed": ["counting"],
        "risks": {
            "non_english_general_text": False,
            "seo_or_content_farm": False,
            "incoherent_or_corrupted": False,
            "factual_unreliability": False,
            "duplicated_boilerplate": False,
            "answer_farm_without_teaching": False,
            "personal_or_secret_data": False,
        },
        "confidence_ppm": 950_000,
        "evidence_quotes": ["worked example explains addition"],
        "rationale": "The document teaches a correct prerequisite-light concept.",
    }


def _signed(path, payload):
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload))


def _workspace(tmp_path):
    candidate = _candidate()
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(candidate) + "\n")
    population = tmp_path / "population.json"
    _signed(
        population,
        {
            "schema": POPULATION_SCHEMA,
            "status": "complete",
            "population": {
                "rows": 1,
                "bytes": candidates.stat().st_size,
                "sha256": sha256_file(candidates),
            },
            "training_ready": False,
        },
    )
    judgments = tmp_path / "judgments"
    judgments.mkdir()
    for slot, perspective in enumerate(PERSPECTIVES):
        judgment = normalize_model_judgment(_raw(), candidate, slot)
        _signed(
            judgments / f"{candidate['candidate_identity_sha256']}.slot{slot}.json",
            {
                "schema": JUDGMENT_SCHEMA,
                "status": "complete",
                "candidate_identity_sha256": candidate["candidate_identity_sha256"],
                "annotator_slot": slot,
                "perspective": perspective,
                "rubric_sha256": RUBRIC_SHA256,
                "requested_model": "stealth/ox-alpha",
                "attempts": [{"outcome": "valid"}],
                "judgment": judgment,
                "api_key_persisted": False,
                "tools_enabled": False,
                "training_ready": False,
            },
        )
    _signed(
        judgments / "shard_00000.summary.json",
        {
            "schema": "sai-nous-agent-label-shard-summary-v1",
            "status": "complete",
            "model": "stealth/ox-alpha",
            "rubric_sha256": RUBRIC_SHA256,
            "logical_shards": 1,
            "shard_index": 0,
            "candidate_rows": 1,
            "judgments_per_candidate": 3,
            "expected_judgments": 3,
            "created_judgments": 3,
            "preexisting_judgments": 0,
            "api_key_persisted": False,
            "training_ready": False,
        },
    )
    return candidates, population, judgments, candidate


def test_semantic_aggregate_requires_every_perspective_and_strips_text(tmp_path):
    candidates, population, judgments, candidate = _workspace(tmp_path)
    output = tmp_path / "aggregate"
    result = build_aggregate(
        candidates,
        population,
        judgments,
        output,
        expected_model="stealth/ox-alpha",
        logical_shards=1,
    )
    assert result["schema"] == SCHEMA
    assert result["counts"]["judgments"] == 3
    assert result["counts"]["disposition::retain"] == 1
    assert result["complete_three_perspective_coverage"] is True
    row = json.loads((output / "decisions.jsonl").read_text())
    assert "text" not in row
    assert row["candidate_identity_sha256"] == candidate["candidate_identity_sha256"]
    assert row["aggregate"]["disposition"] == "retain"


def test_semantic_aggregate_rejects_missing_slot(tmp_path):
    candidates, population, judgments, candidate = _workspace(tmp_path)
    (judgments / f"{candidate['candidate_identity_sha256']}.slot2.json").unlink()
    with pytest.raises(SemanticAuditAggregateError, match="signed receipt"):
        build_aggregate(
            candidates,
            population,
            judgments,
            tmp_path / "aggregate",
            expected_model="stealth/ox-alpha",
            logical_shards=1,
        )
