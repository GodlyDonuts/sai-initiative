from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sai.data.curriculum_control import _record_sha256, _Records
from sai.data.learnability_curriculum import (
    BANDS,
    ORDER_SEED,
    PHASES,
    POLICY_SCHEMA,
    SCORE_SCHEMA,
    LearnabilityCurriculumError,
    build_learnability_curriculum,
    validate_learnability_curriculum,
)
from sai.data.token_stream import (
    canonical_sha256,
    causal_loss_mask_from_start_bits,
    freeze,
    sha256_file,
)
from tests.test_token_stream import CharacterTokenizer, document, write_documents


def _parent(tmp_path: Path) -> tuple[Path, dict]:
    source = write_documents(
        tmp_path / "source.jsonl",
        [document(0, "abcdefghijklmnopqrstuvwx" * 3)],
    )
    root = tmp_path / "parent"
    report = freeze(
        CharacterTokenizer(),
        [source],
        root,
        tokenizer_identity_sha256="1" * 64,
        sequence_length=4,
        prefix_sequences={16},
        sequences_per_shard=3,
        source_qualification_sha256="2" * 64,
    )
    assert report["sequences"] == 16
    return root, report


def _policy(path: Path, source: Path, report: dict) -> dict:
    payload = {
        "schema": POLICY_SCHEMA,
        "status": "prospective",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "source_stream_identity_sha256": report["ordered_stream_identity_sha256"],
        "source_receipt_file_sha256": sha256_file(source / "stream_receipt.json"),
        "sequence_count": 16,
        "sequences_per_update": 2,
        "phase_order": list(PHASES),
        "band_order": list(BANDS),
        "phase_sequence_counts": {
            "grounding": 2,
            "integration": 4,
            "reasoning": 4,
            "specialization": 6,
        },
        "band_sequence_counts": {
            "ready": 5,
            "developing": 4,
            "challenging": 4,
            "stretch": 3,
        },
        "phase_by_band_counts": {
            "grounding": {
                "ready": 2,
                "developing": 0,
                "challenging": 0,
                "stretch": 0,
            },
            "integration": {
                "ready": 1,
                "developing": 2,
                "challenging": 1,
                "stretch": 0,
            },
            "reasoning": {
                "ready": 1,
                "developing": 1,
                "challenging": 1,
                "stretch": 1,
            },
            "specialization": {
                "ready": 1,
                "developing": 1,
                "challenging": 2,
                "stretch": 2,
            },
        },
        "scoring": {
            "method": "weak_minus_strong_normalized_nll_microunits",
            "weak_checkpoint_sha256": "3" * 64,
            "strong_checkpoint_sha256": "4" * 64,
            "tokenizer_sha256": "5" * 64,
            "evaluator_sha256": "6" * 64,
            "runtime_sha256": "7" * 64,
            "treatment_checkpoint_used": False,
            "terminal_benchmark_feedback_used": False,
        },
        "within_phase_order": {
            "method": "sha256_ranked_without_score_order_within_phase",
            "seed": ORDER_SEED,
        },
        "controls": {
            "same_sequence_multiset": True,
            "same_source_bytes": True,
            "tokenizer_factor_isolated": True,
            "architecture_factor_isolated": True,
            "only_score_to_order_changed": True,
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def _scores(path: Path, source: Path, report: dict) -> list[dict]:
    rows = []
    with _Records(source, report) as records:
        for index in range(report["sequences"]):
            tokens, starts = records.record(index)
            row = {
                "schema": SCORE_SCHEMA,
                "sequence_index": index,
                "record_sha256": _record_sha256(tokens, starts).hex(),
                "target_count": sum(
                    causal_loss_mask_from_start_bits(starts, report["sequence_length"])
                ),
                "weak_nll_microunits_per_target": 1_000_000 + 10_000 * index,
                "strong_nll_microunits_per_target": 900_000,
                "preference_delta_microunits": 100_000 + 10_000 * index,
            }
            rows.append(row)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )
    return rows


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    source, report = _parent(tmp_path)
    policy = tmp_path / "policy.json"
    scores = tmp_path / "scores.jsonl"
    _policy(policy, source, report)
    _scores(scores, source, report)
    return source, scores, policy, report


def test_builds_and_replays_exact_record_model_centric_curriculum(
    tmp_path: Path,
) -> None:
    source, scores, policy, parent = _fixture(tmp_path)
    output = tmp_path / "learnability"
    report = build_learnability_curriculum(source, scores, policy, output)

    schedule = report["learnability_curriculum"]
    assert report["sequences"] == parent["sequences"]
    assert (
        report["ordered_stream_identity_sha256"]
        != parent["ordered_stream_identity_sha256"]
    )
    assert report["prefix_utf8_bytes"] == {"16": parent["admitted_utf8_bytes"]}
    assert schedule["same_sequence_multiset"] is True
    assert schedule["only_sequence_order_changed"] is True
    assert schedule["semantic_prerequisite_order_proven"] is False
    assert {phase: schedule["phases"][phase]["sequences"] for phase in PHASES} == {
        "grounding": 2,
        "integration": 4,
        "reasoning": 4,
        "specialization": 6,
    }
    assert (
        validate_learnability_curriculum(
            output, source=source, scores_path=scores, policy_path=policy
        )
        == report
    )


def test_rejects_posthoc_policy_or_score_tamper(tmp_path: Path) -> None:
    source, scores, policy, _ = _fixture(tmp_path)
    output = tmp_path / "learnability"
    build_learnability_curriculum(source, scores, policy, output)

    policy_payload = json.loads(policy.read_text())
    policy_payload["phase_by_band_counts"]["grounding"]["stretch"] = 1
    policy_payload["phase_by_band_counts"]["grounding"]["ready"] = 1
    policy_payload["phase_by_band_counts"]["specialization"]["stretch"] = 1
    policy_payload["phase_by_band_counts"]["specialization"]["ready"] = 2
    policy_payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in policy_payload.items() if key != "receipt_sha256"}
    )
    policy.write_text(json.dumps(policy_payload) + "\n")
    with pytest.raises(LearnabilityCurriculumError, match="rehearsal boundary"):
        validate_learnability_curriculum(
            output, source=source, scores_path=scores, policy_path=policy
        )

    _policy(policy, source, json.loads((source / "stream_receipt.json").read_text()))
    score_rows = [json.loads(line) for line in scores.read_text().splitlines()]
    score_rows[0]["record_sha256"] = "f" * 64
    scores.write_text("".join(json.dumps(row) + "\n" for row in score_rows))
    with pytest.raises(LearnabilityCurriculumError, match="score record"):
        validate_learnability_curriculum(
            output, source=source, scores_path=scores, policy_path=policy
        )


def test_rejects_treatment_feedback_and_nonprogressive_schedule(tmp_path: Path) -> None:
    source, _, policy, _ = _fixture(tmp_path)
    payload = json.loads(policy.read_text())
    payload["scoring"]["treatment_checkpoint_used"] = True
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    policy.write_text(json.dumps(payload) + "\n")
    from sai.data.learnability_curriculum import load_policy

    with pytest.raises(LearnabilityCurriculumError, match="scoring contract"):
        load_policy(
            policy, source, json.loads((source / "stream_receipt.json").read_text())
        )

    payload["scoring"]["treatment_checkpoint_used"] = False
    payload["phase_by_band_counts"] = deepcopy(
        {
            "grounding": {"ready": 2, "developing": 0, "challenging": 0, "stretch": 0},
            "integration": {
                "ready": 1,
                "developing": 2,
                "challenging": 1,
                "stretch": 0,
            },
            "reasoning": {"ready": 1, "developing": 1, "challenging": 1, "stretch": 1},
            "specialization": {
                "ready": 1,
                "developing": 1,
                "challenging": 2,
                "stretch": 2,
            },
        }
    )
    (
        payload["phase_by_band_counts"]["reasoning"],
        payload["phase_by_band_counts"]["specialization"],
    ) = (
        payload["phase_by_band_counts"]["specialization"],
        payload["phase_by_band_counts"]["reasoning"],
    )
    payload["phase_sequence_counts"]["reasoning"] = 6
    payload["phase_sequence_counts"]["specialization"] = 4
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    policy.write_text(json.dumps(payload) + "\n")
    with pytest.raises(LearnabilityCurriculumError, match="phase boundary|difficulty"):
        load_policy(
            policy, source, json.loads((source / "stream_receipt.json").read_text())
        )


def test_create_only_and_output_tamper_fail(tmp_path: Path) -> None:
    source, scores, policy, _ = _fixture(tmp_path)
    output = tmp_path / "learnability"
    build_learnability_curriculum(source, scores, policy, output)
    with pytest.raises(LearnabilityCurriculumError, match="already exists"):
        build_learnability_curriculum(source, scores, policy, output)
    token_path = next(output.glob("*.tokens.u32le"))
    payload = bytearray(token_path.read_bytes())
    payload[0] ^= 1
    token_path.write_bytes(payload)
    with pytest.raises(Exception, match="content differs|multiset differs"):
        validate_learnability_curriculum(
            output, source=source, scores_path=scores, policy_path=policy
        )
