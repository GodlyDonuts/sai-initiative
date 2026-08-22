from __future__ import annotations

import json
from pathlib import Path

import pytest

import sai.evaluation.curriculum_order_compare as comparison
from sai.data.token_stream import canonical_sha256
from sai.evaluation.curriculum_order_compare import (
    CurriculumOrderComparisonError,
    compare_curriculum_order,
    write_comparison,
)

CURRICULUM_ID = "1" * 64
CONTROL_ID = "2" * 64
DEVELOPMENT_ID = "3" * 64
SPLIT_ID = "4" * 64
TOKENIZER_ID = "5" * 64


def _stream(identity: str, *, sequences: int, source: str) -> dict:
    payload = {
        "source_qualification_sha256": None,
        "tokenizer_identity_sha256": TOKENIZER_ID,
        "ordered_stream_identity_sha256": identity,
        "sequence_length": 2_048,
        "sequences": sequences,
        "admitted_utf8_bytes": 10_000 if sequences == 244_140 else 2_000,
        "source_receipts": [{"path": source}],
    }
    if sequences == 1_024:
        phases = ("grounding", "integration", "reasoning", "specialization")
        payload["curriculum"] = {
            "phase_order": list(phases),
            "phase_token_budget_enforced": True,
            "phase_sequences_emitted": {phase: 256 for phase in phases},
            "consumed_phase_utf8_bytes": {phase: 500 for phase in phases},
        }
    return payload


def _result(training_identity: str, checkpoint: str, nll: float) -> dict:
    payload = {
        "schema": "sai-sub-4b-short-screen-v1",
        "evidence_class": "mechanics/development-screen-only",
        "scientific_promotion_authorized": False,
        "four_b_training_authorized": False,
        "config": {"mixer_family": "gated_gqa"},
        "config_sha256": "6" * 64,
        "model_sha256": "7" * 64,
        "delta_backend": "reference",
        "initialization_policy_sha256": "8" * 64,
        "initialization_seed": 20260821,
        "training_stream_identity_sha256": training_identity,
        "development_stream_identity_sha256": DEVELOPMENT_ID,
        "code_sha256": "9" * 64,
        "environment_sha256": "a" * 64,
        "optimizer": {
            "optimizer_steps": 954,
            "learning_rate": 0.0006,
        },
        "precision": {"activation_execution": "bfloat16_autocast"},
        "micro_batch_size_sequences": 8,
        "sequences_per_update": 256,
        "training_sequences": 244_140,
        "training_utf8_bytes": 10_000,
        "development_sequences": 1_024,
        "development_batch_size_sequences": 8,
        "checkpoint_interval_steps": 10,
        "mechanics_only": False,
    }
    payload["run_sha256"] = canonical_sha256(payload)
    payload.update(
        {
            "status": "complete",
            "parameter_count": 100_481_024,
            "initialization": {"seed": 20260821},
            "counters": {
                "optimizer_steps": 954,
                "sequences": 244_140,
                "targets": 498_000_000,
            },
            "development_nll": {
                "stream_identity_sha256": DEVELOPMENT_ID,
                "sequences": 1_024,
                "targets": 2_000_000,
                "admitted_utf8_bytes": 2_000,
                "negative_log_likelihood": nll * 2_000_000,
                "nll_per_target": nll,
                "perplexity": 2.718281828**nll,
                "nll_per_utf8_byte": nll * 1_000,
                "strata": {
                    phase: {
                        "sequences": 256,
                        "targets": 500_000,
                        "admitted_utf8_bytes": 500,
                        "negative_log_likelihood": nll * 500_000,
                        "nll_per_target": nll,
                        "perplexity": 2.718281828**nll,
                        "nll_per_utf8_byte": nll * 1_000,
                    }
                    for phase in (
                        "grounding",
                        "integration",
                        "reasoning",
                        "specialization",
                    )
                },
            },
            "checkpoint": {"sha256": checkpoint},
        }
    )
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path]:
    split = tmp_path / "split.json"
    split.write_text("{}\n")
    split_sha = comparison.sha256_file(split)
    curriculum = _stream(CURRICULUM_ID, sequences=244_140, source="train.jsonl")
    control = _stream(CONTROL_ID, sequences=244_140, source="train.jsonl")
    development = _stream(DEVELOPMENT_ID, sequences=1_024, source="development.jsonl")
    for stream in (curriculum, control, development):
        stream["source_qualification_sha256"] = split_sha
    control["ordering_control"] = {
        "parent_stream": {"ordered_stream_identity_sha256": CURRICULUM_ID},
        "same_tokens_and_boundary_masks": True,
        "same_sequence_multiset": True,
        "only_sequence_order_changed": True,
        "sequence_multiset_sha256": "b" * 64,
        "permutation_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        comparison,
        "validate_curriculum_split",
        lambda _, **__: {
            "receipt_sha256": SPLIT_ID,
            "train": {"path": "train.jsonl"},
            "development": {"path": "development.jsonl"},
        },
    )
    monkeypatch.setattr(
        comparison,
        "validate_frozen_stream",
        lambda path, verify_sources: (
            curriculum if path.name == "curriculum" else development
        ),
    )
    monkeypatch.setattr(comparison, "validate_order_control", lambda _: control)
    curriculum_result = tmp_path / "curriculum.result.json"
    control_result = tmp_path / "control.result.json"
    curriculum_result.write_text(json.dumps(_result(CURRICULUM_ID, "d" * 64, 5.0)))
    control_result.write_text(json.dumps(_result(CONTROL_ID, "e" * 64, 5.2)))
    return split, curriculum_result, control_result


def test_compares_only_order_and_writes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split, curriculum_result, control_result = _fixture(tmp_path, monkeypatch)
    payload = compare_curriculum_order(
        curriculum_result,
        control_result,
        curriculum_stream=Path("curriculum"),
        control_stream=Path("control"),
        development_stream=Path("development"),
        split_receipt=split,
    )
    assert payload["curriculum_order_supported_by_heldout_nll"] is True
    assert payload["curriculum_minus_control"]["nll_per_target"] == pytest.approx(-0.2)
    assert payload["heldout_phase_no_regression"] is True
    assert payload["phase_minus_control"]["specialization"][
        "nll_per_target"
    ] == pytest.approx(-0.2)
    assert payload["only_training_sequence_order_changed"] is True
    assert payload["replication_required_for_architecture_claim"] is True
    assert payload["four_b_training_authorized"] is False
    output = tmp_path / "comparison.json"
    write_comparison(output, payload)
    assert json.loads(output.read_text()) == payload
    with pytest.raises(CurriculumOrderComparisonError, match="output path differs"):
        write_comparison(output, payload)


def test_rejects_resigned_mismatched_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split, curriculum_result, control_result = _fixture(tmp_path, monkeypatch)
    control = json.loads(control_result.read_text())
    control["optimizer"]["learning_rate"] = 0.0005
    specification = {
        field: control.get(field) for field in comparison.RUN_SPECIFICATION_FIELDS
    }
    control["run_sha256"] = canonical_sha256(specification)
    unsigned = dict(control)
    unsigned.pop("receipt_sha256")
    control["receipt_sha256"] = canonical_sha256(unsigned)
    control_result.write_text(json.dumps(control))
    with pytest.raises(
        CurriculumOrderComparisonError, match="matched training specification"
    ):
        compare_curriculum_order(
            curriculum_result,
            control_result,
            curriculum_stream=Path("curriculum"),
            control_stream=Path("control"),
            development_stream=Path("development"),
            split_receipt=split,
        )


def test_one_phase_regression_vetoes_an_aggregate_win(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split, curriculum_result, control_result = _fixture(tmp_path, monkeypatch)
    curriculum = json.loads(curriculum_result.read_text())
    specialization = curriculum["development_nll"]["strata"]["specialization"]
    old_nll = specialization["negative_log_likelihood"]
    specialization.update(
        {
            "negative_log_likelihood": 5.3 * 500_000,
            "nll_per_target": 5.3,
            "perplexity": 2.718281828**5.3,
            "nll_per_utf8_byte": 5_300.0,
        }
    )
    curriculum["development_nll"]["negative_log_likelihood"] += (
        specialization["negative_log_likelihood"] - old_nll
    )
    curriculum["development_nll"]["nll_per_target"] = (
        curriculum["development_nll"]["negative_log_likelihood"] / 2_000_000
    )
    curriculum["development_nll"]["perplexity"] = (
        2.718281828 ** curriculum["development_nll"]["nll_per_target"]
    )
    curriculum["development_nll"]["nll_per_utf8_byte"] = (
        curriculum["development_nll"]["negative_log_likelihood"] / 2_000
    )
    unsigned = dict(curriculum)
    unsigned.pop("receipt_sha256")
    curriculum["receipt_sha256"] = canonical_sha256(unsigned)
    curriculum_result.write_text(json.dumps(curriculum))

    payload = compare_curriculum_order(
        curriculum_result,
        control_result,
        curriculum_stream=Path("curriculum"),
        control_stream=Path("control"),
        development_stream=Path("development"),
        split_receipt=split,
    )
    assert payload["curriculum_minus_control"]["nll_per_target"] < 0
    assert payload["phase_minus_control"]["specialization"]["nll_per_target"] > 0
    assert payload["heldout_phase_no_regression"] is False
    assert payload["curriculum_order_supported_by_heldout_nll"] is False


def test_launcher_is_matched_independent_single_h100_and_nonretrying() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "jobs" / "sai-launch-curriculum-order-screen-cpu.sbatch"
    ).read_text()
    comparator = (root / "jobs" / "sai-compare-curriculum-order-cpu.sbatch").read_text()
    for job in (launcher, comparator):
        assert "#SBATCH --no-requeue" in job
        assert "#SBATCH --gres=" not in job
        assert 'rev-parse HEAD)" = "$EXPECTED_COMMIT"' in job
        assert "retry" not in job.lower()
    assert "validate_curriculum_order_bundle" in launcher
    assert "curriculum_workers=curriculum_workers" in launcher
    assert "#SBATCH --cpus-per-task=8" in launcher
    assert "#SBATCH --mem=16G" in launcher
    assert "#SBATCH --time=06:00:00" in launcher
    assert "sbatch --parsable --time=04:00:00" in launcher
    assert launcher.count("--time=18:00:00") == 2
    assert "MIN_QUOTA_HEADROOM_KIB=25165824" in launcher
    assert "MIN_QUOTA_HEADROOM_FILES=10000" in launcher
    assert "/usr/bin/lfs quota" in launcher
    assert '"quota_preflight"' in launcher
    assert "MIN_QUOTA_HEADROOM_KIB=$MIN_QUOTA_HEADROOM_KIB" in launcher
    assert "MIN_QUOTA_HEADROOM_FILES=$MIN_QUOTA_HEADROOM_FILES" in launcher
    assert "FAMILY=gated_gqa" in launcher
    assert "TRAINING_SEQUENCES=244140" in launcher
    assert "OPTIMIZER_STEPS=954" in launcher
    assert launcher.count("MILESTONE_STEPS=191:429:715") == 2
    assert '"milestone_steps_per_arm": [191, 429, 715]' in launcher
    assert "SEED=20260821" in launcher
    assert "MECHANICS_ONLY=1" in launcher
    assert '--dependency="afterok:$canary_job"' in launcher
    assert '--dependency="afterok:$curriculum_job:$control_job"' in launcher
    assert '"gpu_jobs_submitted": 3' in launcher
    assert '"maximum_concurrent_gpu_jobs": 2' in launcher
    assert '"four_b_training_authorized": False' in launcher
    assert "sai.evaluation.curriculum_order_compare" in comparator
    assert 'test "$row" = "COMPLETED|0:0|0"' in comparator
    assert "--curriculum-result" in comparator
    assert "--control-result" in comparator
