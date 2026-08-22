from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sai.data.learnability_score import (
    OUTPUT_NAME,
    RECEIPT_NAME,
    SCHEMA,
    LearnabilityScoreError,
    exact_record_independence,
    score_model_pair,
    validate_score_population,
)
from sai.data.token_stream import canonical_sha256, freeze
from sai.model.config import SaiModelConfig
from sai.model.reference import SaiCausalLM
from tests.test_token_stream import CharacterTokenizer, document, write_documents


def _model() -> SaiCausalLM:
    return SaiCausalLM(
        SaiModelConfig(
            vocab_size=512,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            mixer_family="gated_gqa",
            mla_kv_rank=8,
            mla_qk_head_dim=8,
            mla_value_head_dim=8,
        )
    )


def _stream(tmp_path: Path, name: str, text: str) -> tuple[Path, dict]:
    source = write_documents(
        tmp_path / f"{name}.jsonl",
        [document(0, text)],
    )
    root = tmp_path / name
    report = freeze(
        CharacterTokenizer(),
        [source],
        root,
        tokenizer_identity_sha256="1" * 64,
        sequence_length=8,
        prefix_sequences={2},
        sequences_per_shard=1,
        source_qualification_sha256="2" * 64,
    )
    return root, report


def test_scores_every_exact_sequence_without_mutating_models_or_rng(
    tmp_path: Path,
) -> None:
    stream, report = _stream(tmp_path, "target", "abcdefghijklmno")
    torch.manual_seed(7)
    weak = _model()
    strong = _model()
    strong.load_state_dict(weak.state_dict())
    with torch.no_grad():
        strong.embed_tokens.weight.mul_(0.9)
    weak_state = {key: value.clone() for key, value in weak.state_dict().items()}
    strong_state = {key: value.clone() for key, value in strong.state_dict().items()}
    rng = torch.get_rng_state().clone()

    rows, targets = score_model_pair(
        weak,
        strong,
        stream,
        expected_stream_identity_sha256=report["ordered_stream_identity_sha256"],
        batch_size=2,
        device="cpu",
        autocast_dtype=None,
    )

    assert len(rows) == 2
    assert targets == sum(row["target_count"] for row in rows)
    assert [row["sequence_index"] for row in rows] == [0, 1]
    assert all(
        row["schema"] == "sai-model-centric-learnability-score-v1" for row in rows
    )
    assert all(
        row["preference_delta_microunits"]
        == row["weak_nll_microunits_per_target"]
        - row["strong_nll_microunits_per_target"]
        for row in rows
    )
    assert all(
        torch.equal(value, weak_state[key]) for key, value in weak.state_dict().items()
    )
    assert all(
        torch.equal(value, strong_state[key])
        for key, value in strong.state_dict().items()
    )
    assert torch.equal(torch.get_rng_state(), rng)


def test_exact_record_independence_rejects_overlap(tmp_path: Path) -> None:
    target, target_report = _stream(tmp_path, "target", "abcdefghijklmno")
    probe, probe_report = _stream(tmp_path, "probe", "qrstuvwxyzABCDE")
    receipt = exact_record_independence(target, target_report, probe, probe_report)
    assert receipt["exact_record_overlap_count"] == 0
    assert receipt["target_unique_records"] == 2
    assert receipt["probe_unique_records"] == 2

    with pytest.raises(LearnabilityScoreError, match="overlap"):
        exact_record_independence(target, target_report, target, target_report)


def _file(path: str, fill: str) -> dict:
    return {"path": path, "bytes": 10, "sha256": fill * 64}


def _receipt(root: Path, rows: list[dict]) -> dict:
    score_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    (root / OUTPUT_NAME).write_bytes(score_bytes)
    stream = {
        "path": "/immutable/stream",
        "receipt_file_sha256": "1" * 64,
        "ordered_stream_identity_sha256": "2" * 64,
        "source_manifest_sha256": "3" * 64,
        "tokenizer_identity_sha256": "4" * 64,
        "sequences": len(rows),
        "sequence_length": 8,
    }
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "target_stream": stream,
        "probe_training_stream": {
            **stream,
            "path": "/immutable/probe",
            "receipt_file_sha256": "5" * 64,
            "ordered_stream_identity_sha256": "6" * 64,
            "source_manifest_sha256": "7" * 64,
        },
        "exact_record_independence": {
            "method": "exact_sha256_of_tokens_and_boundary_mask",
            "target_unique_records": len(rows),
            "probe_unique_records": len(rows),
            "exact_record_overlap_count": 0,
        },
        "probe": {
            "family": "gated_gqa",
            "scale": "100m",
            "parameter_count": 100_000_000,
            "config_sha256": "8" * 64,
            "model_sha256": "9" * 64,
            "run_sha256": "a" * 64,
            "result": {
                **_file("/immutable/result.json", "b"),
                "receipt_sha256": "c" * 64,
            },
            "weak_milestone": {
                "path": "step-000191.model.pt",
                "bytes": 400,
                "sha256": "d" * 64,
                "optimizer_step": 191,
                "sequences": 48_896,
                "targets": 48_000,
                "model_state_sha256": "e" * 64,
            },
            "strong_checkpoint": {
                "checkpoint": _file("/immutable/strong.pt", "f"),
                "manifest": _file("/immutable/strong.manifest.json", "1"),
                "final_state_sha256": "2" * 64,
                "optimizer_step": 954,
            },
        },
        "scoring": {
            "method": "weak_minus_strong_normalized_nll_microunits",
            "evaluator_sha256": "3" * 64,
            "runtime_receipt": _file("/immutable/runtime.json", "4"),
            "execution_dtype": "bfloat16",
            "device_type": "cuda",
            "device_name": "NVIDIA H100 PCIe",
            "batch_size_sequences": 1,
            "sequences": len(rows),
            "targets": sum(row["target_count"] for row in rows),
            "inference_mode": True,
            "optimizer_steps": 0,
            "backward_calls": 0,
        },
        "scores": {
            "path": OUTPUT_NAME,
            "bytes": len(score_bytes),
            "sha256": __import__("hashlib").sha256(score_bytes).hexdigest(),
            "rows": len(rows),
            "ordered_population_sha256": canonical_sha256(rows),
        },
        "model_state_unchanged": True,
        "rng_state_unchanged": True,
        "limitations": [
            "exact_record_disjointness_does_not_prove_near_duplicate_disjointness",
            "probe_preferences_may_be_checkpoint_specific",
            "score_population_does_not_authorize_training_or_4b",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    (root / RECEIPT_NAME).write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def test_score_population_receipt_is_strict_and_tamper_evident(tmp_path: Path) -> None:
    stream, report = _stream(tmp_path, "target", "abcdefghijklmno")
    weak = _model()
    strong = _model()
    strong.load_state_dict(weak.state_dict())
    with torch.no_grad():
        strong.embed_tokens.weight.mul_(0.95)
    rows, _ = score_model_pair(
        weak,
        strong,
        stream,
        expected_stream_identity_sha256=report["ordered_stream_identity_sha256"],
        batch_size=1,
        device="cpu",
        autocast_dtype=None,
    )
    root = tmp_path / "scores"
    root.mkdir()
    payload = _receipt(root, rows)
    assert validate_score_population(root) == payload

    score_path = root / OUTPUT_NAME
    score_path.write_bytes(score_path.read_bytes() + b"{}\n")
    with pytest.raises(LearnabilityScoreError, match="row count|output"):
        validate_score_population(root)


def test_score_root_rejects_extra_members_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "scores"
    root.mkdir()
    (root / OUTPUT_NAME).write_text("{}\n")
    (root / RECEIPT_NAME).write_text("{}\n")
    (root / "extra").write_text("drift")
    with pytest.raises(LearnabilityScoreError, match="membership"):
        validate_score_population(root)

    unsafe = tmp_path / "unsafe"
    unsafe.symlink_to(root, target_is_directory=True)
    with pytest.raises(LearnabilityScoreError, match="unsafe"):
        validate_score_population(unsafe)
