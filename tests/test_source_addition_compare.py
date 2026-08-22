from __future__ import annotations

import copy
import json

import pytest

from sai.data.token_stream import canonical_sha256
from sai.evaluation.source_addition_compare import (
    SourceAdditionComparisonError,
    _load_result,
    compare_payloads,
)
from sai.model.config import SaiModelConfig, parameter_ledger
from sai.model.initialization import POLICY_SHA256


def _nll(base: float) -> dict:
    rows = {
        "grounding": {
            "sequences": 2,
            "targets": 20,
            "admitted_utf8_bytes": 100,
            "negative_log_likelihood": 20.0 * base,
            "nll_per_target": base,
            "perplexity": 2.0,
            "nll_per_utf8_byte": base / 5.0,
        },
        "specialization": {
            "sequences": 2,
            "targets": 20,
            "admitted_utf8_bytes": 100,
            "negative_log_likelihood": 20.0 * (base + 0.2),
            "nll_per_target": base + 0.2,
            "perplexity": 2.1,
            "nll_per_utf8_byte": (base + 0.2) / 5.0,
        },
    }
    total = sum(row["negative_log_likelihood"] for row in rows.values())
    return {
        "stream_identity_sha256": "d" * 64,
        "sequences": 4,
        "targets": 40,
        "admitted_utf8_bytes": 200,
        "negative_log_likelihood": total,
        "nll_per_target": total / 40,
        "perplexity": 2.05,
        "nll_per_utf8_byte": total / 200,
        "strata": rows,
    }


def _result(*, treatment: bool, base: float) -> dict:
    shared = {
        "config": {"mixer_family": "gated_gqa"},
        "config_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "delta_backend": "reference",
        "initialization_policy_sha256": "3" * 64,
        "initialization_seed": 7,
        "development_stream_identity_sha256": "d" * 64,
        "code_sha256": "4" * 64,
        "environment_sha256": "5" * 64,
        "optimizer": {"optimizer_steps": 2},
        "precision": "bf16_execution_fp32_master_and_optimizer",
        "micro_batch_size_sequences": 2,
        "sequences_per_update": 4,
        "training_sequences": 8,
        "development_sequences": 4,
        "development_batch_size_sequences": 2,
        "checkpoint_interval_steps": 1,
        "mechanics_only": False,
        "parameter_count": 100,
        "initialization": {"seed": 7},
    }
    return {
        **shared,
        "run_sha256": ("6" if treatment else "7") * 64,
        "training_stream_identity_sha256": ("8" if treatment else "9") * 64,
        "development_nll": _nll(base),
    }


def _stream(*, treatment: bool) -> dict:
    return {
        "schema": "sai-frozen-ordered-token-stream-v1",
        "status": "complete",
        "tokenizer_identity_sha256": "a" * 64,
        "source_manifest_sha256": ("b" if treatment else "c") * 64,
        "source_receipts": (
            [{"source": "web"}, {"source": "math"}]
            if treatment
            else [{"source": "web"}]
        ),
        "source_qualification_sha256": ("e" if treatment else "f") * 64,
        "sequence_length": 8,
        "sequences": 8,
        "valid_tokens": 64,
        "admitted_utf8_bytes": 512 if treatment else 480,
        "benchmark_disjoint": True,
        "cross_document_targets_masked": True,
        "token_encoding": "little_endian_uint32",
        "segment_start_encoding": "little_endian_bitset_lsb_first",
        "eos_token_id": 1,
        "vocab_size": 48_000,
        "ordered_stream_identity_sha256": ("8" if treatment else "9") * 64,
    }


def _development() -> dict:
    return {
        "ordered_stream_identity_sha256": "d" * 64,
        "sequence_length": 8,
        "sequences": 4,
        "valid_tokens": 32,
    }


def test_equal_token_source_addition_must_pass_every_nll_stratum() -> None:
    payload = compare_payloads(
        _result(treatment=True, base=2.8),
        _result(treatment=False, base=3.0),
        treatment_stream=_stream(treatment=True),
        control_stream=_stream(treatment=False),
        development_stream=_development(),
    )
    assert payload["equal_training_tokens"] == 64
    assert payload["source_addition_supported_by_heldout_nll"] is True
    assert payload["source_addition_retained"] is False
    assert payload["real_source_disjoint_benchmark_confirmation_required"] is True
    assert payload["four_b_training_authorized"] is False


def test_one_stratum_regression_vetoes_source_addition() -> None:
    treatment = _result(treatment=True, base=2.8)
    treatment["development_nll"]["strata"]["specialization"]["nll_per_utf8_byte"] = 1.0
    payload = compare_payloads(
        treatment,
        _result(treatment=False, base=3.0),
        treatment_stream=_stream(treatment=True),
        control_stream=_stream(treatment=False),
        development_stream=_development(),
    )
    assert payload["source_addition_supported_by_heldout_nll"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda result, stream: result.update(initialization_seed=8),
        lambda result, stream: result.update(run_sha256="6" * 64),
        lambda result, stream: result.update(training_stream_identity_sha256="8" * 64),
        lambda result, stream: stream.update(valid_tokens=56),
        lambda result, stream: stream.update(source_qualification_sha256="e" * 64),
    ),
)
def test_rejects_unmatched_budget_or_source_identity(mutation) -> None:
    control_result = copy.deepcopy(_result(treatment=False, base=3.0))
    control_stream = copy.deepcopy(_stream(treatment=False))
    mutation(control_result, control_stream)
    with pytest.raises(SourceAdditionComparisonError):
        compare_payloads(
            _result(treatment=True, base=2.8),
            control_result,
            treatment_stream=_stream(treatment=True),
            control_stream=control_stream,
            development_stream=_development(),
        )


def _terminal_payload() -> dict:
    config = SaiModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        mixer_family="gated_gqa",
    )
    config_sha256 = canonical_sha256(config.as_dict())
    model_sha256 = canonical_sha256(
        {
            "config_sha256": config_sha256,
            "delta_backend": "reference",
            "initialization_policy_sha256": POLICY_SHA256,
            "initialization_seed": 7,
        }
    )
    specification = {
        "schema": "sai-sub-4b-short-screen-v1",
        "evidence_class": "sub_4b_mechanics_and_development_nll",
        "scientific_promotion_authorized": False,
        "four_b_training_authorized": False,
        "config": config.as_dict(),
        "config_sha256": config_sha256,
        "model_sha256": model_sha256,
        "delta_backend": "reference",
        "initialization_policy_sha256": POLICY_SHA256,
        "initialization_seed": 7,
        "training_stream_identity_sha256": "8" * 64,
        "development_stream_identity_sha256": "d" * 64,
        "code_sha256": "4" * 64,
        "environment_sha256": "5" * 64,
        "optimizer": {"optimizer_steps": 2},
        "precision": "bf16_execution_fp32_master_and_optimizer",
        "micro_batch_size_sequences": 2,
        "sequences_per_update": 4,
        "training_sequences": 8,
        "training_utf8_bytes": 512,
        "development_sequences": 4,
        "development_batch_size_sequences": 2,
        "checkpoint_interval_steps": 1,
        "mechanics_only": False,
    }
    payload = {
        **specification,
        "status": "complete",
        "parameter_count": parameter_ledger(config)["total"],
        "initialization": {"seed": 7},
        "run_sha256": canonical_sha256(specification),
        "development_nll": _nll(3.0),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def test_terminal_loader_recomputes_model_and_run_identity(tmp_path) -> None:
    path = tmp_path / "result.json"
    payload = _terminal_payload()
    path.write_text(json.dumps(payload))
    observed, file_sha256 = _load_result(path)
    assert observed == payload
    assert len(file_sha256) == 64

    payload["run_sha256"] = "f" * 64
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256")
    payload["receipt_sha256"] = canonical_sha256(unsigned)
    path.write_text(json.dumps(payload))
    with pytest.raises(SourceAdditionComparisonError, match="receipt differs"):
        _load_result(path)
