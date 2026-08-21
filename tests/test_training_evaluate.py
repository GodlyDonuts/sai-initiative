from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from sai.model.config import SaiModelConfig
from sai.model.reference import SaiCausalLM
from sai.training.evaluate import TrainingEvaluationError, evaluate_nll
from sai.training.runner import CausalTrainingBatch


def _model() -> SaiCausalLM:
    return SaiCausalLM(
        SaiModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            mixer_family="gated_gqa",
            mla_kv_rank=8,
            mla_qk_head_dim=8,
            mla_value_head_dim=8,
        )
    )


def _batch() -> CausalTrainingBatch:
    return CausalTrainingBatch(
        input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        target_ids=torch.tensor([[2, 3, 4, -100]], dtype=torch.long),
        target_mask=torch.tensor([[True, True, True, False]]),
        segment_ids=torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
    )


def test_validation_nll_matches_manual_sum_and_preserves_model() -> None:
    torch.manual_seed(20260821)
    model = _model()
    model.train()
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    batch = _batch()
    with torch.inference_mode():
        logits = model(batch.input_ids, batch.segment_ids)
        expected = float(
            F.cross_entropy(
                logits[batch.target_mask].float(),
                batch.target_ids[batch.target_mask],
                reduction="sum",
            )
        )

    result = evaluate_nll(
        model,
        [batch],
        stream_identity_sha256="a" * 64,
        expected_sequences=1,
        admitted_utf8_bytes=11,
        benchmark_disjoint=True,
    )
    assert model.training is True
    assert result.negative_log_likelihood == pytest.approx(expected)
    assert result.nll_per_target == pytest.approx(expected / 3)
    assert result.perplexity == pytest.approx(math.exp(expected / 3))
    assert result.nll_per_utf8_byte == pytest.approx(expected / 11)
    assert all(
        torch.equal(model.state_dict()[name], value) for name, value in state.items()
    )


def test_validation_requires_complete_disjoint_coverage() -> None:
    model = _model()
    with pytest.raises(TrainingEvaluationError, match="contract"):
        evaluate_nll(
            model,
            [_batch()],
            stream_identity_sha256="a" * 64,
            expected_sequences=1,
            admitted_utf8_bytes=11,
            benchmark_disjoint=False,
        )
    with pytest.raises(TrainingEvaluationError, match="coverage"):
        evaluate_nll(
            model,
            [_batch()],
            stream_identity_sha256="a" * 64,
            expected_sequences=2,
            admitted_utf8_bytes=11,
            benchmark_disjoint=True,
        )
    with pytest.raises(TrainingEvaluationError, match="autocast requires CUDA"):
        evaluate_nll(
            model,
            [_batch()],
            stream_identity_sha256="a" * 64,
            expected_sequences=1,
            admitted_utf8_bytes=11,
            benchmark_disjoint=True,
            autocast_dtype=torch.bfloat16,
        )
