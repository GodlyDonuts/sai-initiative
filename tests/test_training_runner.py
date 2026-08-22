from __future__ import annotations

import math

import pytest
import torch

from sai.model.config import SaiModelConfig
from sai.model.reference import SaiCausalLM
from sai.training.runner import (
    CausalTrainingBatch,
    TrainingRunConfig,
    TrainingRunnerError,
    build_adamw,
    learning_rate_multiplier,
    tensorize_stream_batch,
    train,
)
from sai.training.stream import StreamCursor, TrainingBatch


def tiny_gated_gqa() -> SaiCausalLM:
    return SaiCausalLM(
        SaiModelConfig(
            vocab_size=32,
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


def batch(offset: int = 0) -> CausalTrainingBatch:
    input_ids = (
        torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.long) + offset
    ) % 32
    return CausalTrainingBatch(
        input_ids=input_ids,
        target_ids=(
            torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]], dtype=torch.long)
            + offset
        )
        % 32,
        target_mask=torch.tensor(
            [[True, True, False, True, False], [True, False, True, True, False]]
        ),
        segment_ids=torch.tensor([[0, 0, 0, 1, 1], [2, 2, 3, 3, 3]]),
    )


def test_two_step_cpu_gated_gqa_smoke_reports_exact_work() -> None:
    torch.manual_seed(20260821)
    model = tiny_gated_gqa()
    initial = model.embed_tokens.weight.detach().clone()
    result = train(
        model,
        [batch(), batch(1)],
        TrainingRunConfig(
            optimizer_steps=2,
            learning_rate=1e-3,
            warmup_steps=1,
            minimum_learning_rate_ratio=0.25,
        ),
    )

    assert result.sequences == 4
    assert result.targets == 12
    assert result.optimizer_steps == 2
    assert result.learning_rates == pytest.approx((1e-3, 2.5e-4))
    assert len(result.losses) == 2
    assert all(math.isfinite(loss) and loss > 0 for loss in result.losses)
    assert not torch.equal(model.embed_tokens.weight, initial)


def test_receipt_bound_stream_batch_tensorizes_without_a_second_shift() -> None:
    cursor = StreamCursor("a" * 64, 1)
    packed = TrainingBatch(
        x=((1, 2, 3),),
        y=((2, -100, 4),),
        segment_ids=((0, 0, 1),),
        loss_mask=((True, False, True),),
        first_sequence=0,
        resume_cursor=cursor,
    )
    observed = tensorize_stream_batch(packed, device="cpu")
    assert observed.input_ids.tolist() == [[1, 2, 3]]
    assert observed.target_ids.tolist() == [[2, -100, 4]]
    assert observed.target_mask.tolist() == [[True, False, True]]
    assert observed.segment_ids is not None
    assert observed.segment_ids.tolist() == [[0, 0, 1]]


def test_adamw_decays_only_matrix_parameters() -> None:
    model = tiny_gated_gqa()
    optimizer = build_adamw(
        model, TrainingRunConfig(optimizer_steps=1, weight_decay=0.125)
    )
    by_identity = {
        id(parameter): parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    decays = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert set(decays) == set(by_identity)
    assert all(
        decays[identity] == (0.125 if parameter.ndim >= 2 else 0.0)
        for identity, parameter in by_identity.items()
    )


def test_linear_warmup_cosine_schedule_has_exact_endpoints() -> None:
    values = [
        learning_rate_multiplier(step, total_steps=6, warmup_steps=2, minimum_ratio=0.1)
        for step in range(1, 7)
    ]
    assert values[:2] == [0.5, 1.0]
    assert values[2] < values[1]
    assert values == sorted(values[:2]) + sorted(values[2:], reverse=True)
    assert values[-1] == pytest.approx(0.1)
    no_warmup = [
        learning_rate_multiplier(step, total_steps=3, warmup_steps=0, minimum_ratio=0.1)
        for step in range(1, 4)
    ]
    assert no_warmup == pytest.approx([1.0, 0.55, 0.1])


@pytest.mark.parametrize(
    "bad_batch,match",
    [
        (
            CausalTrainingBatch(
                torch.ones(1, 3, dtype=torch.long),
                torch.ones(1, 3, dtype=torch.float32),
                torch.tensor([[True, True, True]]),
            ),
            "target IDs",
        ),
        (
            CausalTrainingBatch(
                torch.ones(1, 3, dtype=torch.long),
                torch.ones(1, 3, dtype=torch.long),
                torch.zeros(1, 3, dtype=torch.bool),
            ),
            "no valid",
        ),
    ],
)
def test_invalid_target_masks_fail_closed(
    bad_batch: CausalTrainingBatch, match: str
) -> None:
    with pytest.raises(TrainingRunnerError, match=match):
        train(tiny_gated_gqa(), [bad_batch], TrainingRunConfig(optimizer_steps=1))


def test_exact_optimizer_budget_rejects_short_stream() -> None:
    with pytest.raises(TrainingRunnerError, match="ended before"):
        train(tiny_gated_gqa(), [batch()], TrainingRunConfig(optimizer_steps=2))


def test_optimizer_budget_does_not_consume_unadmitted_batches() -> None:
    batches = iter([batch(), batch(1)])
    result = train(tiny_gated_gqa(), batches, TrainingRunConfig(optimizer_steps=1))
    assert result.optimizer_steps == 1
    assert next(batches).input_ids[0, 0].item() == 2


def test_nonfinite_loss_fails_closed() -> None:
    model = tiny_gated_gqa()
    with torch.no_grad():
        model.embed_tokens.weight[0, 0] = float("nan")
    poisoned = CausalTrainingBatch(
        input_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        target_ids=torch.tensor([[1, 2, 0]], dtype=torch.long),
        target_mask=torch.tensor([[True, True, False]]),
    )
    with pytest.raises(TrainingRunnerError, match="loss is nonfinite"):
        train(model, [poisoned], TrainingRunConfig(optimizer_steps=1))


def test_nonfinite_gradient_fails_closed() -> None:
    model = tiny_gated_gqa()
    model.embed_tokens.weight.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    with pytest.raises(TrainingRunnerError, match="gradient is nonfinite"):
        train(model, [batch()], TrainingRunConfig(optimizer_steps=1))
