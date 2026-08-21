"""Minimal, fail-closed training loop for auditable Sai mechanics runs."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from sai.model.reference import SaiCausalLM
from sai.training.stream import TrainingBatch


class TrainingRunnerError(RuntimeError):
    """A batch, optimization value, or finite-training invariant differs."""


@dataclass(frozen=True)
class CausalTrainingBatch:
    """One packed batch whose mask marks valid next-token predictions."""

    input_ids: torch.Tensor
    target_ids: torch.Tensor
    target_mask: torch.Tensor
    segment_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class TrainingRunConfig:
    """Optimizer and deterministic step-schedule inputs for one run."""

    optimizer_steps: int
    learning_rate: float = 6e-4
    warmup_steps: int = 0
    minimum_learning_rate_ratio: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    gradient_clip_norm: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.optimizer_steps, bool)
            or not isinstance(self.optimizer_steps, int)
            or self.optimizer_steps <= 0
            or isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or not 0 <= self.warmup_steps <= self.optimizer_steps
        ):
            raise TrainingRunnerError("optimizer-step schedule differs")
        finite_positive = (
            (self.learning_rate, "learning rate"),
            (self.epsilon, "AdamW epsilon"),
            (self.gradient_clip_norm, "gradient clip norm"),
        )
        for value, field in finite_positive:
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise TrainingRunnerError(f"{field} must be finite and positive")
        if (
            isinstance(self.weight_decay, bool)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise TrainingRunnerError("weight decay must be finite and nonnegative")
        if (
            isinstance(self.minimum_learning_rate_ratio, bool)
            or not math.isfinite(self.minimum_learning_rate_ratio)
            or not 0 <= self.minimum_learning_rate_ratio <= 1
        ):
            raise TrainingRunnerError("minimum learning-rate ratio differs")
        if (
            not isinstance(self.betas, tuple)
            or len(self.betas) != 2
            or any(
                isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value < 1
                for value in self.betas
            )
        ):
            raise TrainingRunnerError("AdamW betas differ")


@dataclass(frozen=True)
class TrainingRunResult:
    """Exact work counters and observed scalar values for a completed run."""

    sequences: int
    targets: int
    optimizer_steps: int
    losses: tuple[float, ...]
    learning_rates: tuple[float, ...]


def tensorize_stream_batch(
    batch: TrainingBatch, *, device: torch.device | str
) -> CausalTrainingBatch:
    """Move one receipt-bound stream batch into the explicit runner contract."""

    if not isinstance(batch, TrainingBatch):
        raise TrainingRunnerError("receipt-bound stream batch type differs")
    return CausalTrainingBatch(
        input_ids=torch.tensor(batch.x, dtype=torch.long, device=device),
        target_ids=torch.tensor(batch.y, dtype=torch.long, device=device),
        target_mask=torch.tensor(batch.loss_mask, dtype=torch.bool, device=device),
        segment_ids=torch.tensor(batch.segment_ids, dtype=torch.long, device=device),
    )


def learning_rate_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    """Return the deterministic multiplier for a one-indexed optimizer step."""

    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or not 1 <= step <= total_steps
        or not 0 <= warmup_steps <= total_steps
        or isinstance(minimum_ratio, bool)
        or not math.isfinite(minimum_ratio)
        or not 0 <= minimum_ratio <= 1
    ):
        raise TrainingRunnerError("learning-rate schedule inputs differ")
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    decay_steps = total_steps - warmup_steps
    if decay_steps == 0:
        return 1.0
    if warmup_steps:
        progress = (step - warmup_steps) / decay_steps
    elif total_steps == 1:
        progress = 0.0
    else:
        progress = (step - 1) / (total_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


def _parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    matrices = []
    vectors = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (matrices if parameter.ndim >= 2 else vectors).append(parameter)
    if not matrices:
        raise TrainingRunnerError("model has no trainable matrix parameters")
    groups: list[dict[str, object]] = [
        {"params": matrices, "weight_decay": weight_decay}
    ]
    if vectors:
        groups.append({"params": vectors, "weight_decay": 0.0})
    return groups


def build_adamw(model: nn.Module, config: TrainingRunConfig) -> torch.optim.AdamW:
    """Build AdamW with decay applied to matrix parameters only."""

    return torch.optim.AdamW(
        _parameter_groups(model, config.weight_decay),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.epsilon,
    )


def _validate_batch(batch: CausalTrainingBatch) -> int:
    if not isinstance(batch, CausalTrainingBatch):
        raise TrainingRunnerError("training batch type differs")
    input_ids = batch.input_ids
    target_ids = batch.target_ids
    target_mask = batch.target_mask
    if (
        input_ids.ndim != 2
        or input_ids.dtype != torch.long
        or input_ids.shape[0] <= 0
        or input_ids.shape[1] <= 1
    ):
        raise TrainingRunnerError("input IDs must be a nonempty rank-two LongTensor")
    if target_ids.shape != input_ids.shape or target_ids.dtype != torch.long:
        raise TrainingRunnerError("target IDs must match input IDs as a LongTensor")
    if target_mask.shape != input_ids.shape or target_mask.dtype != torch.bool:
        raise TrainingRunnerError("target mask must match input IDs as a BoolTensor")
    if target_ids.device != input_ids.device or target_mask.device != input_ids.device:
        raise TrainingRunnerError("target tensor device differs")
    target_count = int(target_mask.sum().item())
    if target_count <= 0:
        raise TrainingRunnerError("training batch has no valid next-token targets")
    if batch.segment_ids is not None and batch.segment_ids.device != input_ids.device:
        raise TrainingRunnerError("segment-ID device differs")
    return target_count


def _masked_next_token_loss(
    model: SaiCausalLM, batch: CausalTrainingBatch
) -> tuple[torch.Tensor, int]:
    target_count = _validate_batch(batch)
    logits = model(batch.input_ids, batch.segment_ids)
    expected_shape = (*batch.input_ids.shape, model.config.vocab_size)
    if logits.shape != expected_shape:
        raise TrainingRunnerError("model logit geometry differs")
    selected_logits = logits[batch.target_mask].float()
    selected_targets = batch.target_ids[batch.target_mask]
    loss = F.cross_entropy(selected_logits, selected_targets, reduction="mean")
    if not torch.isfinite(loss).item():
        raise TrainingRunnerError("training loss is nonfinite")
    return loss, target_count


def _assert_finite_gradients(model: nn.Module) -> None:
    observed = False
    for parameter in model.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        observed = True
        if not torch.isfinite(parameter.grad).all().item():
            raise TrainingRunnerError("training gradient is nonfinite")
    if not observed:
        raise TrainingRunnerError("training produced no gradients")


def train(
    model: SaiCausalLM,
    batches: Iterable[CausalTrainingBatch],
    config: TrainingRunConfig,
) -> TrainingRunResult:
    """Run exactly the configured updates or fail without claiming completion."""

    if not isinstance(model, SaiCausalLM):
        raise TrainingRunnerError("runner requires SaiCausalLM")
    iterator = iter(batches)
    optimizer = build_adamw(model, config)
    model.train()
    sequences = 0
    targets = 0
    losses = []
    learning_rates = []
    for step in range(1, config.optimizer_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration as error:
            raise TrainingRunnerError(
                "training stream ended before the exact budget"
            ) from error
        multiplier = learning_rate_multiplier(
            step,
            total_steps=config.optimizer_steps,
            warmup_steps=config.warmup_steps,
            minimum_ratio=config.minimum_learning_rate_ratio,
        )
        learning_rate = config.learning_rate * multiplier
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss, target_count = _masked_next_token_loss(model, batch)
        loss.backward()
        _assert_finite_gradients(model)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip_norm
        )
        if not torch.isfinite(gradient_norm).item():
            raise TrainingRunnerError("clipped gradient norm is nonfinite")
        optimizer.step()
        sequences += batch.input_ids.shape[0]
        targets += target_count
        losses.append(float(loss.detach()))
        learning_rates.append(learning_rate)
    return TrainingRunResult(
        sequences=sequences,
        targets=targets,
        optimizer_steps=config.optimizer_steps,
        losses=tuple(losses),
        learning_rates=tuple(learning_rates),
    )
