"""Exact held-out NLL evaluation for Sai mechanics checkpoints."""

from __future__ import annotations

import math
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from sai.model.reference import SaiCausalLM
from sai.training.runner import (
    CausalTrainingBatch,
    TrainingRunnerError,
    _validate_batch,
)


class TrainingEvaluationError(RuntimeError):
    """A validation population, score, or model-state invariant differs."""


@dataclass(frozen=True)
class ValidationResult:
    stream_identity_sha256: str
    sequences: int
    targets: int
    admitted_utf8_bytes: int
    negative_log_likelihood: float
    nll_per_target: float
    perplexity: float
    nll_per_utf8_byte: float


def _state_versions(model: SaiCausalLM) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, tensor._version)
        for name, tensor in sorted(model.state_dict(keep_vars=True).items())
    )


def evaluate_nll(
    model: SaiCausalLM,
    batches: Iterable[CausalTrainingBatch],
    *,
    stream_identity_sha256: str,
    expected_sequences: int,
    admitted_utf8_bytes: int,
    benchmark_disjoint: bool,
    autocast_dtype: torch.dtype | None = None,
) -> ValidationResult:
    """Evaluate a complete frozen validation prefix without changing model state."""

    if (
        not isinstance(model, SaiCausalLM)
        or not isinstance(stream_identity_sha256, str)
        or len(stream_identity_sha256) != 64
        or benchmark_disjoint is not True
        or isinstance(expected_sequences, bool)
        or not isinstance(expected_sequences, int)
        or expected_sequences <= 0
        or isinstance(admitted_utf8_bytes, bool)
        or not isinstance(admitted_utf8_bytes, int)
        or admitted_utf8_bytes <= 0
    ):
        raise TrainingEvaluationError("validation contract differs")
    try:
        bytes.fromhex(stream_identity_sha256)
    except ValueError as error:
        raise TrainingEvaluationError("validation stream identity differs") from error

    if autocast_dtype is not None and autocast_dtype is not torch.bfloat16:
        raise TrainingEvaluationError("validation autocast dtype differs")
    model_device = next(model.parameters()).device
    if autocast_dtype is not None and model_device.type != "cuda":
        raise TrainingEvaluationError("validation autocast requires CUDA")

    original_training = model.training
    versions_before = _state_versions(model)
    total_nll = 0.0
    sequences = targets = 0
    model.eval()
    try:
        autocast = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            for batch in batches:
                try:
                    target_count = _validate_batch(batch)
                except TrainingRunnerError as error:
                    raise TrainingEvaluationError("validation batch differs") from error
                logits = model(batch.input_ids, batch.segment_ids)
                expected_shape = (*batch.input_ids.shape, model.config.vocab_size)
                if logits.shape != expected_shape:
                    raise TrainingEvaluationError("validation logit geometry differs")
                selected_logits = logits[batch.target_mask].float()
                selected_targets = batch.target_ids[batch.target_mask]
                loss_sum = F.cross_entropy(
                    selected_logits, selected_targets, reduction="sum"
                )
                if not torch.isfinite(loss_sum).item():
                    raise TrainingEvaluationError("validation NLL is nonfinite")
                total_nll += float(loss_sum)
                sequences += batch.input_ids.shape[0]
                targets += target_count
    finally:
        model.train(original_training)

    if sequences != expected_sequences or targets <= 0:
        raise TrainingEvaluationError("validation coverage differs")
    if _state_versions(model) != versions_before:
        raise TrainingEvaluationError("validation mutated model state")
    nll_per_target = total_nll / targets
    if not math.isfinite(nll_per_target):
        raise TrainingEvaluationError("validation aggregate is nonfinite")
    return ValidationResult(
        stream_identity_sha256=stream_identity_sha256,
        sequences=sequences,
        targets=targets,
        admitted_utf8_bytes=admitted_utf8_bytes,
        negative_log_likelihood=total_nll,
        nll_per_target=nll_per_target,
        perplexity=math.exp(nll_per_target),
        nll_per_utf8_byte=total_nll / admitted_utf8_bytes,
    )
