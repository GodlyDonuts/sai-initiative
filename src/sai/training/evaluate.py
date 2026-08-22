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
class ValidationStratumResult:
    sequences: int
    targets: int
    admitted_utf8_bytes: int
    negative_log_likelihood: float
    nll_per_target: float
    perplexity: float
    nll_per_utf8_byte: float


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
    strata: dict[str, ValidationStratumResult] | None


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
    sequence_strata: list[tuple[str, int, int]] | None = None,
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
    normalized_strata = [] if sequence_strata is None else list(sequence_strata)
    if normalized_strata and (
        len({name for name, _, _ in normalized_strata}) != len(normalized_strata)
        or any(
            not isinstance(name, str)
            or not name
            or isinstance(stratum_sequences, bool)
            or not isinstance(stratum_sequences, int)
            or stratum_sequences <= 0
            or isinstance(stratum_bytes, bool)
            or not isinstance(stratum_bytes, int)
            or stratum_bytes <= 0
            for name, stratum_sequences, stratum_bytes in normalized_strata
        )
        or sum(row[1] for row in normalized_strata) != expected_sequences
        or sum(row[2] for row in normalized_strata) != admitted_utf8_bytes
    ):
        raise TrainingEvaluationError("validation strata differ")
    stratum_boundaries: list[tuple[str, int]] = []
    stratum_accumulators: dict[str, dict[str, float | int]] = {}
    cumulative_sequences = 0
    for name, stratum_sequences, _ in normalized_strata:
        cumulative_sequences += stratum_sequences
        stratum_boundaries.append((name, cumulative_sequences))
        stratum_accumulators[name] = {
            "sequences": 0,
            "targets": 0,
            "negative_log_likelihood": 0.0,
        }

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
                selected_losses = F.cross_entropy(
                    selected_logits, selected_targets, reduction="none"
                )
                losses = selected_losses.sum()
                if not torch.isfinite(losses).item():
                    raise TrainingEvaluationError("validation NLL is nonfinite")
                total_nll += float(losses)
                if normalized_strata:
                    row_target_counts = batch.target_mask.sum(dim=1).tolist()
                    offset = 0
                    for local_index, row_targets in enumerate(row_target_counts):
                        if row_targets <= 0:
                            raise TrainingEvaluationError(
                                "validation stratum sequence has no target"
                            )
                        global_index = sequences + local_index
                        stratum_name = next(
                            name
                            for name, boundary in stratum_boundaries
                            if global_index < boundary
                        )
                        row_nll = float(
                            selected_losses[offset : offset + row_targets].sum()
                        )
                        accumulator = stratum_accumulators[stratum_name]
                        accumulator["sequences"] += 1
                        accumulator["targets"] += row_targets
                        accumulator["negative_log_likelihood"] += row_nll
                        offset += row_targets
                    if offset != target_count:
                        raise TrainingEvaluationError("validation strata differ")
                sequences += batch.input_ids.shape[0]
                targets += target_count
    finally:
        model.train(original_training)

    if sequences != expected_sequences or targets <= 0:
        raise TrainingEvaluationError("validation coverage differs")
    if _state_versions(model) != versions_before:
        raise TrainingEvaluationError("validation mutated model state")
    strata_result = None
    if normalized_strata:
        strata_result = {}
        for name, stratum_sequences, stratum_bytes in normalized_strata:
            accumulator = stratum_accumulators[name]
            stratum_targets = int(accumulator["targets"])
            stratum_nll = float(accumulator["negative_log_likelihood"])
            if (
                accumulator["sequences"] != stratum_sequences
                or stratum_targets <= 0
                or not math.isfinite(stratum_nll)
            ):
                raise TrainingEvaluationError("validation stratum coverage differs")
            stratum_nll_per_target = stratum_nll / stratum_targets
            strata_result[name] = ValidationStratumResult(
                sequences=stratum_sequences,
                targets=stratum_targets,
                admitted_utf8_bytes=stratum_bytes,
                negative_log_likelihood=stratum_nll,
                nll_per_target=stratum_nll_per_target,
                perplexity=math.exp(stratum_nll_per_target),
                nll_per_utf8_byte=stratum_nll / stratum_bytes,
            )
        total_nll = sum(row.negative_log_likelihood for row in strata_result.values())
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
        strata=strata_result,
    )
