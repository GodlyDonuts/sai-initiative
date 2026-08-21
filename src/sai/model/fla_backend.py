"""Version-pinned FLA adapters for Sai's delta mixers.

The reference model remains the scientific oracle.  This module only maps its
already-materialized tensors onto FLA 0.4.2 chunk kernels; it does not change
geometry, parameterization, or initialization.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise

import torch
import torch.nn.functional as F

FLA_VERSION = "0.4.2"


class FlaBackendError(RuntimeError):
    """The pinned FLA runtime or Sai-to-FLA tensor mapping differs."""


@dataclass(frozen=True)
class FlaBackendOperators:
    gated_delta_chunk: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
    kda_chunk: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
    causal_conv1d: Callable[..., torch.Tensor | tuple[torch.Tensor, ...]]
    version: str


def load_fla_backend_operators() -> FlaBackendOperators:
    """Load only the exact production APIs after the caller selects FLA."""

    version = None
    for distribution in ("fla-core", "flash-linear-attention"):
        try:
            version = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    if version != FLA_VERSION:
        raise FlaBackendError("FLA distribution version differs")
    try:
        gated_delta = importlib.import_module("fla.ops.gated_delta_rule")
        kda = importlib.import_module("fla.ops.kda")
        conv = importlib.import_module("fla.modules.conv")
        return FlaBackendOperators(
            gated_delta_chunk=gated_delta.chunk_gated_delta_rule,
            kda_chunk=kda.chunk_kda,
            causal_conv1d=conv.causal_conv1d,
            version=version,
        )
    except (ImportError, AttributeError) as error:
        raise FlaBackendError("required FLA 0.4.2 operators are unavailable") from error


def packed_cu_seqlens(segment_ids: torch.Tensor) -> torch.Tensor:
    """Return varlen offsets with an unconditional boundary between batch rows."""

    if (
        segment_ids.ndim != 2
        or segment_ids.dtype != torch.long
        or not segment_ids.shape[0]
        or not segment_ids.shape[1]
    ):
        raise FlaBackendError("packed segment identities differ")
    batch, sequence = segment_ids.shape
    offsets = [0]
    for row_index in range(batch):
        row = segment_ids[row_index]
        changes = torch.nonzero(row[1:] != row[:-1], as_tuple=False).flatten()
        row_start = row_index * sequence
        offsets.extend(row_start + int(index.item()) + 1 for index in changes)
        offsets.append((row_index + 1) * sequence)
    if any(left >= right for left, right in pairwise(offsets)):
        raise FlaBackendError("packed segment offsets are not increasing")
    return torch.tensor(offsets, dtype=torch.long, device=segment_ids.device)


def _flatten_packed(
    value: torch.Tensor, segment_ids: torch.Tensor | None
) -> torch.Tensor:
    if segment_ids is None:
        return value
    if value.shape[:2] != segment_ids.shape:
        raise FlaBackendError("packed tensor geometry differs")
    return value.reshape(1, value.shape[0] * value.shape[1], *value.shape[2:])


def fla_causal_conv1d(
    value: torch.Tensor,
    weight: torch.Tensor,
    segment_ids: torch.Tensor | None,
    *,
    operators: FlaBackendOperators | None = None,
) -> torch.Tensor:
    """Apply Sai's depthwise causal convolution and SiLU with exact resets."""

    if value.ndim != 3 or weight.ndim != 3 or weight.shape[1] != 1:
        raise FlaBackendError("causal convolution geometry differs")
    if weight.shape[0] != value.shape[-1]:
        raise FlaBackendError("causal convolution channels differ")
    loaded = operators or load_fla_backend_operators()
    cu_seqlens = None if segment_ids is None else packed_cu_seqlens(segment_ids)
    flattened = _flatten_packed(value, segment_ids)
    output = loaded.causal_conv1d(
        x=flattened,
        weight=weight[:, 0, :],
        bias=None,
        activation="silu",
        cu_seqlens=cu_seqlens,
        output_final_state=False,
    )
    if isinstance(output, tuple):
        if not output or not isinstance(output[0], torch.Tensor):
            raise FlaBackendError("FLA convolution output differs")
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.shape != flattened.shape:
        raise FlaBackendError("FLA convolution output geometry differs")
    return output.reshape_as(value)


def fla_delta_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    segment_ids: torch.Tensor | None,
    *,
    channel_wise_decay: bool,
    operators: FlaBackendOperators | None = None,
) -> torch.Tensor:
    """Map Sai's materialized alpha/beta tensors to FLA's log-gate API."""

    if query.shape != key.shape or query.ndim != 4 or value.ndim != 4:
        raise FlaBackendError("delta recurrence geometry differs")
    if value.shape[:3] != query.shape[:3]:
        raise FlaBackendError("delta recurrence heads differ")
    expected_alpha = (*query.shape[:3], query.shape[-1] if channel_wise_decay else 1)
    if alpha.shape != expected_alpha or beta.shape != (*query.shape[:3], 1):
        raise FlaBackendError("delta gate geometry differs")
    if not bool((alpha > 0).all().item()) or not bool((alpha <= 1).all().item()):
        raise FlaBackendError("delta decay is outside (0, 1]")

    loaded = operators or load_fla_backend_operators()
    cu_seqlens = None if segment_ids is None else packed_cu_seqlens(segment_ids)
    q = F.normalize(query.float(), dim=-1).to(query.dtype)
    k = F.normalize(key.float(), dim=-1).to(key.dtype)
    log_decay = alpha.float().log()
    if not channel_wise_decay:
        log_decay = log_decay.squeeze(-1)
    mapped = {
        "q": _flatten_packed(q, segment_ids),
        "k": _flatten_packed(k, segment_ids),
        "v": _flatten_packed(value, segment_ids),
        "g": _flatten_packed(log_decay, segment_ids),
        "beta": _flatten_packed(beta.squeeze(-1), segment_ids),
        "scale": 1.0,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": False,
        "cu_seqlens": cu_seqlens,
    }
    if channel_wise_decay:
        output = loaded.kda_chunk(
            **mapped,
            use_gate_in_kernel=False,
            safe_gate=False,
            disable_recompute=False,
            transpose_state_layout=False,
        )
    else:
        output = loaded.gated_delta_chunk(**mapped, transpose_state_layout=False)
    if (
        not isinstance(output, tuple)
        or not output
        or not isinstance(output[0], torch.Tensor)
    ):
        raise FlaBackendError("FLA delta output differs")
    recurrence = output[0]
    expected = _flatten_packed(value, segment_ids).shape
    if recurrence.shape != expected:
        raise FlaBackendError("FLA delta output geometry differs")
    return recurrence.reshape_as(value)
