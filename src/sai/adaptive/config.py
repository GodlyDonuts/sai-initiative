"""Exact no-training geometry and accounting for Sai's latent workspace."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


class WorkspaceConfigError(ValueError):
    """A workspace dimension or accounting request is invalid."""


@dataclass(frozen=True)
class WorkspaceConfig:
    hidden_size: int
    workspace_size: int
    num_slots: int = 16
    num_heads: int = 6
    reactor_layers: int = 4
    reactor_intermediate_size: int = 1536
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        fields = (
            "hidden_size",
            "workspace_size",
            "num_slots",
            "num_heads",
            "reactor_layers",
            "reactor_intermediate_size",
        )
        if any(
            isinstance(getattr(self, field), bool)
            or not isinstance(getattr(self, field), int)
            or getattr(self, field) <= 0
            for field in fields
        ):
            raise WorkspaceConfigError("workspace dimensions must be positive integers")
        if self.workspace_size % self.num_heads:
            raise WorkspaceConfigError("workspace heads must divide its width exactly")
        if (
            not isinstance(self.rms_norm_eps, float)
            or not math.isfinite(self.rms_norm_eps)
            or not self.rms_norm_eps > 0
        ):
            raise WorkspaceConfigError("workspace RMSNorm epsilon must be positive")

    @property
    def head_dim(self) -> int:
        return self.workspace_size // self.num_heads

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_workspace_config(hidden_size: int) -> WorkspaceConfig:
    """Return the frozen 16-slot Gate-0 geometry for a selected 300M base."""

    if hidden_size != 768:
        raise WorkspaceConfigError("Gate-0 workspace requires the 300M hidden width")
    return WorkspaceConfig(
        hidden_size=hidden_size,
        workspace_size=384,
        num_slots=16,
        num_heads=6,
        reactor_layers=4,
        reactor_intermediate_size=1536,
    )


def workspace_parameter_ledger(config: WorkspaceConfig) -> dict[str, int]:
    hidden = config.hidden_size
    width = config.workspace_size
    slots = config.num_slots
    intermediate = config.reactor_intermediate_size

    learned_slots = slots * width
    compiler = 2 * width**2 + 2 * hidden * width + hidden + width
    reactor_per_layer = 4 * width**2 + 3 * width * intermediate + 2 * width
    reactor = config.reactor_layers * reactor_per_layer
    reader = 2 * hidden * width + 2 * width**2 + hidden + width
    total = learned_slots + compiler + reactor + reader
    return {
        "learned_slots": learned_slots,
        "compiler": compiler,
        "reactor_per_layer": reactor_per_layer,
        "reactor": reactor,
        "reader_zero_initialized_output_included": reader,
        "total": total,
    }


def workspace_forward_flop_ledger(
    config: WorkspaceConfig, sequence_length: int, iterations: int
) -> dict[str, int | str]:
    """Count workspace matmul/attention FLOPs for one decoded position."""

    for value, field in (
        (sequence_length, "sequence length"),
        (iterations, "iterations"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise WorkspaceConfigError(f"{field} must be a positive integer")

    hidden = config.hidden_size
    width = config.workspace_size
    slots = config.num_slots
    intermediate = config.reactor_intermediate_size

    compiler_projections = 4 * sequence_length * hidden * width + 4 * slots * width**2
    compiler_attention = 4 * slots * sequence_length * width
    compiler = compiler_projections + compiler_attention
    reactor_per_layer = (
        8 * slots * width**2 + 4 * slots**2 * width + 6 * slots * width * intermediate
    )
    reactor_per_iteration = config.reactor_layers * reactor_per_layer
    reader = 4 * hidden * width + 4 * slots * width**2 + 4 * slots * width
    forced_slow_increment = compiler + iterations * reactor_per_iteration + reader
    return {
        "convention": "workspace_matmul_attention_one_multiply_add_equals_two",
        "scope": "one_next_token_decision_excluding_selected_base",
        "sequence_length": sequence_length,
        "iterations": iterations,
        "forced_fast_increment": 0,
        "compiler": compiler,
        "reactor_per_layer": reactor_per_layer,
        "reactor_per_iteration": reactor_per_iteration,
        "reader": reader,
        "forced_slow_increment": forced_slow_increment,
    }


def workspace_activation_ledger(
    config: WorkspaceConfig, sequence_length: int, *, bytes_per_element: int = 2
) -> dict[str, int | str]:
    """Count analytical incremental tensor geometry, excluding the base model."""

    for value, field in (
        (sequence_length, "sequence length"),
        (bytes_per_element, "bytes per element"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise WorkspaceConfigError(f"{field} must be a positive integer")

    hidden = config.hidden_size
    width = config.workspace_size
    slots = config.num_slots
    heads = config.num_heads
    intermediate = config.reactor_intermediate_size

    stages = {
        "compiler": 2 * sequence_length * width
        + 2 * slots * width
        + heads * slots * sequence_length,
        "reactor_attention": 6 * slots * width + heads * slots**2,
        "reactor_feed_forward": 3 * slots * width + 2 * slots * intermediate,
        "reader": width + 2 * slots * width + heads * slots + 2 * hidden,
    }
    maximum = max(stages.values())
    return {
        "convention": (
            "analytical_incremental_workspace_tensor_geometry_excluding_"
            "backbone_allocator_and_autograd"
        ),
        "sequence_length": sequence_length,
        "bytes_per_element": bytes_per_element,
        "stage_elements": stages,
        "maximum_stage_elements": maximum,
        "maximum_stage_bytes": maximum * bytes_per_element,
        "persistent_workspace_state_elements": slots * width,
        "persistent_workspace_state_bytes": slots * width * bytes_per_element,
    }
