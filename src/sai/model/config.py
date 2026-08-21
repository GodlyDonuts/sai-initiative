"""Exact configurations and analytical parameter ledgers for Sai candidates."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Literal

MixerFamily = Literal["gated_gqa", "gdn_hybrid", "kda_mla_hybrid"]


class SaiConfigError(ValueError):
    """A model configuration is internally inconsistent or outside the tournament."""


@dataclass(frozen=True)
class SaiModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    mixer_family: MixerFamily
    linear_conv_kernel: int = 4
    mla_kv_rank: int = 128
    mla_qk_head_dim: int = 64
    mla_value_head_dim: int = 64
    rms_norm_eps: float = 1e-6
    rope_fraction: float = 0.5
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    dropout: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "linear_conv_kernel",
            "mla_kv_rank",
            "mla_qk_head_dim",
            "mla_value_head_dim",
        )
        if any(
            isinstance(getattr(self, field), bool)
            or not isinstance(getattr(self, field), int)
            or getattr(self, field) <= 0
            for field in integer_fields
        ):
            raise SaiConfigError("model dimensions must be positive integers")
        if self.mixer_family not in {"gated_gqa", "gdn_hybrid", "kda_mla_hybrid"}:
            raise SaiConfigError("mixer family is outside the frozen tournament")
        if self.num_attention_heads % self.num_key_value_heads:
            raise SaiConfigError("attention heads must divide key/value heads exactly")
        if self.head_dim % 2:
            raise SaiConfigError("attention head dimension must be even")
        rotary_dimensions = int(self.head_dim * self.rope_fraction)
        if not 0 <= self.rope_fraction <= 1 or rotary_dimensions % 2:
            raise SaiConfigError("partial RoPE dimensions must be even")
        if not self.tie_word_embeddings or self.attention_bias or self.dropout != 0:
            raise SaiConfigError("Sai requires tied, bias-free, zero-dropout mechanics")
        if not math.isfinite(self.rms_norm_eps) or self.rms_norm_eps <= 0:
            raise SaiConfigError("RMSNorm epsilon must be positive")

    @property
    def attention_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def layer_types(self) -> list[str]:
        if self.mixer_family == "gated_gqa":
            return ["gated_gqa"] * self.num_hidden_layers
        linear, global_mixer = {
            "gdn_hybrid": ("gated_deltanet", "gated_gqa"),
            "kda_mla_hybrid": ("kda", "gated_mla"),
        }[self.mixer_family]
        return [
            global_mixer if (index + 1) % 4 == 0 else linear
            for index in range(self.num_hidden_layers)
        ]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _gated_gqa_parameters(config: SaiModelConfig) -> int:
    hidden = config.hidden_size
    attention = config.attention_width
    key_value = config.key_value_width
    return hidden * (3 * attention + 2 * key_value) + 2 * config.head_dim


def _gated_deltanet_parameters(config: SaiModelConfig) -> int:
    hidden = config.hidden_size
    attention = config.attention_width
    heads = config.num_attention_heads
    return (
        4 * hidden * attention
        + 2 * hidden * heads
        + hidden * config.head_dim
        + config.head_dim * attention
        + 3 * attention * config.linear_conv_kernel
        + config.head_dim
    )


def _kda_parameters(config: SaiModelConfig) -> int:
    hidden = config.hidden_size
    attention = config.attention_width
    head_dim = config.head_dim
    heads = config.num_attention_heads
    return (
        4 * hidden * attention
        + 2 * (hidden * head_dim + head_dim * attention)
        + hidden * heads
        + 3 * attention * config.linear_conv_kernel
        + attention
        + heads
        + head_dim
    )


def _gated_mla_parameters(config: SaiModelConfig) -> int:
    hidden = config.hidden_size
    heads = config.num_attention_heads
    q_width = heads * config.mla_qk_head_dim
    value_width = heads * config.mla_value_head_dim
    expanded_kv = heads * (config.mla_qk_head_dim + config.mla_value_head_dim)
    return (
        hidden * q_width
        + hidden * config.mla_kv_rank
        + config.mla_kv_rank * expanded_kv
        + hidden * value_width
        + value_width * hidden
        + config.mla_kv_rank
    )


def mixer_parameters(config: SaiModelConfig, layer_type: str) -> int:
    functions = {
        "gated_gqa": _gated_gqa_parameters,
        "gated_deltanet": _gated_deltanet_parameters,
        "kda": _kda_parameters,
        "gated_mla": _gated_mla_parameters,
    }
    try:
        return functions[layer_type](config)
    except KeyError as error:
        raise SaiConfigError(f"unknown layer type: {layer_type}") from error


def parameter_ledger(config: SaiModelConfig) -> dict[str, int]:
    embedding = config.vocab_size * config.hidden_size
    mixers = sum(mixer_parameters(config, layer) for layer in config.layer_types())
    feed_forward = (
        3 * config.hidden_size * config.intermediate_size * config.num_hidden_layers
    )
    layer_norms = 2 * config.hidden_size * config.num_hidden_layers
    final_norm = config.hidden_size
    total = embedding + mixers + feed_forward + layer_norms + final_norm
    return {
        "tied_embedding": embedding,
        "mixers": mixers,
        "feed_forward": feed_forward,
        "layer_norms": layer_norms,
        "final_norm": final_norm,
        "total": total,
    }


def _mixer_forward_flops(
    config: SaiModelConfig, layer_type: str, sequence_length: int
) -> int:
    """Matmul/conv/recurrent FLOPs, counting one multiply-add as two FLOPs."""

    hidden = config.hidden_size
    heads = config.num_attention_heads
    head_dim = config.head_dim
    attention = config.attention_width
    key_value = config.key_value_width
    if layer_type == "gated_gqa":
        projection_weights = hidden * (3 * attention + 2 * key_value)
        attention_flops = 4 * heads * sequence_length**2 * head_dim
        return 2 * sequence_length * projection_weights + attention_flops
    if layer_type == "gated_deltanet":
        projection_weights = (
            4 * hidden * attention
            + 2 * hidden * heads
            + hidden * head_dim
            + head_dim * attention
        )
        convolution_flops = (
            2 * sequence_length * 3 * attention * config.linear_conv_kernel
        )
        recurrent_flops = sequence_length * heads * (7 * head_dim**2 + 8 * head_dim)
        return (
            2 * sequence_length * projection_weights
            + convolution_flops
            + recurrent_flops
        )
    if layer_type == "kda":
        projection_weights = (
            4 * hidden * attention
            + 2 * (hidden * head_dim + head_dim * attention)
            + hidden * heads
        )
        convolution_flops = (
            2 * sequence_length * 3 * attention * config.linear_conv_kernel
        )
        recurrent_flops = sequence_length * heads * (7 * head_dim**2 + 8 * head_dim)
        return (
            2 * sequence_length * projection_weights
            + convolution_flops
            + recurrent_flops
        )
    if layer_type == "gated_mla":
        projection_weights = _gated_mla_parameters(config) - config.mla_kv_rank
        attention_flops = (
            2
            * heads
            * sequence_length**2
            * (config.mla_qk_head_dim + config.mla_value_head_dim)
        )
        return 2 * sequence_length * projection_weights + attention_flops
    raise SaiConfigError(f"unknown layer type: {layer_type}")


def forward_flop_ledger(
    config: SaiModelConfig, sequence_length: int
) -> dict[str, int | str]:
    """Deterministic model FLOP convention for iso-FLOP planning.

    The ledger includes dense projections, depthwise convolutions, recurrent
    state updates, quadratic attention, FFN matmuls, and tied-output logits.
    Elementwise nonlinearities, normalization, embedding lookup, and loss are
    excluded and must be reported separately by production profilers.
    """

    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 0
    ):
        raise SaiConfigError("sequence length must be a positive integer")
    mixers = sum(
        _mixer_forward_flops(config, layer, sequence_length)
        for layer in config.layer_types()
    )
    feed_forward = (
        2
        * sequence_length
        * 3
        * config.hidden_size
        * config.intermediate_size
        * config.num_hidden_layers
    )
    tied_output_logits = 2 * sequence_length * config.hidden_size * config.vocab_size
    forward = mixers + feed_forward + tied_output_logits
    return {
        "convention": "matmul_conv_recurrence_one_multiply_add_equals_two",
        "sequence_length": sequence_length,
        "mixers": mixers,
        "feed_forward": feed_forward,
        "tied_output_logits": tied_output_logits,
        "forward": forward,
        "forward_plus_backward_approximation": 3 * forward,
    }


@dataclass(frozen=True)
class ScaleTemplate:
    name: str
    target_parameters: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    mla_kv_rank: int
    mla_qk_head_dim: int
    mla_value_head_dim: int


SCALE_TEMPLATES = (
    ScaleTemplate("100m", 100_000_000, 512, 12, 8, 2, 64, 128, 64, 64),
    ScaleTemplate("300m", 300_000_000, 768, 24, 12, 3, 64, 192, 64, 64),
    ScaleTemplate("1b", 1_000_000_000, 1536, 24, 12, 3, 128, 384, 128, 128),
    ScaleTemplate("4b", 4_000_000_000, 2560, 40, 20, 5, 128, 512, 128, 128),
)


def fit_scale_geometry(
    template: ScaleTemplate,
    mixer_family: MixerFamily,
    vocab_size: int,
    *,
    multiple: int = 64,
) -> SaiModelConfig:
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
    ):
        raise SaiConfigError("vocabulary size must be a positive integer")
    if isinstance(multiple, bool) or not isinstance(multiple, int) or multiple <= 0:
        raise SaiConfigError("intermediate-size multiple must be positive")
    prototype = SaiModelConfig(
        vocab_size=vocab_size,
        hidden_size=template.hidden_size,
        intermediate_size=multiple,
        num_hidden_layers=template.num_hidden_layers,
        num_attention_heads=template.num_attention_heads,
        num_key_value_heads=template.num_key_value_heads,
        head_dim=template.head_dim,
        mixer_family=mixer_family,
        mla_kv_rank=template.mla_kv_rank,
        mla_qk_head_dim=template.mla_qk_head_dim,
        mla_value_head_dim=template.mla_value_head_dim,
    )
    prototype_ledger = parameter_ledger(prototype)
    slope = 3 * template.hidden_size * template.num_hidden_layers
    fixed = prototype_ledger["total"] - slope * multiple
    raw_intermediate = (template.target_parameters - fixed) / slope
    rounded = max(multiple, multiple * round(raw_intermediate / multiple))
    return replace(prototype, intermediate_size=rounded)


def frozen_scale_geometries(vocab_size: int = 48_000) -> list[dict[str, object]]:
    geometries = []
    for template in SCALE_TEMPLATES:
        for family in ("gated_gqa", "gdn_hybrid", "kda_mla_hybrid"):
            config = fit_scale_geometry(template, family, vocab_size)
            ledger = parameter_ledger(config)
            geometries.append(
                {
                    "scale": template.name,
                    "target_parameters": template.target_parameters,
                    "mixer_family": family,
                    "config": config.as_dict(),
                    "parameter_ledger": ledger,
                    "flop_ledger_2048": forward_flop_ledger(config, 2048),
                    "relative_error": (ledger["total"] - template.target_parameters)
                    / template.target_parameters,
                }
            )
    return geometries
