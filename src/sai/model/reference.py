"""Slow, auditable CPU reference modules for the Sai architecture tournament."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from sai.model.config import SaiModelConfig


class SaiReferenceError(RuntimeError):
    """Reference mixer inputs or state geometry are invalid."""


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(value.dtype)


class SwiGLU(nn.Module):
    def __init__(self, config: SaiModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        nn.init.normal_(self.weight, mean=0.0, std=kernel_size**-0.5)

    def _unsegmented(self, value: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = value.shape
        convolved = F.conv1d(
            value.transpose(1, 2),
            self.weight,
            padding=self.kernel_size - 1,
            groups=channels,
        )[..., :sequence]
        return convolved.transpose(1, 2).reshape(batch, sequence, channels)

    def forward(
        self, value: torch.Tensor, segment_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if segment_ids is None:
            return self._unsegmented(value)
        output = torch.zeros_like(value)
        for batch_index, row in enumerate(segment_ids.tolist()):
            start = 0
            for index in range(1, len(row) + 1):
                if index == len(row) or row[index] != row[start]:
                    output[batch_index : batch_index + 1, start:index] = (
                        self._unsegmented(
                            value[batch_index : batch_index + 1, start:index]
                        )
                    )
                    start = index
        return output


def causal_delta_recurrence(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference KDA/GDN delta-rule recurrence with FP32 recurrent state."""

    if query.shape != key.shape or query.ndim != 4 or value.ndim != 4:
        raise SaiReferenceError("query/key/value geometry differs")
    batch, sequence, heads, key_dim = query.shape
    if not batch or not sequence or not heads or not key_dim:
        raise SaiReferenceError("delta recurrence dimensions must be nonzero")
    if value.shape[:3] != (batch, sequence, heads):
        raise SaiReferenceError("value heads differ")
    value_dim = value.shape[-1]
    expected_alpha_prefix = (batch, sequence, heads)
    if alpha.shape[:3] != expected_alpha_prefix or alpha.shape[-1] not in {1, key_dim}:
        raise SaiReferenceError("alpha must be scalar- or channel-wise per head")
    if beta.shape != (batch, sequence, heads, 1):
        raise SaiReferenceError("beta must be scalar per head")
    expected_state = (batch, heads, key_dim, value_dim)
    if initial_state is not None and initial_state.shape != expected_state:
        raise SaiReferenceError("initial recurrent state differs")
    if reset_mask is not None and (
        reset_mask.shape != (batch, sequence) or reset_mask.dtype != torch.bool
    ):
        raise SaiReferenceError("reset mask must be boolean per token")

    state = (
        torch.zeros(expected_state, device=query.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float()
    )
    outputs = []
    q_float = F.normalize(query.float(), dim=-1)
    k_float = F.normalize(key.float(), dim=-1)
    for index in range(sequence):
        if reset_mask is not None:
            keep = (~reset_mask[:, index]).view(batch, 1, 1, 1)
            state = torch.where(keep, state, torch.zeros_like(state))
        q_t = q_float[:, index]
        k_t = k_float[:, index]
        v_t = value[:, index].float()
        decay = alpha[:, index].float()
        state = decay.unsqueeze(-1) * state
        prediction = torch.einsum("bhk,bhkv->bhv", k_t, state)
        error = beta[:, index].float() * (v_t - prediction)
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, error)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state))
    output = torch.stack(outputs, dim=1).to(value.dtype)
    return output, state


def _apply_partial_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    fraction: float,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = int(query.shape[-1] * fraction)
    if rotary_dim == 0:
        return query, key
    if position_ids is None:
        position_ids = torch.arange(query.shape[1], device=query.device).expand(
            query.shape[0], -1
        )
    inverse = 1.0 / (
        10_000
        ** (
            torch.arange(0, rotary_dim, 2, device=query.device, dtype=torch.float32)
            / rotary_dim
        )
    )
    angles = position_ids.float().unsqueeze(-1) * inverse[None, None, :]
    cosine = angles.cos().unsqueeze(2)
    sine = angles.sin().unsqueeze(2)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        rotated, remainder = value[..., :rotary_dim], value[..., rotary_dim:]
        even, odd = rotated[..., 0::2], rotated[..., 1::2]
        paired = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)
        return torch.cat((paired.to(value.dtype), remainder), dim=-1)

    return rotate(query), rotate(key)


def _segment_position_ids(segment_ids: torch.Tensor) -> torch.Tensor:
    positions = torch.zeros_like(segment_ids)
    for index in range(1, segment_ids.shape[1]):
        positions[:, index] = torch.where(
            segment_ids[:, index] == segment_ids[:, index - 1],
            positions[:, index - 1] + 1,
            0,
        )
    return positions


def _segment_attention_mask(
    segment_ids: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    sequence = segment_ids.shape[1]
    same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
    causal = torch.ones(
        sequence, sequence, device=segment_ids.device, dtype=torch.bool
    ).tril()
    allowed = same_segment & causal
    mask = torch.zeros(
        segment_ids.shape[0],
        1,
        sequence,
        sequence,
        device=segment_ids.device,
        dtype=dtype,
    )
    return mask.masked_fill(~allowed.unsqueeze(1), float("-inf"))


def _segment_reset_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    reset = torch.ones_like(segment_ids, dtype=torch.bool)
    reset[:, 1:] = segment_ids[:, 1:] != segment_ids[:, :-1]
    return reset


class GatedGQA(nn.Module):
    def __init__(self, config: SaiModelConfig) -> None:
        super().__init__()
        self.config = config
        attention_width = config.attention_width
        self.q_gate_proj = nn.Linear(
            config.hidden_size, 2 * attention_width, bias=False
        )
        self.k_proj = nn.Linear(config.hidden_size, config.key_value_width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.key_value_width, bias=False)
        self.o_proj = nn.Linear(attention_width, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self, hidden: torch.Tensor, segment_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        heads = self.config.num_attention_heads
        key_value_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim
        query, gate = self.q_gate_proj(hidden).chunk(2, dim=-1)
        query = self.q_norm(query.view(batch, sequence, heads, head_dim))
        key = self.k_norm(
            self.k_proj(hidden).view(batch, sequence, key_value_heads, head_dim)
        )
        value = self.v_proj(hidden).view(batch, sequence, key_value_heads, head_dim)
        repeats = heads // key_value_heads
        key = key.repeat_interleave(repeats, dim=2)
        value = value.repeat_interleave(repeats, dim=2)
        positions = None if segment_ids is None else _segment_position_ids(segment_ids)
        query, key = _apply_partial_rope(
            query, key, self.config.rope_fraction, positions
        )
        attention_mask = (
            None
            if segment_ids is None
            else _segment_attention_mask(segment_ids, query.dtype)
        )
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
        ).transpose(1, 2)
        output = output * torch.sigmoid(gate.view(batch, sequence, heads, head_dim))
        return self.o_proj(output.reshape(batch, sequence, -1))


class DeltaMixer(nn.Module):
    def __init__(self, config: SaiModelConfig, *, channel_wise_decay: bool) -> None:
        super().__init__()
        self.config = config
        self.channel_wise_decay = channel_wise_decay
        width = config.attention_width
        self.q_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, width, bias=False)
        self.q_conv = CausalDepthwiseConv1d(width, config.linear_conv_kernel)
        self.k_conv = CausalDepthwiseConv1d(width, config.linear_conv_kernel)
        self.v_conv = CausalDepthwiseConv1d(width, config.linear_conv_kernel)
        if channel_wise_decay:
            self.alpha_down = nn.Linear(config.hidden_size, config.head_dim, bias=False)
            self.alpha_up = nn.Linear(config.head_dim, width, bias=False)
            self.alpha_log_scale = nn.Parameter(torch.zeros(config.num_attention_heads))
            self.alpha_bias = nn.Parameter(torch.zeros(width))
        else:
            self.alpha_proj = nn.Linear(
                config.hidden_size, config.num_attention_heads, bias=False
            )
        self.beta_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads, bias=False
        )
        self.gate_down = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.gate_up = nn.Linear(config.head_dim, width, bias=False)
        self.output_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.o_proj = nn.Linear(width, config.hidden_size, bias=False)

    def forward(
        self, hidden: torch.Tensor, segment_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        heads, head_dim = self.config.num_attention_heads, self.config.head_dim
        q = F.silu(self.q_conv(self.q_proj(hidden), segment_ids)).view(
            batch, sequence, heads, head_dim
        )
        k = F.silu(self.k_conv(self.k_proj(hidden), segment_ids)).view(
            batch, sequence, heads, head_dim
        )
        v = F.silu(self.v_conv(self.v_proj(hidden), segment_ids)).view(
            batch, sequence, heads, head_dim
        )
        if self.channel_wise_decay:
            raw_alpha = self.alpha_up(self.alpha_down(hidden))
            raw_alpha = raw_alpha + self.alpha_bias
            scale = F.softplus(self.alpha_log_scale).view(1, 1, heads, 1)
            alpha = torch.exp(-scale * F.softplus(raw_alpha).view_as(q))
        else:
            alpha = torch.sigmoid(self.alpha_proj(hidden)).unsqueeze(-1)
        beta = torch.sigmoid(self.beta_proj(hidden)).unsqueeze(-1)
        reset_mask = None if segment_ids is None else _segment_reset_mask(segment_ids)
        output, _ = causal_delta_recurrence(q, k, v, alpha, beta, reset_mask=reset_mask)
        gate = self.gate_up(self.gate_down(hidden)).view_as(output)
        output = self.output_norm(output) * torch.sigmoid(gate)
        return self.o_proj(output.reshape(batch, sequence, -1))


class GatedMLA(nn.Module):
    def __init__(self, config: SaiModelConfig) -> None:
        super().__init__()
        self.config = config
        heads = config.num_attention_heads
        self.q_proj = nn.Linear(
            config.hidden_size, heads * config.mla_qk_head_dim, bias=False
        )
        self.kv_a_proj = nn.Linear(config.hidden_size, config.mla_kv_rank, bias=False)
        self.kv_a_norm = RMSNorm(config.mla_kv_rank, config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.mla_kv_rank,
            heads * (config.mla_qk_head_dim + config.mla_value_head_dim),
            bias=False,
        )
        self.gate_proj = nn.Linear(
            config.hidden_size, heads * config.mla_value_head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            heads * config.mla_value_head_dim, config.hidden_size, bias=False
        )

    def forward(
        self, hidden: torch.Tensor, segment_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch, sequence, _ = hidden.shape
        heads = self.config.num_attention_heads
        query = self.q_proj(hidden).view(
            batch, sequence, heads, self.config.mla_qk_head_dim
        )
        latent = self.kv_a_norm(self.kv_a_proj(hidden))
        key_value = self.kv_b_proj(latent).view(
            batch,
            sequence,
            heads,
            self.config.mla_qk_head_dim + self.config.mla_value_head_dim,
        )
        key, value = key_value.split(
            [self.config.mla_qk_head_dim, self.config.mla_value_head_dim], dim=-1
        )
        attention_mask = (
            None
            if segment_ids is None
            else _segment_attention_mask(segment_ids, query.dtype)
        )
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
            scale=self.config.mla_qk_head_dim**-0.5,
        ).transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(hidden)).view_as(output)
        return self.o_proj((output * gate).reshape(batch, sequence, -1))


class SaiBlock(nn.Module):
    def __init__(self, config: SaiModelConfig, layer_type: str) -> None:
        super().__init__()
        if layer_type == "gated_gqa":
            self.mixer = GatedGQA(config)
        elif layer_type == "gated_deltanet":
            self.mixer = DeltaMixer(config, channel_wise_decay=False)
        elif layer_type == "kda":
            self.mixer = DeltaMixer(config, channel_wise_decay=True)
        elif layer_type == "gated_mla":
            self.mixer = GatedMLA(config)
        else:
            raise SaiReferenceError(f"unknown reference layer type: {layer_type}")
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_mixer_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.feed_forward = SwiGLU(config)

    def forward(
        self, hidden: torch.Tensor, segment_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = hidden + self.mixer(self.input_norm(hidden), segment_ids)
        return hidden + self.feed_forward(self.post_mixer_norm(hidden))


class SaiCausalLM(nn.Module):
    def __init__(self, config: SaiModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            SaiBlock(config, layer_type) for layer_type in config.layer_types()
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    @property
    def lm_head_weight(self) -> nn.Parameter:
        return self.embed_tokens.weight

    @staticmethod
    def _validate_segment_ids(
        input_ids: torch.Tensor, segment_ids: torch.Tensor | None
    ) -> None:
        if segment_ids is None:
            return
        if (
            segment_ids.shape != input_ids.shape
            or segment_ids.dtype != torch.long
            or segment_ids.device != input_ids.device
        ):
            raise SaiReferenceError("segment_ids must match input_ids as a LongTensor")
        for row in segment_ids.tolist():
            seen: set[int] = set()
            previous = None
            for value in row:
                if value != previous:
                    if value in seen:
                        raise SaiReferenceError("segment identities must be contiguous")
                    seen.add(value)
                    previous = value

    def forward(
        self, input_ids: torch.Tensor, segment_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise SaiReferenceError("input_ids must be a rank-two LongTensor")
        self._validate_segment_ids(input_ids, segment_ids)
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, segment_ids)
        hidden = self.norm(hidden)
        return F.linear(hidden, self.lm_head_weight)


def exact_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
