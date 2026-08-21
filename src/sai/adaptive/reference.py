"""Auditable CPU reference for Sai's prospective private latent workspace."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from sai.adaptive.config import WorkspaceConfig
from sai.model.reference import RMSNorm, SaiCausalLM


class WorkspaceReferenceError(RuntimeError):
    """Workspace inputs, recurrence horizon, or mode are invalid."""


class WorkspaceCompiler(nn.Module):
    def __init__(self, config: WorkspaceConfig) -> None:
        super().__init__()
        self.config = config
        self.learned_slots = nn.Parameter(
            torch.empty(config.num_slots, config.workspace_size)
        )
        nn.init.normal_(self.learned_slots, std=config.workspace_size**-0.5)
        self.context_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.slot_norm = RMSNorm(config.workspace_size, config.rms_norm_eps)
        self.q_proj = nn.Linear(
            config.workspace_size, config.workspace_size, bias=False
        )
        self.k_proj = nn.Linear(config.hidden_size, config.workspace_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.workspace_size, bias=False)
        self.o_proj = nn.Linear(
            config.workspace_size, config.workspace_size, bias=False
        )

    def forward(
        self, context: torch.Tensor, context_mask: torch.Tensor | None
    ) -> torch.Tensor:
        batch, sequence, _ = context.shape
        heads, head_dim = self.config.num_heads, self.config.head_dim
        slots = self.learned_slots.unsqueeze(0).expand(batch, -1, -1)
        query = self.q_proj(self.slot_norm(slots)).view(
            batch, self.config.num_slots, heads, head_dim
        )
        normalized = self.context_norm(context)
        key = self.k_proj(normalized).view(batch, sequence, heads, head_dim)
        value = self.v_proj(normalized).view(batch, sequence, heads, head_dim)
        attention_mask = None
        if context_mask is not None:
            attention_mask = context_mask[:, None, None, :].expand(
                -1, heads, self.config.num_slots, -1
            )
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
        ).transpose(1, 2)
        return slots + self.o_proj(output.reshape(batch, self.config.num_slots, -1))


class WorkspaceReactorBlock(nn.Module):
    def __init__(self, config: WorkspaceConfig) -> None:
        super().__init__()
        self.config = config
        width = config.workspace_size
        self.input_norm = RMSNorm(width, config.rms_norm_eps)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.post_attention_norm = RMSNorm(width, config.rms_norm_eps)
        self.gate_proj = nn.Linear(width, config.reactor_intermediate_size, bias=False)
        self.up_proj = nn.Linear(width, config.reactor_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.reactor_intermediate_size, width, bias=False)

    def forward(self, slots: torch.Tensor, initial_slots: torch.Tensor) -> torch.Tensor:
        hidden = self.input_norm(slots + initial_slots)
        batch = hidden.shape[0]
        heads, head_dim = self.config.num_heads, self.config.head_dim

        def heads_view(value: torch.Tensor) -> torch.Tensor:
            return value.view(batch, self.config.num_slots, heads, head_dim).transpose(
                1, 2
            )

        output = F.scaled_dot_product_attention(
            heads_view(self.q_proj(hidden)),
            heads_view(self.k_proj(hidden)),
            heads_view(self.v_proj(hidden)),
        ).transpose(1, 2)
        slots = slots + self.o_proj(output.reshape_as(slots))
        hidden = self.post_attention_norm(slots)
        return slots + self.down_proj(
            F.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


class WorkspaceReader(nn.Module):
    def __init__(self, config: WorkspaceConfig) -> None:
        super().__init__()
        self.config = config
        self.context_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.slot_norm = RMSNorm(config.workspace_size, config.rms_norm_eps)
        self.q_proj = nn.Linear(config.hidden_size, config.workspace_size, bias=False)
        self.k_proj = nn.Linear(
            config.workspace_size, config.workspace_size, bias=False
        )
        self.v_proj = nn.Linear(
            config.workspace_size, config.workspace_size, bias=False
        )
        self.o_proj = nn.Linear(config.workspace_size, config.hidden_size, bias=False)
        nn.init.zeros_(self.o_proj.weight)

    def forward(self, final_hidden: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        batch = final_hidden.shape[0]
        heads, head_dim = self.config.num_heads, self.config.head_dim
        query = self.q_proj(self.context_norm(final_hidden)).view(
            batch, 1, heads, head_dim
        )
        normalized = self.slot_norm(slots)
        key = self.k_proj(normalized).view(
            batch, self.config.num_slots, heads, head_dim
        )
        value = self.v_proj(normalized).view(
            batch, self.config.num_slots, heads, head_dim
        )
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)
        return self.o_proj(output.reshape(batch, self.config.workspace_size))


@dataclass(frozen=True)
class WorkspaceDiagnostics:
    iterations: int
    initial_slot_rms: torch.Tensor
    final_slot_rms: torch.Tensor
    last_update_rms: torch.Tensor
    output_delta_rms: torch.Tensor


class LatentWorkspace(nn.Module):
    def __init__(self, config: WorkspaceConfig) -> None:
        super().__init__()
        self.config = config
        self.compiler = WorkspaceCompiler(config)
        self.reactor = nn.ModuleList(
            WorkspaceReactorBlock(config) for _ in range(config.reactor_layers)
        )
        self.reader = WorkspaceReader(config)

    def forward(
        self,
        context: torch.Tensor,
        *,
        iterations: int,
        context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, WorkspaceDiagnostics]:
        if context.ndim != 3 or context.shape[-1] != self.config.hidden_size:
            raise WorkspaceReferenceError("context hidden-state geometry differs")
        if context.shape[1] <= 0:
            raise WorkspaceReferenceError("workspace context must be nonempty")
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations <= 0
        ):
            raise WorkspaceReferenceError("workspace iterations must be positive")
        if context_mask is not None and (
            context_mask.shape != context.shape[:2]
            or context_mask.dtype != torch.bool
            or context_mask.device != context.device
            or not context_mask.any(dim=1).all()
        ):
            raise WorkspaceReferenceError("workspace context mask differs")

        initial = self.compiler(context, context_mask)
        slots = initial
        previous = slots
        for _ in range(iterations):
            previous = slots
            for block in self.reactor:
                slots = block(slots, initial)
        final_hidden = context[:, -1]
        delta = self.reader(final_hidden, slots)
        diagnostics = WorkspaceDiagnostics(
            iterations=iterations,
            initial_slot_rms=initial.float().square().mean(dim=(1, 2)).sqrt(),
            final_slot_rms=slots.float().square().mean(dim=(1, 2)).sqrt(),
            last_update_rms=(slots - previous).float().square().mean(dim=(1, 2)).sqrt(),
            output_delta_rms=delta.float().square().mean(dim=1).sqrt(),
        )
        return delta, diagnostics


class AdaptiveSaiCausalLM(nn.Module):
    """Wrap one frozen mixer candidate with an optional last-position workspace."""

    def __init__(self, base: SaiCausalLM, workspace: LatentWorkspace) -> None:
        super().__init__()
        if base.config.hidden_size != workspace.config.hidden_size:
            raise WorkspaceReferenceError("base and workspace hidden widths differ")
        self.base = base
        self.workspace = workspace

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
        *,
        mode: str = "fast",
        iterations: int = 1,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, WorkspaceDiagnostics | None]:
        if mode == "fast":
            logits = self.base(input_ids, segment_ids)
            return (logits, None) if return_diagnostics else logits
        if mode != "slow":
            raise WorkspaceReferenceError("adaptive mode must be fast or slow")

        hidden = self.base.hidden_states(input_ids, segment_ids)
        logits = self.base.project(hidden)
        context_mask = (
            None
            if segment_ids is None
            else segment_ids == segment_ids[:, -1:].expand_as(segment_ids)
        )
        delta, diagnostics = self.workspace(
            hidden, iterations=iterations, context_mask=context_mask
        )
        delta_logits = F.linear(delta, self.base.lm_head_weight)
        result = logits.clone()
        result[:, -1] = result[:, -1] + delta_logits
        return (result, diagnostics) if return_diagnostics else result
