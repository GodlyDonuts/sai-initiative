"""Matched recurrent and reset-state latent workspaces over a frozen HF parent."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from sai.adaptive.reference import LatentWorkspace, WorkspaceReferenceError

WorkspaceStateMode = Literal["recurrent", "reset_average"]


def matched_workspace_delta(
    workspace: LatentWorkspace,
    context: torch.Tensor,
    *,
    iterations: int,
    context_mask: torch.Tensor | None,
    state_mode: WorkspaceStateMode,
) -> torch.Tensor:
    """Execute matched-compute state propagation and return one hidden delta."""

    if (
        context.ndim != 3
        or context.shape[-1] != workspace.config.hidden_size
        or context.shape[0] <= 0
        or context.shape[1] <= 0
    ):
        raise WorkspaceReferenceError("HF workspace context geometry differs")
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
    ):
        raise WorkspaceReferenceError("HF workspace iterations must be positive")
    if state_mode not in {"recurrent", "reset_average"}:
        raise WorkspaceReferenceError("HF workspace state mode differs")
    if context_mask is not None and (
        context_mask.shape != context.shape[:2]
        or context_mask.dtype is not torch.bool
        or context_mask.device != context.device
        or not bool(context_mask.any(dim=1).all().item())
        or not bool(context_mask[:, -1].all().item())
        or bool((context_mask[:, :-1] & ~context_mask[:, 1:]).any().item())
    ):
        raise WorkspaceReferenceError("HF workspace mask is not a contiguous suffix")

    initial = workspace.compiler(context, context_mask)
    if state_mode == "recurrent":
        slots = initial
        for _ in range(iterations):
            for block in workspace.reactor:
                slots = block(slots, initial)
    else:
        branches = []
        for _ in range(iterations):
            slots = initial
            for block in workspace.reactor:
                slots = block(slots, initial)
            branches.append(slots)
        slots = torch.stack(branches).mean(dim=0)
    return workspace.reader(context[:, -1], slots)


class FrozenHFWorkspaceSystem(nn.Module):
    """Attach one trainable workspace without changing the parent fast path."""

    def __init__(self, parent: nn.Module, workspace: LatentWorkspace) -> None:
        super().__init__()
        language_model = getattr(parent, "model", None)
        lm_head = getattr(parent, "lm_head", None)
        hidden_size = getattr(getattr(parent, "config", None), "hidden_size", None)
        if (
            not isinstance(language_model, nn.Module)
            or not isinstance(lm_head, nn.Module)
            or hidden_size != workspace.config.hidden_size
        ):
            raise WorkspaceReferenceError("HF parent workspace interface differs")
        self.parent = parent
        self.parent.requires_grad_(False)
        self.parent.eval()
        self.workspace = workspace

    def train(self, mode: bool = True):
        super().train(mode)
        self.parent.eval()
        return self

    def parent_hidden(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run the frozen causal body once and return detached hidden states."""

        if (
            input_ids.ndim != 2
            or attention_mask.shape != input_ids.shape
            or input_ids.dtype is not torch.long
        ):
            raise WorkspaceReferenceError("HF parent token geometry differs")
        with torch.no_grad():
            output = self.parent.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden = getattr(output, "last_hidden_state", None)
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.shape[:2] != input_ids.shape
            or hidden.shape[-1] != self.workspace.config.hidden_size
            or hidden.requires_grad
        ):
            raise WorkspaceReferenceError("HF parent hidden-state geometry differs")
        return hidden

    def logits_at(
        self,
        hidden: torch.Tensor,
        segment_ids: torch.Tensor,
        *,
        position: int,
        iterations: int,
        state_mode: WorkspaceStateMode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return matched candidate and unchanged-parent logits at one position."""

        if (
            hidden.ndim != 3
            or segment_ids.shape != hidden.shape[:2]
            or isinstance(position, bool)
            or not isinstance(position, int)
            or not 0 <= position < hidden.shape[1]
        ):
            raise WorkspaceReferenceError("HF workspace target position differs")
        context = hidden[:, : position + 1]
        local_segments = segment_ids[:, : position + 1]
        context_mask = local_segments.eq(local_segments[:, -1:])
        delta = matched_workspace_delta(
            self.workspace,
            context,
            iterations=iterations,
            context_mask=context_mask,
            state_mode=state_mode,
        )
        base_hidden = hidden[:, position]
        parent_logits = self.parent.lm_head(base_hidden)
        candidate_logits = self.parent.lm_head(base_hidden + delta)
        if candidate_logits.shape != parent_logits.shape or candidate_logits.ndim != 2:
            raise WorkspaceReferenceError("HF workspace logit geometry differs")
        return candidate_logits, parent_logits.detach()

    def trainable_parameter_count(self) -> int:
        """Count only workspace parameters; the parent is immutable."""

        if any(parameter.requires_grad for parameter in self.parent.parameters()):
            raise WorkspaceReferenceError("HF parent is not frozen")
        return sum(
            parameter.numel()
            for parameter in self.workspace.parameters()
            if parameter.requires_grad
        )
