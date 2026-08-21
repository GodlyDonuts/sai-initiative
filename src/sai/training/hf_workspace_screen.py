"""Train one matched latent-workspace arm on the exact Qwen3.5-0.8B parent.

This is a sub-4B development screen.  The parent is immutable, every target is
scored without cross-document context, and the recurrent treatment differs
from the reset-state control only in whether reactor state carries between
otherwise identical iterations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from sai.adaptive.config import (
    WorkspaceConfig,
    workspace_forward_flop_ledger,
    workspace_parameter_ledger,
)
from sai.adaptive.hf_workspace import FrozenHFWorkspaceSystem, WorkspaceStateMode
from sai.adaptive.reference import LatentWorkspace
from sai.data.hf_model_snapshot import validate_snapshot
from sai.data.token_stream import canonical_sha256, validate_frozen_stream
from sai.evaluation.hf_parent import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_VOCAB_SIZE,
    load_text_parent,
    validate_mechanics_receipt,
)
from sai.training.checkpoint import (
    CheckpointBindings,
    TrainingCounters,
    load_mechanics_checkpoint,
    save_mechanics_checkpoint,
)
from sai.training.runner import (
    TrainingRunConfig,
    build_adamw,
    learning_rate_multiplier,
    tensorize_stream_batch,
)
from sai.training.stream import IGNORE_TARGET, ReceiptBoundTokenStream, TrainingBatch

SCHEMA = "sai-qwen35-0p8b-matched-workspace-screen-v1"
SEED = 2_026_082_108
SEQUENCE_LENGTH = 2_048
INPUT_SEQUENCE_LENGTH = SEQUENCE_LENGTH - 1
TRAINING_PREFIXES = (256, 61_035)
POSITIONS = (255, 511, 767, 1023, 1279, 1535, 1791, 2046)
ITERATIONS = 2
SEQUENCES_PER_UPDATE = 32
KL_COEFFICIENT = 1.0
WORKSPACE_CONFIG = WorkspaceConfig(
    hidden_size=1024,
    workspace_size=512,
    num_slots=16,
    num_heads=8,
    reactor_layers=4,
    reactor_intermediate_size=2048,
)
EXPECTED_WORKSPACE_PARAMETERS = 19_938_304


class HFWorkspaceScreenError(RuntimeError):
    """A frozen input, matched objective, or execution invariant differs."""


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HFWorkspaceScreenError(f"{field} must be a lowercase SHA256")
    return value


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise HFWorkspaceScreenError("bound artifact is missing or unsafe")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        header = json.dumps(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _tensor_versions(module: torch.nn.Module) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, value._version)
        for name, value in sorted(module.state_dict(keep_vars=True).items())
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if (
        not path.name
        or path.name in {".", ".."}
        or not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.exists()
        or path.is_symlink()
    ):
        raise HFWorkspaceScreenError("screen output target differs")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _segment_start(segment_ids: torch.Tensor, position: int) -> int:
    """Return the first token of the final contiguous segment at ``position``."""

    if (
        segment_ids.ndim != 1
        or segment_ids.dtype != torch.long
        or isinstance(position, bool)
        or not isinstance(position, int)
        or not 0 <= position < segment_ids.numel()
    ):
        raise HFWorkspaceScreenError("segment target geometry differs")
    value = segment_ids[position]
    matches = segment_ids[: position + 1].eq(value)
    indices = matches.nonzero(as_tuple=False).flatten()
    if indices.numel() <= 0:
        raise HFWorkspaceScreenError("target segment is empty")
    start = int(indices[0].item())
    if not bool(matches[start:].all().item()):
        raise HFWorkspaceScreenError("target segment is not contiguous")
    return start


def selected_target_count(batch: TrainingBatch) -> int:
    """Count admitted fixed probe targets before allocating the parent model."""

    if not isinstance(batch, TrainingBatch) or len(batch.y) != 1:
        raise HFWorkspaceScreenError("workspace screen requires micro-batch one")
    if len(batch.y[0]) != INPUT_SEQUENCE_LENGTH:
        raise HFWorkspaceScreenError("workspace screen sequence length differs")
    admitted = 0
    for position in POSITIONS:
        target_present = batch.y[0][position] != IGNORE_TARGET
        if target_present != batch.loss_mask[0][position]:
            raise HFWorkspaceScreenError("workspace target mask differs")
        admitted += target_present
    return admitted


def matched_objective_sum(
    system: FrozenHFWorkspaceSystem,
    batch: TrainingBatch,
    *,
    state_mode: WorkspaceStateMode,
    device: torch.device | str,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Return CE+parent-KL summed over fixed, document-safe probe targets."""

    tensor_batch = tensorize_stream_batch(batch, device=device)
    if tensor_batch.input_ids.shape != (1, INPUT_SEQUENCE_LENGTH):
        raise HFWorkspaceScreenError("workspace tensor batch geometry differs")
    admitted = selected_target_count(batch)
    if admitted <= 0:
        raise HFWorkspaceScreenError("workspace sequence has no admitted probe target")

    objective = torch.zeros((), dtype=torch.float32, device=device)
    cross_entropy_sum = 0.0
    kl_sum = 0.0
    observed = 0
    for position in POSITIONS:
        target = tensor_batch.target_ids[0, position]
        if int(target.item()) == IGNORE_TARGET:
            continue
        start = _segment_start(tensor_batch.segment_ids[0], position)
        local_ids = tensor_batch.input_ids[:, start : position + 1]
        local_attention = torch.ones_like(local_ids)
        local_segments = torch.zeros_like(local_ids)
        hidden = system.parent_hidden(local_ids, local_attention)
        candidate_logits, parent_logits = system.logits_at(
            hidden,
            local_segments,
            position=hidden.shape[1] - 1,
            iterations=ITERATIONS,
            state_mode=state_mode,
        )
        candidate_fp32 = candidate_logits.float()
        parent_fp32 = parent_logits.float()
        cross_entropy = F.cross_entropy(
            candidate_fp32, target.reshape(1), reduction="sum"
        )
        parent_probability = F.softmax(parent_fp32, dim=-1)
        kl = F.kl_div(
            F.log_softmax(candidate_fp32, dim=-1),
            parent_probability,
            reduction="sum",
        )
        if not bool(torch.isfinite(cross_entropy).item()) or not bool(
            torch.isfinite(kl).item()
        ):
            raise HFWorkspaceScreenError("workspace objective is nonfinite")
        objective = objective + cross_entropy + KL_COEFFICIENT * kl
        cross_entropy_sum += float(cross_entropy.detach())
        kl_sum += float(kl.detach())
        observed += 1
    if observed != admitted:
        raise HFWorkspaceScreenError("workspace admitted-target accounting differs")
    return objective, {
        "targets": observed,
        "cross_entropy_sum": cross_entropy_sum,
        "parent_kl_sum": kl_sum,
        "objective_sum": cross_entropy_sum + KL_COEFFICIENT * kl_sum,
    }


def make_bindings(
    *,
    state_mode: WorkspaceStateMode,
    snapshot_tree_sha256: str,
    mechanics_file_sha256: str,
    stream_identity_sha256: str,
    source_manifest_sha256: str,
    training_sequences: int,
    training_utf8_bytes: int,
    optimizer: TrainingRunConfig,
    code_sha256: str,
    environment_sha256: str,
) -> tuple[CheckpointBindings, dict[str, Any]]:
    """Derive a matched run identity from every scientific input."""

    if state_mode not in {"recurrent", "reset_average"}:
        raise HFWorkspaceScreenError("workspace state mode differs")
    for value, field in (
        (snapshot_tree_sha256, "snapshot tree"),
        (mechanics_file_sha256, "mechanics file"),
        (stream_identity_sha256, "stream identity"),
        (source_manifest_sha256, "source manifest"),
        (code_sha256, "code identity"),
        (environment_sha256, "environment identity"),
    ):
        _sha256(value, field)
    if training_sequences not in TRAINING_PREFIXES or training_utf8_bytes <= 0:
        raise HFWorkspaceScreenError("workspace training prefix differs")
    config = WORKSPACE_CONFIG.as_dict()
    config_sha256 = canonical_sha256(config)
    model_sha256 = canonical_sha256(
        {
            "parent_snapshot_tree_sha256": snapshot_tree_sha256,
            "workspace_config_sha256": config_sha256,
            "workspace_initialization_seed": SEED,
        }
    )
    specification = {
        "schema": SCHEMA,
        "evidence_class": "sub_4b_matched_development_screen",
        "parent": {
            "model": "Qwen/Qwen3.5-0.8B",
            "revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "snapshot_tree_sha256": snapshot_tree_sha256,
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "frozen": True,
        },
        "workspace_config": config,
        "workspace_config_sha256": config_sha256,
        "workspace_parameter_count": EXPECTED_WORKSPACE_PARAMETERS,
        "workspace_initialization_seed": SEED,
        "state_mode": state_mode,
        "matched_control_contract": {
            "treatment": "recurrent",
            "control": "reset_average",
            "only_changed_factor": "reactor_state_carry_between_iterations",
            "identical_parameters": True,
            "identical_compiler_reactor_reader_calls": True,
            "identical_modeled_workspace_flops": True,
        },
        "iterations": ITERATIONS,
        "probe_positions": list(POSITIONS),
        "objective": {
            "selected_target_cross_entropy": 1.0,
            "frozen_parent_kl": KL_COEFFICIENT,
            "cross_document_targets_masked": True,
            "cross_document_context_excluded": True,
        },
        "training_stream_identity_sha256": stream_identity_sha256,
        "training_source_manifest_sha256": source_manifest_sha256,
        "training_sequences": training_sequences,
        "training_utf8_bytes": training_utf8_bytes,
        "sequences_per_update": SEQUENCES_PER_UPDATE,
        "optimizer_steps": math.ceil(training_sequences / SEQUENCES_PER_UPDATE),
        "optimizer": asdict(optimizer),
        "mechanics_receipt_file_sha256": mechanics_file_sha256,
        "code_sha256": code_sha256,
        "environment_sha256": environment_sha256,
        "four_b_training_executed": False,
        "four_b_training_authorized_by_this_result": False,
    }
    run_sha256 = canonical_sha256(specification)
    specification["run_sha256"] = run_sha256
    return (
        CheckpointBindings(
            model_sha256=model_sha256,
            config_sha256=config_sha256,
            ordered_stream_identity_sha256=stream_identity_sha256,
            code_sha256=code_sha256,
            environment_sha256=environment_sha256,
            run_sha256=run_sha256,
        ),
        specification,
    )


def _validate_run_args(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise HFWorkspaceScreenError("completed screen output already exists")
    if args.training_sequences not in TRAINING_PREFIXES:
        raise HFWorkspaceScreenError("training prefix is not frozen")
    if args.state_mode not in {"recurrent", "reset_average"}:
        raise HFWorkspaceScreenError("workspace state mode differs")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or "H100" not in torch.cuda.get_device_name(0)
    ):
        raise HFWorkspaceScreenError("exactly one H100 BF16 GPU is required")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one receipt-bound recurrent or matched reset-state arm."""

    _validate_run_args(args)
    mechanics_file_sha256 = _sha256_file(args.mechanics_receipt)
    if mechanics_file_sha256 != _sha256(
        args.mechanics_receipt_sha256, "mechanics receipt file"
    ):
        raise HFWorkspaceScreenError("mechanics receipt file identity differs")
    mechanics = validate_mechanics_receipt(
        args.mechanics_receipt,
        expected_file_sha256=mechanics_file_sha256,
        model_root=args.model_root,
    )
    snapshot = validate_snapshot(args.model_root)
    report = validate_frozen_stream(args.train_stream, verify_sources=True)
    if (
        report["sequence_length"] != SEQUENCE_LENGTH
        or report["vocab_size"] != EXPECTED_VOCAB_SIZE
        or report["eos_token_id"] != 248_044
        or report["ordered_stream_identity_sha256"]
        != _sha256(args.train_identity, "training stream identity")
        or args.training_sequences > report["sequences"]
    ):
        raise HFWorkspaceScreenError("Qwen workspace training stream differs")
    training_utf8_bytes = report["prefix_utf8_bytes"].get(str(args.training_sequences))
    if (
        isinstance(training_utf8_bytes, bool)
        or not isinstance(training_utf8_bytes, int)
        or training_utf8_bytes <= 0
    ):
        raise HFWorkspaceScreenError("training UTF-8 prefix differs")

    optimizer_steps = math.ceil(args.training_sequences / SEQUENCES_PER_UPDATE)
    optimizer_config = TrainingRunConfig(
        optimizer_steps=optimizer_steps,
        learning_rate=3e-4,
        warmup_steps=min(100, optimizer_steps),
        minimum_learning_rate_ratio=0.1,
        weight_decay=0.1,
        gradient_clip_norm=1.0,
    )
    bindings, specification = make_bindings(
        state_mode=args.state_mode,
        snapshot_tree_sha256=snapshot["tree_sha256"],
        mechanics_file_sha256=mechanics_file_sha256,
        stream_identity_sha256=report["ordered_stream_identity_sha256"],
        source_manifest_sha256=report["source_manifest_sha256"],
        training_sequences=args.training_sequences,
        training_utf8_bytes=training_utf8_bytes,
        optimizer=optimizer_config,
        code_sha256=_sha256(args.code_sha256, "code identity"),
        environment_sha256=_sha256(args.environment_sha256, "environment identity"),
    )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    workspace = LatentWorkspace(WORKSPACE_CONFIG).to("cuda:0")
    if sum(value.numel() for value in workspace.parameters()) != (
        EXPECTED_WORKSPACE_PARAMETERS
    ):
        raise HFWorkspaceScreenError("workspace parameter count differs")
    initial_workspace_sha256 = _state_sha256(workspace)
    parent, tokenizer, runtime = load_text_parent(args.model_root)
    if runtime != mechanics["runtime"] or len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise HFWorkspaceScreenError("live parent differs from mechanics")
    system = FrozenHFWorkspaceSystem(parent, workspace)
    parent_versions = _tensor_versions(parent)
    optimizer = build_adamw(workspace, optimizer_config)

    resumed_from = None
    if args.resume:
        restored = load_mechanics_checkpoint(
            args.checkpoint,
            model=workspace,
            optimizer=optimizer,
            expected_bindings=bindings,
        )
        counters = restored.counters
        cursor = restored.cursor
        resumed_from = {
            "checkpoint_sha256": restored.checkpoint_sha256,
            "checkpoint_bytes": restored.checkpoint_bytes,
            "recovered_from_previous": restored.recovered_from_previous,
            "counters": counters.as_dict(),
        }
    else:
        manifest = args.checkpoint.with_name(f"{args.checkpoint.name}.manifest.json")
        if args.checkpoint.exists() or manifest.exists():
            raise HFWorkspaceScreenError("checkpoint already exists; use --resume")
        counters = TrainingCounters(0, 0, 0)
        cursor = None
    expected_sequences = min(
        counters.optimizer_steps * SEQUENCES_PER_UPDATE, args.training_sequences
    )
    if counters.sequences != expected_sequences:
        raise HFWorkspaceScreenError("checkpoint training position differs")
    if counters.optimizer_steps > optimizer_steps:
        raise HFWorkspaceScreenError("checkpoint exceeds optimizer budget")

    stream = ReceiptBoundTokenStream(
        args.train_stream,
        expected_ordered_stream_identity_sha256=report[
            "ordered_stream_identity_sha256"
        ],
        resume_cursor=cursor,
        verify_sources=True,
    )
    remaining = args.training_sequences - counters.sequences
    if stream.remaining_sequences < remaining:
        raise HFWorkspaceScreenError("training stream cannot satisfy run budget")

    losses: list[float] = []
    cross_entropies: list[float] = []
    parent_kls: list[float] = []
    gradient_norms: list[float] = []
    system.train()
    torch.cuda.reset_peak_memory_stats(0)
    for step in range(counters.optimizer_steps + 1, optimizer_steps + 1):
        update_sequences = min(
            SEQUENCES_PER_UPDATE,
            args.training_sequences - counters.sequences,
        )
        raw_batches = [stream.next_batch(1) for _ in range(update_sequences)]
        update_targets = sum(selected_target_count(batch) for batch in raw_batches)
        if update_targets <= 0:
            raise HFWorkspaceScreenError("optimizer update has no probe targets")
        multiplier = learning_rate_multiplier(
            step,
            total_steps=optimizer_steps,
            warmup_steps=optimizer_config.warmup_steps,
            minimum_ratio=optimizer_config.minimum_learning_rate_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = optimizer_config.learning_rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        objective_sum = cross_entropy_sum = parent_kl_sum = 0.0
        observed_targets = 0
        for raw_batch in raw_batches:
            if selected_target_count(raw_batch) == 0:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                objective, observation = matched_objective_sum(
                    system,
                    raw_batch,
                    state_mode=args.state_mode,
                    device="cuda:0",
                )
            (objective / update_targets).backward()
            objective_sum += observation["objective_sum"]
            cross_entropy_sum += observation["cross_entropy_sum"]
            parent_kl_sum += observation["parent_kl_sum"]
            observed_targets += observation["targets"]
        if observed_targets != update_targets:
            raise HFWorkspaceScreenError("optimizer target accounting differs")
        gradients = [
            value.grad for value in workspace.parameters() if value.requires_grad
        ]
        if not gradients or any(
            value is None or not bool(torch.isfinite(value).all().item())
            for value in gradients
        ):
            raise HFWorkspaceScreenError("workspace gradient is missing or nonfinite")
        if not any(
            bool(value.ne(0).any().item()) for value in gradients if value is not None
        ):
            raise HFWorkspaceScreenError("workspace update has no nonzero gradient")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            workspace.parameters(), optimizer_config.gradient_clip_norm
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise HFWorkspaceScreenError("workspace gradient norm is nonfinite")
        optimizer.step()
        counters = TrainingCounters(
            optimizer_steps=step,
            sequences=counters.sequences + update_sequences,
            targets=counters.targets + update_targets,
        )
        losses.append(objective_sum / update_targets)
        cross_entropies.append(cross_entropy_sum / update_targets)
        parent_kls.append(parent_kl_sum / update_targets)
        gradient_norms.append(float(gradient_norm.detach()))
        if step % args.checkpoint_interval == 0 or step == optimizer_steps:
            save_mechanics_checkpoint(
                args.checkpoint,
                model=workspace,
                optimizer=optimizer,
                bindings=bindings,
                counters=counters,
                cursor=stream.cursor,
            )

    if counters.sequences != args.training_sequences or counters.optimizer_steps != (
        optimizer_steps
    ):
        raise HFWorkspaceScreenError("terminal training accounting differs")
    if _tensor_versions(parent) != parent_versions:
        raise HFWorkspaceScreenError("frozen parent state changed during training")
    final_workspace_sha256 = _state_sha256(workspace)
    if final_workspace_sha256 == initial_workspace_sha256:
        raise HFWorkspaceScreenError("workspace did not change during training")
    torch.cuda.synchronize()
    manifest_path = args.checkpoint.with_name(f"{args.checkpoint.name}.manifest.json")
    checkpoint_manifest = json.loads(manifest_path.read_text())
    payload: dict[str, Any] = {
        **specification,
        "status": "complete",
        "mechanics_receipt_sha256": mechanics["receipt_sha256"],
        "parent_runtime": runtime,
        "workspace_initial_state_sha256": initial_workspace_sha256,
        "workspace_final_state_sha256": final_workspace_sha256,
        "workspace_parameter_ledger": workspace_parameter_ledger(WORKSPACE_CONFIG),
        "workspace_flop_ledger_per_probe": workspace_forward_flop_ledger(
            WORKSPACE_CONFIG, INPUT_SEQUENCE_LENGTH, ITERATIONS
        ),
        "counters": counters.as_dict(),
        "stream_cursor": stream.cursor.as_dict(),
        "terminal_process_observations": {
            "steps_in_this_process": len(losses),
            "first_objective": losses[0] if losses else None,
            "last_objective": losses[-1] if losses else None,
            "first_cross_entropy": cross_entropies[0] if cross_entropies else None,
            "last_cross_entropy": cross_entropies[-1] if cross_entropies else None,
            "first_parent_kl": parent_kls[0] if parent_kls else None,
            "last_parent_kl": parent_kls[-1] if parent_kls else None,
            "maximum_gradient_norm": max(gradient_norms) if gradient_norms else None,
        },
        "checkpoint": checkpoint_manifest["checkpoint"],
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(0),
        "parent_state_unchanged": True,
        "resumed_from": resumed_from,
        "architecture_improvement_demonstrated": False,
        "benchmark_evaluation_required": True,
        "claim_limit": (
            "Training completion only. This result is not a benchmark win, does not "
            "promote an architecture, and does not authorize or execute 4B training."
        ),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--mechanics-receipt", type=Path, required=True)
    parser.add_argument("--mechanics-receipt-sha256", required=True)
    parser.add_argument("--train-stream", type=Path, required=True)
    parser.add_argument("--train-identity", required=True)
    parser.add_argument(
        "--state-mode", choices=("recurrent", "reset_average"), required=True
    )
    parser.add_argument(
        "--training-sequences", type=int, choices=TRAINING_PREFIXES, required=True
    )
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        isinstance(args.checkpoint_interval, bool)
        or not isinstance(args.checkpoint_interval, int)
        or args.checkpoint_interval <= 0
    ):
        raise HFWorkspaceScreenError("checkpoint interval differs")
    payload = run(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "run_sha256": payload["run_sha256"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
