"""Train the matched Sai workspace factor on the frozen SmolLM3-3B parent."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sai.adaptive.config import (
    WorkspaceConfig,
    workspace_forward_flop_ledger,
    workspace_parameter_ledger,
)
from sai.adaptive.hf_workspace import FrozenHFWorkspaceSystem, WorkspaceStateMode
from sai.adaptive.reference import LatentWorkspace
from sai.data.external_hf_snapshot import validate_external_snapshot
from sai.data.token_stream import canonical_sha256, validate_frozen_stream
from sai.evaluation.hf_smol_parent import (
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_VOCAB_SIZE,
    SNAPSHOT_SPEC,
    load_smol_parent,
    validate_smol_mechanics_receipt,
)
from sai.training.checkpoint import (
    CheckpointBindings,
    TrainingCounters,
    load_mechanics_checkpoint,
    save_mechanics_checkpoint,
)
from sai.training.hf_workspace_screen import (
    INPUT_SEQUENCE_LENGTH,
    ITERATIONS,
    KL_COEFFICIENT,
    POSITIONS,
    SEQUENCE_LENGTH,
    SEQUENCES_PER_UPDATE,
    TRAINING_PREFIXES,
    HFWorkspaceScreenError,
    _atomic_json,
    _sha256,
    _sha256_file,
    _state_sha256,
    _tensor_versions,
    matched_objective_sum,
    selected_target_count,
)
from sai.training.runner import (
    TrainingRunConfig,
    build_adamw,
    learning_rate_multiplier,
)
from sai.training.stream import ReceiptBoundTokenStream

SCHEMA = "sai-smollm3-3b-matched-workspace-screen-v1"
SEED = 2_026_082_108
EXPECTED_EOS_TOKEN_ID = 128_012
WORKSPACE_CONFIG = WorkspaceConfig(
    hidden_size=2048,
    workspace_size=1024,
    num_slots=16,
    num_heads=16,
    reactor_layers=4,
    reactor_intermediate_size=4096,
)
EXPECTED_WORKSPACE_PARAMETERS = 79_722_496


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
    """Derive a Smol run identity with only state carry changed by arm."""

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
        "evidence_class": "sub_4b_cross_family_matched_development_screen",
        "parent": {
            "model": SNAPSHOT_SPEC.repository,
            "revision": SNAPSHOT_SPEC.revision,
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


def _validate_args(args: argparse.Namespace) -> None:
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
    """Execute one Smol recurrent treatment or reset-state control."""

    _validate_args(args)
    mechanics_file_sha256 = _sha256_file(args.mechanics_receipt)
    if mechanics_file_sha256 != _sha256(
        args.mechanics_receipt_sha256, "mechanics receipt file"
    ):
        raise HFWorkspaceScreenError("mechanics receipt file identity differs")
    mechanics = validate_smol_mechanics_receipt(
        args.mechanics_receipt,
        expected_file_sha256=mechanics_file_sha256,
        model_root=args.model_root,
        manifest_path=args.model_manifest,
        restoration_receipt_path=args.restoration_receipt,
    )
    snapshot = validate_external_snapshot(
        args.model_root,
        manifest_path=args.model_manifest,
        receipt_path=args.restoration_receipt,
        spec=SNAPSHOT_SPEC,
    )
    report = validate_frozen_stream(args.train_stream, verify_sources=True)
    if (
        report["sequence_length"] != SEQUENCE_LENGTH
        or report["vocab_size"] != EXPECTED_VOCAB_SIZE
        or report["eos_token_id"] != EXPECTED_EOS_TOKEN_ID
        or report["ordered_stream_identity_sha256"]
        != _sha256(args.train_identity, "training stream identity")
        or args.training_sequences > report["sequences"]
    ):
        raise HFWorkspaceScreenError("Smol workspace training stream differs")
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
    parent, tokenizer, runtime = load_smol_parent(
        args.model_root,
        manifest_path=args.model_manifest,
        restoration_receipt_path=args.restoration_receipt,
    )
    if runtime != mechanics["runtime"] or len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise HFWorkspaceScreenError("live Smol parent differs from mechanics")
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
    if stream.remaining_sequences < args.training_sequences - counters.sequences:
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
            objective_sum += float(observation["objective_sum"])
            cross_entropy_sum += float(observation["cross_entropy_sum"])
            parent_kl_sum += float(observation["parent_kl_sum"])
            observed_targets += int(observation["targets"])
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
            "Training completion only. This cross-family result is not a benchmark "
            "win and cannot authorize or execute 4B training."
        ),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--restoration-receipt", type=Path, required=True)
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
    if args.checkpoint_interval <= 0:
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
