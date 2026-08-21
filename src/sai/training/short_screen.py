"""Receipt-bound, single-H100 training for sub-4B Sai mechanics screens.

This runner deliberately produces only mechanics/screen evidence.  A lower
held-out NLL is not a public-benchmark win and never authorizes a 4B run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from sai.data.token_stream import canonical_sha256, validate_frozen_stream
from sai.model.config import SaiModelConfig, parameter_ledger
from sai.model.initialization import POLICY_SHA256, initialize_sai_model
from sai.model.reference import SaiCausalLM, exact_parameter_count
from sai.training.checkpoint import (
    CheckpointBindings,
    TrainingCounters,
    load_mechanics_checkpoint,
    save_mechanics_checkpoint,
)
from sai.training.evaluate import evaluate_nll
from sai.training.runner import (
    TrainingRunConfig,
    build_adamw,
    learning_rate_multiplier,
    tensorize_stream_batch,
)
from sai.training.stream import ReceiptBoundTokenStream

SCHEMA = "sai-sub-4b-short-screen-v1"
# The frozen "100m" comparison rows deliberately straddle 100M slightly.  The
# scale identity and exact checked ledger are authoritative; this upper bound
# only catches a maliciously relabeled larger geometry.
MAX_PARAMETERS = 101_000_000
FAMILIES = ("gated_gqa", "gdn_hybrid", "kda_mla_hybrid")


class ShortScreenError(RuntimeError):
    """A model, stream, device, resume, or bounded-run invariant differs."""


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ShortScreenError(f"{field} must be a lowercase SHA256")
    return value


def _regular_file(path: Path, field: str) -> Path:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ShortScreenError(f"{field} is missing or unsafe")
    return path


def load_bounded_config(path: Path, family: str) -> tuple[SaiModelConfig, dict]:
    """Load one immutable 100M row and enforce its narrow geometry envelope."""

    _regular_file(path, "geometry artifact")
    if family not in FAMILIES:
        raise ShortScreenError("mixer family differs")
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShortScreenError("geometry artifact is unreadable") from error
    rows = payload.get("geometries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ShortScreenError("geometry artifact differs")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("scale") == "100m"
        and row.get("mixer_family") == family
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("config"), dict):
        raise ShortScreenError("100M family geometry is not unique")
    row = matches[0]
    config = SaiModelConfig(**row["config"])
    if config.mixer_family != family:
        raise ShortScreenError("geometry family and model configuration differ")
    ledger = parameter_ledger(config)
    if row.get("parameter_ledger") != ledger:
        raise ShortScreenError("geometry parameter ledger differs")
    if ledger["total"] > MAX_PARAMETERS:
        raise ShortScreenError(
            "exact parameter count exceeds the frozen 100M geometry envelope"
        )
    return config, row


def make_bindings(
    *,
    config: SaiModelConfig,
    family: str,
    seed: int,
    train_identity_sha256: str,
    development_identity_sha256: str,
    code_sha256: str,
    environment_sha256: str,
    optimizer: TrainingRunConfig,
    micro_batch_size: int,
    sequences_per_update: int,
    training_sequences: int,
    training_utf8_bytes: int,
    development_sequences: int,
    development_batch_size: int = 1,
    checkpoint_interval: int = 1,
    mechanics_only: bool = False,
) -> tuple[CheckpointBindings, dict[str, Any]]:
    """Derive every checkpoint identity from the complete immutable run spec."""

    train_identity = _sha256(train_identity_sha256, "training stream identity")
    development_identity = _sha256(
        development_identity_sha256, "development stream identity"
    )
    if train_identity == development_identity:
        raise ShortScreenError("training and development streams must differ")
    code_identity = _sha256(code_sha256, "code identity")
    environment_identity = _sha256(environment_sha256, "environment identity")
    backend = "fla" if family in {"gdn_hybrid", "kda_mla_hybrid"} else "reference"
    config_identity = canonical_sha256(config.as_dict())
    model_identity = canonical_sha256(
        {
            "config_sha256": config_identity,
            "delta_backend": backend,
            "initialization_policy_sha256": POLICY_SHA256,
            "initialization_seed": seed,
        }
    )
    specification: dict[str, Any] = {
        "schema": SCHEMA,
        "evidence_class": "mechanics/development-screen-only",
        "scientific_promotion_authorized": False,
        "four_b_training_authorized": False,
        "config": config.as_dict(),
        "config_sha256": config_identity,
        "model_sha256": model_identity,
        "delta_backend": backend,
        "initialization_policy_sha256": POLICY_SHA256,
        "initialization_seed": seed,
        "training_stream_identity_sha256": train_identity,
        "development_stream_identity_sha256": development_identity,
        "code_sha256": code_identity,
        "environment_sha256": environment_identity,
        "optimizer": asdict(optimizer),
        "precision": {
            "parameter_storage": "float32",
            "optimizer_state": "float32",
            "activation_execution": "bfloat16_autocast",
            "recurrent_state": "operator_defined_float32",
        },
        "micro_batch_size_sequences": micro_batch_size,
        "sequences_per_update": sequences_per_update,
        "training_sequences": training_sequences,
        "training_utf8_bytes": training_utf8_bytes,
        "development_sequences": development_sequences,
        "development_batch_size_sequences": development_batch_size,
        "checkpoint_interval_steps": checkpoint_interval,
        "mechanics_only": mechanics_only,
    }
    run_identity = canonical_sha256(specification)
    specification["run_sha256"] = run_identity
    return (
        CheckpointBindings(
            model_sha256=model_identity,
            config_sha256=config_identity,
            ordered_stream_identity_sha256=train_identity,
            code_sha256=code_identity,
            environment_sha256=environment_identity,
            run_sha256=run_identity,
        ),
        specification,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if not path.name or path.name in {".", ".."}:
        raise ShortScreenError("output target differs")
    if not path.parent.is_dir() or path.parent.is_symlink() or path.is_symlink():
        raise ShortScreenError("output parent or target is unsafe")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise ShortScreenError("completed output already exists") from error
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _state_sha256(model: SaiCausalLM) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
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


def _prefix_bytes(report: dict[str, Any], sequences: int) -> int:
    value = report["prefix_utf8_bytes"].get(str(sequences))
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ShortScreenError(
            "development sequence count is not an exact frozen UTF-8 prefix"
        )
    return value


def update_micro_batch_sizes(
    *,
    global_step: int,
    training_sequences: int,
    sequences_per_update: int,
    micro_batch_size: int,
) -> tuple[int, ...]:
    """Return the exact micro-batch partition for one possibly partial update."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (
            global_step,
            training_sequences,
            sequences_per_update,
            micro_batch_size,
        )
    ):
        raise ShortScreenError("update sequence geometry differs")
    first_sequence = (global_step - 1) * sequences_per_update
    update_sequences = min(
        sequences_per_update,
        training_sequences - first_sequence,
    )
    if update_sequences <= 0 or micro_batch_size > sequences_per_update:
        raise ShortScreenError("update sequence geometry differs")
    full_batches, remainder = divmod(update_sequences, micro_batch_size)
    return (micro_batch_size,) * full_batches + ((remainder,) if remainder else ())


def _development_batches(
    root: Path,
    identity: str,
    *,
    sequences: int,
    batch_size: int,
):
    stream = ReceiptBoundTokenStream(
        root,
        expected_ordered_stream_identity_sha256=identity,
        verify_sources=True,
    )
    remaining = sequences
    while remaining:
        current = min(batch_size, remaining)
        yield tensorize_stream_batch(stream.next_batch(current), device="cuda")
        remaining -= current


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one bounded run, checkpointing every admitted interval."""

    if args.output.exists():
        raise ShortScreenError("completed output already exists")
    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.device_count() != 1
    ):
        raise ShortScreenError("exactly one CUDA BF16 GPU is required")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (
            args.batch_size,
            args.sequences_per_update,
            args.training_sequences,
            args.development_batch_size,
            args.optimizer_steps,
            args.checkpoint_interval,
            args.development_sequences,
        )
    ):
        raise ShortScreenError("run counts must be positive integers")
    if args.checkpoint_interval > args.optimizer_steps:
        raise ShortScreenError("checkpoint interval exceeds optimizer budget")
    expected_optimizer_steps = (
        args.training_sequences + args.sequences_per_update - 1
    ) // args.sequences_per_update
    if args.optimizer_steps != expected_optimizer_steps:
        raise ShortScreenError("optimizer steps do not match the exact sequence budget")
    if args.batch_size > args.sequences_per_update:
        raise ShortScreenError("micro-batch exceeds the update sequence budget")

    train_report = validate_frozen_stream(args.train_stream, verify_sources=True)
    development_report = validate_frozen_stream(
        args.development_stream, verify_sources=True
    )
    train_identity = train_report["ordered_stream_identity_sha256"]
    development_identity = development_report["ordered_stream_identity_sha256"]
    if train_identity != _sha256(args.train_identity, "training stream identity"):
        raise ShortScreenError("training stream identity differs")
    if development_identity != _sha256(
        args.development_identity, "development stream identity"
    ):
        raise ShortScreenError("development stream identity differs")
    if (
        train_report["tokenizer_identity_sha256"]
        != development_report["tokenizer_identity_sha256"]
    ):
        raise ShortScreenError("training and development tokenizer identities differ")
    if (
        train_report["sequence_length"] != 2048
        or development_report["sequence_length"] != 2048
    ):
        raise ShortScreenError("short-screen streams must use 2,048-token sequences")

    config, geometry_row = load_bounded_config(args.geometry, args.family)
    if (
        train_report["vocab_size"] != config.vocab_size
        or development_report["vocab_size"] != config.vocab_size
    ):
        raise ShortScreenError("stream vocabulary and model vocabulary differ")
    optimizer_config = TrainingRunConfig(
        optimizer_steps=args.optimizer_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        minimum_learning_rate_ratio=args.minimum_learning_rate_ratio,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    training_bytes = _prefix_bytes(train_report, args.training_sequences)
    bindings, specification = make_bindings(
        config=config,
        family=args.family,
        seed=args.seed,
        train_identity_sha256=train_identity,
        development_identity_sha256=development_identity,
        code_sha256=args.code_sha256,
        environment_sha256=args.environment_sha256,
        optimizer=optimizer_config,
        micro_batch_size=args.batch_size,
        sequences_per_update=args.sequences_per_update,
        training_sequences=args.training_sequences,
        training_utf8_bytes=training_bytes,
        development_sequences=args.development_sequences,
        development_batch_size=args.development_batch_size,
        checkpoint_interval=args.checkpoint_interval,
        mechanics_only=args.mechanics_only,
    )
    development_bytes = (
        None
        if args.mechanics_only
        else _prefix_bytes(development_report, args.development_sequences)
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = SaiCausalLM(config, delta_backend=specification["delta_backend"])
    initialization = initialize_sai_model(model, seed=args.seed)
    # Keep parameters and Adam moments in FP32.  BF16 is an execution dtype,
    # not a reason to reduce the precision of persistent optimizer state.
    model = model.to(device="cuda")
    parameter_count = exact_parameter_count(model)
    if parameter_count != geometry_row["parameter_ledger"]["total"]:
        raise ShortScreenError("instantiated parameter count differs")
    optimizer = build_adamw(model, optimizer_config)

    resumed_from = None
    if args.resume:
        restored = load_mechanics_checkpoint(
            args.checkpoint,
            model=model,
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
            "cursor": cursor.as_dict(),
        }
    else:
        if (
            args.checkpoint.exists()
            or args.checkpoint.with_name(
                f"{args.checkpoint.name}.manifest.json"
            ).exists()
        ):
            raise ShortScreenError("checkpoint already exists; use --resume")
        counters = TrainingCounters(0, 0, 0)
        cursor = None

    if counters.optimizer_steps > args.optimizer_steps:
        raise ShortScreenError("checkpoint exceeds the optimizer budget")
    expected_completed_sequences = min(
        counters.optimizer_steps * args.sequences_per_update,
        args.training_sequences,
    )
    if counters.sequences != expected_completed_sequences:
        raise ShortScreenError("checkpoint sequence and update budgets differ")
    required_sequences = args.training_sequences - counters.sequences
    if required_sequences < 0:
        raise ShortScreenError("checkpoint exceeds the sequence budget")
    train_stream = ReceiptBoundTokenStream(
        args.train_stream,
        expected_ordered_stream_identity_sha256=train_identity,
        resume_cursor=cursor,
        verify_sources=True,
    )
    if train_stream.remaining_sequences < required_sequences:
        raise ShortScreenError("training stream cannot satisfy the exact budget")

    losses: list[float] = []
    gradient_norms: list[float] = []
    model.train()
    torch.cuda.reset_peak_memory_stats()
    for global_step in range(counters.optimizer_steps + 1, args.optimizer_steps + 1):
        update_batch_sizes = update_micro_batch_sizes(
            global_step=global_step,
            training_sequences=args.training_sequences,
            sequences_per_update=args.sequences_per_update,
            micro_batch_size=args.batch_size,
        )
        update_sequences = sum(update_batch_sizes)
        raw_batches = []
        update_targets = 0
        for current_batch_size in update_batch_sizes:
            raw_batch = train_stream.next_batch(current_batch_size)
            raw_targets = sum(sum(row) for row in raw_batch.loss_mask)
            if raw_targets <= 0:
                raise ShortScreenError("training micro-batch has no valid targets")
            raw_batches.append((raw_batch, raw_targets))
            update_targets += raw_targets
        multiplier = learning_rate_multiplier(
            global_step,
            total_steps=args.optimizer_steps,
            warmup_steps=args.warmup_steps,
            minimum_ratio=args.minimum_learning_rate_ratio,
        )
        learning_rate = args.learning_rate * multiplier
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        update_loss_sum = 0.0
        for raw_batch, raw_targets in raw_batches:
            batch = tensorize_stream_batch(raw_batch, device="cuda")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(batch.input_ids, batch.segment_ids)
            loss_sum = F.cross_entropy(
                logits[batch.target_mask].float(),
                batch.target_ids[batch.target_mask],
                reduction="sum",
            )
            if (
                not bool(torch.isfinite(loss_sum).item())
                or int(batch.target_mask.sum().item()) != raw_targets
            ):
                raise ShortScreenError("training loss or target count differs")
            (loss_sum / update_targets).backward()
            update_loss_sum += float(loss_sum.detach())
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if not gradients or any(
            gradient is None or not bool(torch.isfinite(gradient).all().item())
            for gradient in gradients
        ):
            raise ShortScreenError("training gradients differ")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip_norm
        )
        if not bool(torch.isfinite(gradient_norm).item()):
            raise ShortScreenError("training gradient norm is nonfinite")
        optimizer.step()
        counters = TrainingCounters(
            optimizer_steps=global_step,
            sequences=counters.sequences + update_sequences,
            targets=counters.targets + update_targets,
        )
        losses.append(update_loss_sum / update_targets)
        gradient_norms.append(float(gradient_norm.detach()))
        if (
            global_step % args.checkpoint_interval == 0
            or global_step == args.optimizer_steps
        ):
            save_mechanics_checkpoint(
                args.checkpoint,
                model=model,
                optimizer=optimizer,
                bindings=bindings,
                counters=counters,
                cursor=train_stream.cursor,
            )

    validation = None
    if not args.mechanics_only:
        validation = evaluate_nll(
            model,
            _development_batches(
                args.development_stream,
                development_identity,
                sequences=args.development_sequences,
                batch_size=args.development_batch_size,
            ),
            stream_identity_sha256=development_identity,
            expected_sequences=args.development_sequences,
            admitted_utf8_bytes=development_bytes,
            benchmark_disjoint=True,
            autocast_dtype=torch.bfloat16,
        )
    torch.cuda.synchronize()
    checkpoint_manifest = json.loads(
        args.checkpoint.with_name(f"{args.checkpoint.name}.manifest.json").read_text()
    )
    payload: dict[str, Any] = {
        **specification,
        "status": "complete",
        "parameter_count": parameter_count,
        "initialization": initialization,
        "counters": counters.as_dict(),
        "stream_cursor": train_stream.cursor.as_dict(),
        "terminal_process_observations": {
            "steps": len(losses),
            "first_loss": losses[0] if losses else None,
            "last_loss": losses[-1] if losses else None,
            "minimum_loss": min(losses) if losses else None,
            "maximum_gradient_norm": max(gradient_norms) if gradient_norms else None,
        },
        "development_nll": asdict(validation) if validation is not None else None,
        "checkpoint": checkpoint_manifest["checkpoint"],
        "final_state_sha256": _state_sha256(model),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "resumed_from": resumed_from,
        "claim_limit": (
            "Mechanics/development NLL only; no public-benchmark improvement, "
            "architecture promotion, scaling claim, or 4B authorization."
        ),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--train-stream", type=Path, required=True)
    parser.add_argument("--train-identity", required=True)
    parser.add_argument("--development-stream", type=Path, required=True)
    parser.add_argument("--development-identity", required=True)
    parser.add_argument("--development-sequences", type=int, required=True)
    parser.add_argument("--development-batch-size", type=int, default=1)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--training-sequences", type=int, required=True)
    parser.add_argument("--sequences-per-update", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--minimum-learning-rate-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mechanics-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "receipt_sha256": payload["receipt_sha256"],
                "run_sha256": payload["run_sha256"],
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
