"""Bounded end-to-end parity for Sai's reference and FLA DeltaMixer paths.

This probe executes the real :class:`sai.model.reference.DeltaMixer` parameter
mapping.  It is intentionally tiny enough for the step-by-step reference
recurrence and is not a substitute for the separate B8 x 2,048 mechanics
canary.  A pass is mechanics evidence only; it is not model-quality evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sai.model.config import SaiModelConfig
from sai.model.fla_backend import (
    FLA_VERSION,
    FlaBackendOperators,
    load_fla_backend_operators,
    packed_cu_seqlens,
)
from sai.model.reference import DeltaMixer

SCHEMA = "sai-full-delta-mixer-fla-parity-v1"
_FAMILIES = ("gdn", "kda")
_SEQUENCE_LENGTHS = (1, 63, 64, 65)
_THRESHOLDS = {
    "forward": {"absolute": 0.02, "relative": 0.02},
    "input_gradient": {"absolute": 0.05, "relative": 0.05},
    "parameter_gradient": {"absolute": 0.05, "relative": 0.05},
}


class FullModelFlaParityError(RuntimeError):
    """A full DeltaMixer parity invariant differs or is incomplete."""


@dataclass
class _RecordedOperators:
    operators: FlaBackendOperators
    delta_calls: list[dict[str, Any]]
    conv_calls: list[dict[str, Any]]


def _tiny_config(family: str) -> SaiModelConfig:
    return SaiModelConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        mixer_family="gdn_hybrid" if family == "gdn" else "kda_mla_hybrid",
        linear_conv_kernel=4,
        mla_kv_rank=16,
        mla_qk_head_dim=16,
        mla_value_head_dim=16,
    )


def _segments(sequence_length: int, device: torch.device) -> torch.Tensor:
    """Exercise 1/63/64/65 segments and equal IDs across batch rows."""

    if sequence_length not in _SEQUENCE_LENGTHS:
        raise FullModelFlaParityError("parity sequence boundary differs")
    first = torch.full((sequence_length,), 19, dtype=torch.long)
    second = torch.full((sequence_length,), 19, dtype=torch.long)
    if sequence_length == 64:
        first[:63] = 11
    elif sequence_length == 65:
        first[0] = 7
        first[1:64] = 11
    # The final ID in row zero deliberately equals the first ID in row one.
    # packed_cu_seqlens must nevertheless insert an unconditional row boundary.
    return torch.stack((first, second)).to(device)


def _metric(
    actual: torch.Tensor,
    expected: torch.Tensor,
    threshold: dict[str, float],
) -> dict[str, float | bool]:
    if actual.shape != expected.shape or not actual.numel():
        raise FullModelFlaParityError("parity tensor geometry differs")
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    difference = (actual_float - expected_float).abs()
    allowance = threshold["absolute"] + threshold["relative"] * expected_float.abs()
    max_absolute = float(difference.max().item())
    max_normalized = float((difference / allowance).max().item())
    finite = bool(torch.isfinite(actual_float).all().item()) and bool(
        torch.isfinite(expected_float).all().item()
    )
    finite = finite and math.isfinite(max_absolute) and math.isfinite(max_normalized)
    return {
        "absolute_tolerance": threshold["absolute"],
        "relative_tolerance": threshold["relative"],
        "max_absolute_error": max_absolute,
        "max_normalized_error": max_normalized,
        "passed": finite and max_normalized <= 1.0,
    }


def _recording_operators(base: FlaBackendOperators) -> _RecordedOperators:
    delta_calls: list[dict[str, Any]] = []
    conv_calls: list[dict[str, Any]] = []

    def record_delta(
        operator: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
        family: str,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
        def wrapped(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
            delta_calls.append(
                {
                    "family": family,
                    "scale": kwargs.get("scale"),
                    "cu_seqlens": (
                        None
                        if kwargs.get("cu_seqlens") is None
                        else kwargs["cu_seqlens"].detach().cpu().tolist()
                    ),
                    "use_qk_l2norm_in_kernel": kwargs.get("use_qk_l2norm_in_kernel"),
                    "transpose_state_layout": kwargs.get("transpose_state_layout"),
                    "use_gate_in_kernel": kwargs.get("use_gate_in_kernel"),
                    "safe_gate": kwargs.get("safe_gate"),
                    "disable_recompute": kwargs.get("disable_recompute"),
                    "q_shape": list(kwargs["q"].shape),
                    "k_shape": list(kwargs["k"].shape),
                    "v_shape": list(kwargs["v"].shape),
                    "g_shape": list(kwargs["g"].shape),
                    "beta_shape": list(kwargs["beta"].shape),
                }
            )
            return operator(**kwargs)

        return wrapped

    def record_conv(**kwargs: Any) -> torch.Tensor | tuple[torch.Tensor, ...]:
        conv_calls.append(
            {
                "cu_seqlens": (
                    None
                    if kwargs.get("cu_seqlens") is None
                    else kwargs["cu_seqlens"].detach().cpu().tolist()
                ),
                "activation": kwargs.get("activation"),
                "x_shape": list(kwargs["x"].shape),
                "weight_shape": list(kwargs["weight"].shape),
                "bias": kwargs.get("bias"),
            }
        )
        return base.causal_conv1d(**kwargs)

    return _RecordedOperators(
        FlaBackendOperators(
            gated_delta_chunk=record_delta(base.gated_delta_chunk, "gdn"),
            kda_chunk=record_delta(base.kda_chunk, "kda"),
            causal_conv1d=record_conv,
            version=base.version,
        ),
        delta_calls,
        conv_calls,
    )


def _run_case(
    family: str,
    sequence_length: int,
    *,
    device: torch.device,
    operators: FlaBackendOperators,
    seed: int,
    use_bf16: bool,
) -> dict[str, Any]:
    config = _tiny_config(family)
    torch.manual_seed(seed + sequence_length + (1_000 if family == "kda" else 0))
    reference = DeltaMixer(
        config, channel_wise_decay=family == "kda", backend="reference"
    ).to(device)
    recorded = _recording_operators(operators)
    accelerated = DeltaMixer(
        config,
        channel_wise_decay=family == "kda",
        backend="fla",
        fla_operators=recorded.operators,
    ).to(device)
    accelerated.load_state_dict(reference.state_dict(), strict=True)
    reference.eval()
    accelerated.eval()

    reference_names = tuple(name for name, _ in reference.named_parameters())
    accelerated_names = tuple(name for name, _ in accelerated.named_parameters())
    if reference_names != accelerated_names or not reference_names:
        raise FullModelFlaParityError("DeltaMixer parameter mapping differs")
    for name, expected in reference.state_dict().items():
        actual = accelerated.state_dict()[name]
        if not torch.equal(actual, expected):
            raise FullModelFlaParityError("DeltaMixer initial parameter bytes differ")

    generator = torch.Generator(device="cpu").manual_seed(seed * 3 + sequence_length)
    hidden = torch.randn(2, sequence_length, config.hidden_size, generator=generator)
    reference_input = hidden.to(device).detach().requires_grad_(True)
    accelerated_input = hidden.to(device).detach().requires_grad_(True)
    segment_ids = _segments(sequence_length, device)
    expected_offsets = packed_cu_seqlens(segment_ids).detach().cpu().tolist()

    autocast = torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_bf16,
    )
    with autocast:
        reference_output = reference(reference_input, segment_ids)
        accelerated_output = accelerated(accelerated_input, segment_ids)
        reference_loss = reference_output.float().square().mean()
        reference_loss = reference_loss + 0.013 * reference_output.float().mean()
        accelerated_loss = accelerated_output.float().square().mean()
        accelerated_loss = accelerated_loss + 0.013 * accelerated_output.float().mean()
    reference_loss.backward()
    accelerated_loss.backward()

    if reference_input.grad is None or accelerated_input.grad is None:
        raise FullModelFlaParityError("DeltaMixer input gradient is absent")
    forward_metric = _metric(
        accelerated_output, reference_output, _THRESHOLDS["forward"]
    )
    input_metric = _metric(
        accelerated_input.grad,
        reference_input.grad,
        _THRESHOLDS["input_gradient"],
    )
    parameter_metrics: dict[str, dict[str, float | bool]] = {}
    reference_parameters = dict(reference.named_parameters())
    accelerated_parameters = dict(accelerated.named_parameters())
    for name in reference_names:
        expected_gradient = reference_parameters[name].grad
        actual_gradient = accelerated_parameters[name].grad
        if expected_gradient is None or actual_gradient is None:
            raise FullModelFlaParityError(
                f"DeltaMixer parameter gradient absent: {name}"
            )
        parameter_metrics[name] = _metric(
            actual_gradient,
            expected_gradient,
            _THRESHOLDS["parameter_gradient"],
        )

    if len(recorded.delta_calls) != 1 or len(recorded.conv_calls) != 3:
        raise FullModelFlaParityError("DeltaMixer FLA dispatch count differs")
    delta_call = recorded.delta_calls[0]
    total_tokens = 2 * sequence_length
    expected_common_shape = [1, total_tokens, config.num_attention_heads]
    expected_gate_shape = expected_common_shape + (
        [config.head_dim] if family == "kda" else []
    )
    expected_delta_flags = (
        delta_call["family"] == family
        and delta_call["transpose_state_layout"] is False
        and (
            (
                delta_call["use_gate_in_kernel"] is False
                and delta_call["safe_gate"] is False
                and delta_call["disable_recompute"] is False
            )
            if family == "kda"
            else (
                delta_call["use_gate_in_kernel"] is None
                and delta_call["safe_gate"] is None
                and delta_call["disable_recompute"] is None
            )
        )
    )
    mapping_passed = (
        expected_delta_flags
        and delta_call["scale"] == 1.0
        and delta_call["use_qk_l2norm_in_kernel"] is False
        and delta_call["cu_seqlens"] == expected_offsets
        and delta_call["q_shape"] == expected_common_shape + [config.head_dim]
        and delta_call["k_shape"] == expected_common_shape + [config.head_dim]
        and delta_call["v_shape"] == expected_common_shape + [config.head_dim]
        and delta_call["g_shape"] == expected_gate_shape
        and delta_call["beta_shape"] == expected_common_shape
        and all(call["activation"] == "silu" for call in recorded.conv_calls)
        and all(call["cu_seqlens"] == expected_offsets for call in recorded.conv_calls)
        and all(
            call["x_shape"] == [1, total_tokens, config.attention_width]
            for call in recorded.conv_calls
        )
        and all(
            call["weight_shape"] == [config.attention_width, config.linear_conv_kernel]
            for call in recorded.conv_calls
        )
        and all(call["bias"] is None for call in recorded.conv_calls)
    )
    gradients_complete = set(parameter_metrics) == set(reference_names)
    passed = (
        mapping_passed
        and gradients_complete
        and bool(forward_metric["passed"])
        and bool(input_metric["passed"])
        and all(bool(metric["passed"]) for metric in parameter_metrics.values())
    )
    return {
        "family": family,
        "batch": 2,
        "sequence_length": sequence_length,
        "packed_cu_seqlens": expected_offsets,
        "equal_segment_id_across_row_boundary": bool(
            segment_ids[0, -1] == segment_ids[1, 0]
        ),
        "dtype": "torch.bfloat16" if use_bf16 else "torch.float32",
        "forward": forward_metric,
        "input_gradient": input_metric,
        "parameter_gradients": parameter_metrics,
        "expected_parameter_names": list(reference_names),
        "all_parameter_gradients_compared": gradients_complete,
        "fla_mapping": {
            "scale": delta_call["scale"],
            "use_qk_l2norm_in_kernel": delta_call["use_qk_l2norm_in_kernel"],
            "family_specific_flags_passed": expected_delta_flags,
            "delta_cu_seqlens": delta_call["cu_seqlens"],
            "q_shape": delta_call["q_shape"],
            "k_shape": delta_call["k_shape"],
            "v_shape": delta_call["v_shape"],
            "g_shape": delta_call["g_shape"],
            "beta_shape": delta_call["beta_shape"],
            "conv_dispatches": len(recorded.conv_calls),
            "passed": mapping_passed,
        },
        "passed": passed,
    }


def run_full_delta_mixer_parity(
    *,
    device: torch.device | str = "cuda",
    seed: int = 20260821,
    operators: FlaBackendOperators | None = None,
) -> dict[str, Any]:
    """Compare complete GDN/KDA DeltaMixers without optimizer or training steps.

    Omitting ``operators`` is the only production path: it loads pinned FLA 0.4.2
    and requires CUDA BF16.  Injected operators exist solely for deterministic
    contract tests and can never produce a production-qualified report.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FullModelFlaParityError("parity seed differs")
    resolved_device = torch.device(device)
    production = operators is None
    if production:
        if resolved_device.type != "cuda":
            raise FullModelFlaParityError("production parity requires CUDA")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise FullModelFlaParityError("production CUDA BF16 is unavailable")
        operators = load_fla_backend_operators()
        if operators.version != FLA_VERSION:
            raise FullModelFlaParityError("production FLA version differs")
    if operators is None or operators.version != FLA_VERSION:
        raise FullModelFlaParityError("parity FLA operator version differs")

    cases = [
        _run_case(
            family,
            sequence_length,
            device=resolved_device,
            operators=operators,
            seed=seed,
            use_bf16=production,
        )
        for family in _FAMILIES
        for sequence_length in _SEQUENCE_LENGTHS
    ]
    all_passed = all(case["passed"] for case in cases)
    qualified = production and all_passed
    return {
        "schema": SCHEMA,
        "status": (
            "production_cuda_qualified"
            if qualified
            else ("test_oracle_passed" if all_passed else "parity_failed")
        ),
        "production_cuda_qualified": qualified,
        "scope": "bounded_full_delta_mixer_forward_backward_parity",
        "fla_version": operators.version,
        "seed": seed,
        "thresholds": _THRESHOLDS,
        "cases": cases,
        "checks": {
            "families": list(_FAMILIES),
            "sequence_boundaries": list(_SEQUENCE_LENGTHS),
            "forward_compared": all_passed,
            "input_gradients_compared": all_passed,
            "every_parameter_gradient_compared": all(
                case["all_parameter_gradients_compared"] for case in cases
            ),
            "packed_row_boundaries_verified": all(
                case["equal_segment_id_across_row_boundary"] for case in cases
            ),
            "explicit_scale_one_verified": all(
                case["fla_mapping"]["scale"] == 1.0 for case in cases
            ),
        },
        "optimizer_steps": 0,
        "training_authorized": False,
        "architecture_promoted": False,
        "limitations": [
            "small_reference_geometry_only",
            "not_the_exact_b8_x_2048_canary",
            "mechanics_only_not_model_quality_evidence",
            "no_optimizer_or_parameter_update",
        ],
    }


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FullModelFlaParityError("parity output already exists")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise FullModelFlaParityError("parity output parent is missing or unsafe")
    stage = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    payload = run_full_delta_mixer_parity(device="cuda", seed=args.seed)
    if not payload["production_cuda_qualified"]:
        raise FullModelFlaParityError("production full-model parity failed")
    _write_create_only(args.output, payload)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
