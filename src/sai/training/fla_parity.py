"""Bounded, no-training CUDA parity probes for Sai's required FLA kernels.

The module deliberately imports FLA only when the CUDA qualification entry point is
called.  Importing it on a CPU workstation therefore does not require FLA, Triton,
or a CUDA runtime.  A passing receipt qualifies only the named kernel mechanics; it
does not authorize training, promote an architecture, or authorize a 4B run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCHEMA = "sai-fla-parity-receipt-v1"
_FAMILIES = ("gated_delta", "kda")
_METRICS = (
    "packed_output_chunk_vs_recurrent",
    "packed_state_chunk_vs_recurrent",
    "chunk_packed_vs_reset_output",
    "chunk_packed_vs_reset_state",
    "recurrent_packed_vs_reset_output",
    "recurrent_packed_vs_reset_state",
    "gradient_q_chunk_vs_recurrent",
    "gradient_k_chunk_vs_recurrent",
    "gradient_v_chunk_vs_recurrent",
    "gradient_g_chunk_vs_recurrent",
    "gradient_beta_chunk_vs_recurrent",
)
_TOP_LEVEL_KEYS = {
    "schema",
    "status",
    "parity_qualified",
    "production_cuda_qualified",
    "scope",
    "training_authorized",
    "architecture_promoted",
    "four_b_training_authorized",
    "optimizer_steps",
    "model_parameters",
    "gpu_allocation_consumed",
    "training_gpu_jobs_submitted",
    "seed",
    "dtype",
    "geometry",
    "thresholds",
    "environment",
    "operators",
    "cases",
    "checks",
    "limitations",
    "receipt_sha256",
}
_CASE_KEYS = {
    "family",
    "packed_cu_seqlens",
    "packed_sequences",
    "backward_calls",
    "all_forward_values_finite",
    "all_final_states_finite",
    "all_gradients_present",
    "all_gradients_finite",
    "bf16_forward_backward_mechanics",
    "metrics",
    "tensor_sha256",
    "passed",
}
_LIMITATIONS = [
    "kernel_mechanics_only_not_model_quality_evidence",
    "no_optimizer_or_parameter_update_executed",
    "no_training_or_architecture_promotion_authorized",
    "no_4b_training_authorized",
    "bounded_geometry_not_a_throughput_benchmark",
]


class FlaParityError(RuntimeError):
    """The FLA runtime, mechanics, or signed receipt differs."""


@dataclass(frozen=True)
class FlaOperators:
    """The exact four callables compared by the bounded probe."""

    gated_delta_chunk: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    gated_delta_recurrent: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    kda_chunk: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    kda_recurrent: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    source: str
    version: str
    mock: bool = False


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    return canonical_sha256(
        {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def _load_fla_operators() -> FlaOperators:
    try:
        gated_delta = importlib.import_module("fla.ops.gated_delta_rule")
        kda = importlib.import_module("fla.ops.kda")
        chunk_gated_delta_rule = gated_delta.chunk_gated_delta_rule
        fused_recurrent_gated_delta_rule = gated_delta.fused_recurrent_gated_delta_rule
        chunk_kda = kda.chunk_kda
        fused_recurrent_kda = kda.fused_recurrent_kda
    except (ImportError, AttributeError) as error:
        raise FlaParityError(
            "FLA gated-delta and KDA chunk/recurrent operators are unavailable"
        ) from error
    version = "unknown"
    for distribution in ("fla-core", "flash-linear-attention"):
        try:
            version = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    if version == "unknown":
        raise FlaParityError("installed FLA distribution version is unavailable")
    return FlaOperators(
        gated_delta_chunk=chunk_gated_delta_rule,
        gated_delta_recurrent=fused_recurrent_gated_delta_rule,
        kda_chunk=chunk_kda,
        kda_recurrent=fused_recurrent_kda,
        source="fla.ops",
        version=version,
    )


def _operator_name(operator: Callable[..., Any]) -> str:
    module = getattr(operator, "__module__", "unknown")
    name = getattr(operator, "__qualname__", getattr(operator, "__name__", "unknown"))
    return f"{module}.{name}"


def _finite_tensor(value: torch.Tensor | None) -> bool:
    return value is not None and bool(torch.isfinite(value.float()).all().item())


def _metric(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, float | bool]:
    if actual.shape != expected.shape:
        raise FlaParityError("parity tensor shapes differ")
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    difference = (actual_float - expected_float).abs()
    allowance = absolute_tolerance + relative_tolerance * expected_float.abs()
    if difference.numel() == 0:
        raise FlaParityError("parity tensors must be nonempty")
    max_absolute = float(difference.max().item())
    max_normalized = float((difference / allowance).max().item())
    finite = all(math.isfinite(value) for value in (max_absolute, max_normalized))
    return {
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "max_absolute_error": max_absolute,
        "max_normalized_error": max_normalized,
        "passed": finite and max_normalized <= 1.0,
    }


def _inputs(
    family: str,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    total_tokens: int,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> dict[str, torch.Tensor]:
    shape = (1, total_tokens, heads)

    def normal(tail: tuple[int, ...], scale: float = 0.2) -> torch.Tensor:
        value = torch.randn(*shape, *tail, generator=generator) * scale
        return value.to(device=device, dtype=dtype).detach().requires_grad_(True)

    q = normal((key_dim,))
    k = normal((key_dim,))
    v = normal((value_dim,))
    gate_tail = () if family == "gated_delta" else (key_dim,)
    raw_gate = torch.randn(*shape, *gate_tail, generator=generator)
    g = (-0.05 - 0.15 * F.softplus(raw_gate)).to(device=device, dtype=dtype)
    g = g.detach().requires_grad_(True)
    raw_beta = torch.randn(*shape, generator=generator)
    beta = torch.sigmoid(raw_beta).to(device=device, dtype=dtype)
    beta = beta.detach().requires_grad_(True)
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta}


def _call(
    operator: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = operator(
        **inputs,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    if (
        not isinstance(output, tuple)
        or len(output) != 2
        or not isinstance(output[0], torch.Tensor)
        or not isinstance(output[1], torch.Tensor)
    ):
        raise FlaParityError("FLA operator did not return output and final state")
    return output


def _packed_execution(
    operator: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    base_inputs: dict[str, torch.Tensor],
    cu_seqlens: torch.Tensor,
) -> dict[str, Any]:
    inputs = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in base_inputs.items()
    }
    output, state = _call(operator, inputs, cu_seqlens)
    if not _finite_tensor(output) or not _finite_tensor(state):
        raise FlaParityError("FLA packed forward produced non-finite values")
    loss = output.float().square().mean()
    loss.backward()
    gradients = {name: value.grad for name, value in inputs.items()}
    return {
        "output": output.detach(),
        "state": state.detach(),
        "gradients": gradients,
    }


def _reset_execution(
    operator: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    base_inputs: dict[str, torch.Tensor],
    offsets: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    states = []
    with torch.no_grad():
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True):
            segment = {
                name: value[:, start:stop] for name, value in base_inputs.items()
            }
            output, state = _call(operator, segment, None)
            outputs.append(output)
            states.append(state)
    return torch.cat(outputs, dim=1), torch.cat(states, dim=0)


def _probe_family(
    family: str,
    chunk: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    recurrent: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    offsets: tuple[int, ...],
    heads: int,
    key_dim: int,
    value_dim: int,
    thresholds: dict[str, dict[str, float]],
) -> dict[str, Any]:
    base = _inputs(
        family,
        generator=generator,
        device=device,
        dtype=dtype,
        total_tokens=offsets[-1],
        heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
    )
    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.long)
    chunk_packed = _packed_execution(chunk, base, cu_seqlens)
    recurrent_packed = _packed_execution(recurrent, base, cu_seqlens)
    chunk_reset = _reset_execution(chunk, base, offsets)
    recurrent_reset = _reset_execution(recurrent, base, offsets)

    forward = thresholds["forward"]
    state = thresholds["state"]
    gradient = thresholds["gradient"]
    metrics = {
        "packed_output_chunk_vs_recurrent": _metric(
            chunk_packed["output"], recurrent_packed["output"], **forward
        ),
        "packed_state_chunk_vs_recurrent": _metric(
            chunk_packed["state"], recurrent_packed["state"], **state
        ),
        "chunk_packed_vs_reset_output": _metric(
            chunk_packed["output"], chunk_reset[0], **forward
        ),
        "chunk_packed_vs_reset_state": _metric(
            chunk_packed["state"], chunk_reset[1], **state
        ),
        "recurrent_packed_vs_reset_output": _metric(
            recurrent_packed["output"], recurrent_reset[0], **forward
        ),
        "recurrent_packed_vs_reset_state": _metric(
            recurrent_packed["state"], recurrent_reset[1], **state
        ),
    }
    gradients = []
    tensor_hashes = {
        "packed_chunk_output": _tensor_sha256(chunk_packed["output"]),
        "packed_chunk_state": _tensor_sha256(chunk_packed["state"]),
        "packed_recurrent_output": _tensor_sha256(recurrent_packed["output"]),
        "packed_recurrent_state": _tensor_sha256(recurrent_packed["state"]),
    }
    for name in ("q", "k", "v", "g", "beta"):
        chunk_gradient = chunk_packed["gradients"][name]
        recurrent_gradient = recurrent_packed["gradients"][name]
        gradients.extend((chunk_gradient, recurrent_gradient))
        if chunk_gradient is None or recurrent_gradient is None:
            continue
        metrics[f"gradient_{name}_chunk_vs_recurrent"] = _metric(
            chunk_gradient, recurrent_gradient, **gradient
        )
        tensor_hashes[f"chunk_gradient_{name}"] = _tensor_sha256(chunk_gradient)
        tensor_hashes[f"recurrent_gradient_{name}"] = _tensor_sha256(recurrent_gradient)
    all_present = all(value is not None for value in gradients)
    all_finite = all(_finite_tensor(value) for value in gradients)
    metric_boundary = set(metrics) == set(_METRICS)
    passed = (
        metric_boundary
        and all_present
        and all_finite
        and all(bool(metric["passed"]) for metric in metrics.values())
    )
    return {
        "family": family,
        "packed_cu_seqlens": list(offsets),
        "packed_sequences": len(offsets) - 1,
        "backward_calls": 2,
        "all_forward_values_finite": _finite_tensor(chunk_packed["output"])
        and _finite_tensor(recurrent_packed["output"]),
        "all_final_states_finite": _finite_tensor(chunk_packed["state"])
        and _finite_tensor(recurrent_packed["state"]),
        "all_gradients_present": all_present,
        "all_gradients_finite": all_finite,
        "bf16_forward_backward_mechanics": dtype == torch.bfloat16,
        "metrics": metrics,
        "tensor_sha256": tensor_hashes,
        "passed": passed,
    }


def run_parity_mechanics(
    operators: FlaOperators,
    device: torch.device | str,
    *,
    seed: int = 20260821,
    production_cuda: bool = False,
) -> dict[str, Any]:
    """Run bounded parity mechanics, optionally as real CUDA qualification.

    ``production_cuda=False`` exists for local mock contract tests.  Such a receipt
    is permanently marked non-qualified and cannot be promoted by re-signing it.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FlaParityError("seed differs")
    resolved_device = torch.device(device)
    if production_cuda:
        if operators.mock or resolved_device.type != "cuda":
            raise FlaParityError("production qualification requires real CUDA FLA")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise FlaParityError("CUDA BF16 mechanics are unavailable")
    offsets = (0, 67, 98, 103)
    geometry = {
        "batch": 1,
        "total_tokens": offsets[-1],
        "sequence_lengths": [
            stop - start for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
        ],
        "heads": 2,
        "key_dim": 16,
        "value_dim": 16,
    }
    thresholds = {
        "forward": {"absolute_tolerance": 0.03, "relative_tolerance": 0.08},
        "state": {"absolute_tolerance": 0.03, "relative_tolerance": 0.08},
        "gradient": {"absolute_tolerance": 0.06, "relative_tolerance": 0.15},
    }
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cases = [
        _probe_family(
            "gated_delta",
            operators.gated_delta_chunk,
            operators.gated_delta_recurrent,
            generator=generator,
            device=resolved_device,
            dtype=torch.bfloat16,
            offsets=offsets,
            heads=geometry["heads"],
            key_dim=geometry["key_dim"],
            value_dim=geometry["value_dim"],
            thresholds=thresholds,
        ),
        _probe_family(
            "kda",
            operators.kda_chunk,
            operators.kda_recurrent,
            generator=generator,
            device=resolved_device,
            dtype=torch.bfloat16,
            offsets=offsets,
            heads=geometry["heads"],
            key_dim=geometry["key_dim"],
            value_dim=geometry["value_dim"],
            thresholds=thresholds,
        ),
    ]
    all_cases_passed = all(case["passed"] for case in cases)
    qualified = production_cuda and all_cases_passed
    checks = {
        "both_operator_families_exercised": [case["family"] for case in cases]
        == list(_FAMILIES),
        "chunk_vs_fused_recurrent_compared": all_cases_passed,
        "packed_cu_seqlens_reset_verified": all_cases_passed,
        "backward_gradients_finite": all(
            case["all_gradients_present"] and case["all_gradients_finite"]
            for case in cases
        ),
        "bf16_forward_backward_mechanics": all(
            case["bf16_forward_backward_mechanics"] for case in cases
        ),
        "no_optimizer_step": True,
        "no_training": True,
    }
    status = (
        "cuda_fla_parity_qualified"
        if qualified
        else (
            "mock_mechanics_passed"
            if all_cases_passed and operators.mock and not production_cuda
            else "parity_failed"
        )
    )
    cuda_name = None
    cuda_capability = None
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(resolved_device)
        cuda_capability = list(torch.cuda.get_device_capability(resolved_device))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": status,
        "parity_qualified": qualified,
        "production_cuda_qualified": qualified,
        "scope": "bounded_fla_kernel_parity_mechanics",
        "training_authorized": False,
        "architecture_promoted": False,
        "four_b_training_authorized": False,
        "optimizer_steps": 0,
        "model_parameters": 0,
        "gpu_allocation_consumed": production_cuda,
        "training_gpu_jobs_submitted": 0,
        "seed": seed,
        "dtype": "torch.bfloat16",
        "geometry": geometry,
        "thresholds": thresholds,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device_type": resolved_device.type,
            "cuda_available": torch.cuda.is_available(),
            "cuda_bf16_supported": (
                torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
            ),
            "cuda_device_name": cuda_name,
            "cuda_capability": cuda_capability,
        },
        "operators": {
            "source": operators.source,
            "version": operators.version,
            "mock": operators.mock,
            "gated_delta_chunk": _operator_name(operators.gated_delta_chunk),
            "gated_delta_recurrent": _operator_name(operators.gated_delta_recurrent),
            "kda_chunk": _operator_name(operators.kda_chunk),
            "kda_recurrent": _operator_name(operators.kda_recurrent),
        },
        "cases": cases,
        "checks": checks,
        "limitations": list(_LIMITATIONS),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return validate_receipt(payload)


def run_cuda_fla_parity(*, seed: int = 20260821) -> dict[str, Any]:
    """Qualify real BF16 CUDA FLA mechanics without training a model."""

    return run_parity_mechanics(
        _load_fla_operators(), torch.device("cuda"), seed=seed, production_cuda=True
    )


def _strict_metric(value: Any, expected: dict[str, float]) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "absolute_tolerance",
        "relative_tolerance",
        "max_absolute_error",
        "max_normalized_error",
        "passed",
    }:
        return False
    numbers = (value.get("max_absolute_error"), value.get("max_normalized_error"))
    return (
        value.get("absolute_tolerance") == expected["absolute_tolerance"]
        and value.get("relative_tolerance") == expected["relative_tolerance"]
        and all(
            not isinstance(number, bool)
            and isinstance(number, (int, float))
            and math.isfinite(float(number))
            and float(number) >= 0
            for number in numbers
        )
        and value.get("passed") is (float(value["max_normalized_error"]) <= 1.0)
    )


def validate_receipt(payload: Any) -> dict[str, Any]:
    """Validate the complete receipt and reject promotion/training overclaims."""

    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_KEYS:
        raise FlaParityError("FLA parity receipt keys differ")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != canonical_sha256(unsigned):
        raise FlaParityError("FLA parity receipt hash differs")
    geometry = payload.get("geometry")
    thresholds = payload.get("thresholds")
    if geometry != {
        "batch": 1,
        "total_tokens": 103,
        "sequence_lengths": [67, 31, 5],
        "heads": 2,
        "key_dim": 16,
        "value_dim": 16,
    } or thresholds != {
        "forward": {"absolute_tolerance": 0.03, "relative_tolerance": 0.08},
        "state": {"absolute_tolerance": 0.03, "relative_tolerance": 0.08},
        "gradient": {"absolute_tolerance": 0.06, "relative_tolerance": 0.15},
    }:
        raise FlaParityError("FLA parity geometry or thresholds differ")
    cases = payload.get("cases")
    if (
        not isinstance(cases, list)
        or len(cases) != 2
        or [case.get("family") for case in cases if isinstance(case, dict)]
        != list(_FAMILIES)
    ):
        raise FlaParityError("FLA parity cases differ")
    for case in cases:
        if not isinstance(case, dict) or set(case) != _CASE_KEYS:
            raise FlaParityError("FLA parity case keys differ")
        metrics = case.get("metrics")
        hashes = case.get("tensor_sha256")
        if (
            case.get("packed_cu_seqlens") != [0, 67, 98, 103]
            or case.get("packed_sequences") != 3
            or case.get("backward_calls") != 2
            or not isinstance(metrics, dict)
            or set(metrics) != set(_METRICS)
            or not isinstance(hashes, dict)
            or set(hashes)
            != {
                "packed_chunk_output",
                "packed_chunk_state",
                "packed_recurrent_output",
                "packed_recurrent_state",
                "chunk_gradient_q",
                "recurrent_gradient_q",
                "chunk_gradient_k",
                "recurrent_gradient_k",
                "chunk_gradient_v",
                "recurrent_gradient_v",
                "chunk_gradient_g",
                "recurrent_gradient_g",
                "chunk_gradient_beta",
                "recurrent_gradient_beta",
            }
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in hashes.values()
            )
        ):
            raise FlaParityError("FLA packed/reset evidence differs")
        for name, metric in metrics.items():
            boundary = (
                thresholds["gradient"]
                if name.startswith("gradient_")
                else (
                    thresholds["state"]
                    if name.endswith("state") or "state_" in name
                    else thresholds["forward"]
                )
            )
            if not _strict_metric(metric, boundary):
                raise FlaParityError("FLA parity metric differs")
        computed_pass = all(
            case.get(field) is True
            for field in (
                "all_forward_values_finite",
                "all_final_states_finite",
                "all_gradients_present",
                "all_gradients_finite",
                "bf16_forward_backward_mechanics",
            )
        ) and all(metric["passed"] for metric in metrics.values())
        if case.get("passed") is not computed_pass:
            raise FlaParityError("FLA parity case summary differs")
    all_passed = all(case["passed"] for case in cases)
    environment = payload.get("environment")
    operators = payload.get("operators")
    checks = payload.get("checks")
    if (
        not isinstance(environment, dict)
        or set(environment)
        != {
            "python",
            "torch",
            "platform",
            "device_type",
            "cuda_available",
            "cuda_bf16_supported",
            "cuda_device_name",
            "cuda_capability",
        }
        or not all(
            isinstance(environment.get(field), str) and environment[field]
            for field in ("python", "torch", "platform", "device_type")
        )
        or environment.get("device_type") not in {"cpu", "cuda"}
        or not isinstance(environment.get("cuda_available"), bool)
        or not isinstance(environment.get("cuda_bf16_supported"), bool)
        or (
            environment["device_type"] == "cuda"
            and (
                environment["cuda_available"] is not True
                or environment["cuda_bf16_supported"] is not True
                or not isinstance(environment.get("cuda_device_name"), str)
                or not environment["cuda_device_name"]
                or not isinstance(environment.get("cuda_capability"), list)
                or len(environment["cuda_capability"]) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in environment["cuda_capability"]
                )
            )
        )
        or (
            environment["device_type"] == "cpu"
            and (
                environment.get("cuda_device_name") is not None
                or environment.get("cuda_capability") is not None
            )
        )
    ):
        raise FlaParityError("FLA parity environment differs")
    if (
        not isinstance(operators, dict)
        or set(operators)
        != {
            "source",
            "version",
            "mock",
            "gated_delta_chunk",
            "gated_delta_recurrent",
            "kda_chunk",
            "kda_recurrent",
        }
        or not isinstance(operators.get("mock"), bool)
        or not all(
            isinstance(operators.get(field), str) and operators[field]
            for field in set(operators) - {"mock"}
        )
    ):
        raise FlaParityError("FLA operator identity differs")
    qualified = (
        all_passed
        and environment.get("device_type") == "cuda"
        and environment.get("cuda_available") is True
        and environment.get("cuda_bf16_supported") is True
        and operators.get("mock") is False
        and operators.get("source") == "fla.ops"
        and operators.get("version") != "unknown"
        and operators["gated_delta_chunk"].endswith(".chunk_gated_delta_rule")
        and operators["gated_delta_recurrent"].endswith(
            ".fused_recurrent_gated_delta_rule"
        )
        and operators["kda_chunk"].endswith(".chunk_kda")
        and operators["kda_recurrent"].endswith(".fused_recurrent_kda")
    )
    expected_status = (
        "cuda_fla_parity_qualified"
        if qualified
        else (
            "mock_mechanics_passed"
            if all_passed and operators.get("mock") is True
            else "parity_failed"
        )
    )
    expected_checks = {
        "both_operator_families_exercised": True,
        "chunk_vs_fused_recurrent_compared": all_passed,
        "packed_cu_seqlens_reset_verified": all_passed,
        "backward_gradients_finite": all(
            case["all_gradients_present"] and case["all_gradients_finite"]
            for case in cases
        ),
        "bf16_forward_backward_mechanics": all(
            case["bf16_forward_backward_mechanics"] for case in cases
        ),
        "no_optimizer_step": True,
        "no_training": True,
    }
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != expected_status
        or payload.get("parity_qualified") is not qualified
        or payload.get("production_cuda_qualified") is not qualified
        or payload.get("scope") != "bounded_fla_kernel_parity_mechanics"
        or payload.get("training_authorized") is not False
        or payload.get("architecture_promoted") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("optimizer_steps") != 0
        or payload.get("model_parameters") != 0
        or payload.get("gpu_allocation_consumed")
        is not (environment.get("device_type") == "cuda")
        or payload.get("training_gpu_jobs_submitted") != 0
        or isinstance(payload.get("seed"), bool)
        or not isinstance(payload.get("seed"), int)
        or payload["seed"] < 0
        or payload.get("dtype") != "torch.bfloat16"
        or checks != expected_checks
        or payload.get("limitations") != _LIMITATIONS
    ):
        raise FlaParityError("FLA parity receipt boundary differs")
    return payload


def write_receipt(payload: dict[str, Any], output: Path) -> None:
    validate_receipt(payload)
    if output.exists():
        raise FlaParityError("FLA parity receipt already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    receipt = run_cuda_fla_parity(seed=args.seed)
    write_receipt(receipt, args.output)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "training_authorized": False,
                "four_b_training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
