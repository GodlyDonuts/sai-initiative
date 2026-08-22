"""Prospective, family-separated FLA 0.4.2 semantic parity qualification.

This v2 gate does not reinterpret or replace any v1 receipt.  It compares Sai's
exact packed adapter mapping and the direct convolution/recurrence forward and
backward mechanics using the strict RMSE ratios used by upstream FLA 0.4.2.
Every tensor in every case must pass; results are never averaged for admission.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sai.data.token_stream import canonical_sha256
from sai.model.fla_backend import (
    FLA_VERSION,
    FlaBackendOperators,
    fla_causal_conv1d,
    fla_delta_recurrence,
    load_fla_backend_operators,
    packed_cu_seqlens,
)

SCHEMA = "sai-fla-semantic-parity-v2"
FAMILIES = ("gdn", "kda")
SEQUENCE_LENGTHS = (1, 63, 64, 65)
CALIBRATION_SEEDS = (20260821, 20260822, 20260823)
PRODUCTION_SEEDS = (20260824, 20260825, 20260826)
CONV_THRESHOLDS = {"y": 0.001, "dx": 0.001, "dw": 0.001}
RECURRENCE_THRESHOLDS = {
    "o": 0.005,
    "dq": 0.007,
    "dk": 0.008,
    "dv": 0.007,
    "dg": 0.015,
    "dbeta": 0.015,
}


class FlaSemanticParityError(RuntimeError):
    """A v2 semantic mapping, numerical comparison, or receipt differs."""


@dataclass
class _StructuralRecorder:
    operators: FlaBackendOperators
    conv_calls: list[dict[str, Any]]
    delta_calls: list[dict[str, Any]]


def _segments(sequence_length: int, device: torch.device) -> torch.Tensor:
    if sequence_length not in SEQUENCE_LENGTHS:
        raise FlaSemanticParityError("semantic parity sequence length differs")
    first = torch.full((sequence_length,), 19, dtype=torch.long)
    second = torch.full((sequence_length,), 19, dtype=torch.long)
    if sequence_length == 64:
        first[:63] = 11
    elif sequence_length == 65:
        first[0] = 7
        first[1:64] = 11
    return torch.stack((first, second)).to(device)


def _metric(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    threshold: float,
) -> dict[str, float | int | bool]:
    if actual.shape != expected.shape or actual.numel() == 0:
        raise FlaSemanticParityError("semantic parity tensor geometry differs")
    actual_float = actual.detach().float()
    expected_float = expected.detach().float()
    difference = actual_float - expected_float
    root_mean_square_error = float(difference.square().mean().sqrt().item())
    reference_root_mean_square = float(expected_float.square().mean().sqrt().item())
    ratio = root_mean_square_error / (reference_root_mean_square + 1e-8)
    max_absolute_error = float(difference.abs().max().item())
    finite = all(
        bool(torch.isfinite(value).all().item())
        for value in (actual_float, expected_float, difference)
    ) and all(
        math.isfinite(value)
        for value in (root_mean_square_error, reference_root_mean_square, ratio)
    )
    return {
        "threshold": threshold,
        "root_mean_square_error": root_mean_square_error,
        "reference_root_mean_square": reference_root_mean_square,
        "relative_root_mean_square_error": ratio,
        "max_absolute_error": max_absolute_error,
        "elements_compared": actual.numel(),
        "all_finite": finite,
        "passed": finite and ratio < threshold,
    }


def _cpu_random(
    generator: torch.Generator, shape: tuple[int, ...], *, scale: float = 1.0
) -> torch.Tensor:
    return torch.randn(*shape, generator=generator) * scale


def _leaf(
    value: torch.Tensor, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return value.to(device=device, dtype=dtype).detach().requires_grad_(True)


def _spans(offsets: list[int]) -> list[tuple[int, int]]:
    if len(offsets) < 2 or offsets[0] != 0:
        raise FlaSemanticParityError("packed semantic offsets differ")
    spans = list(zip(offsets[:-1], offsets[1:], strict=True))
    if any(start >= stop for start, stop in spans):
        raise FlaSemanticParityError("packed semantic offsets are not increasing")
    return spans


def _reference_conv(
    value: torch.Tensor, weight: torch.Tensor, offsets: list[int]
) -> torch.Tensor:
    outputs = []
    for start, stop in _spans(offsets):
        segment = value[:, start:stop]
        convolved = F.conv1d(
            segment.transpose(1, 2),
            weight.unsqueeze(1),
            padding=weight.shape[-1] - 1,
            groups=value.shape[-1],
        )[..., : stop - start]
        outputs.append(F.silu(convolved.transpose(1, 2)))
    return torch.cat(outputs, dim=1)


def _reference_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    offsets: list[int],
) -> torch.Tensor:
    outputs = []
    for start, stop in _spans(offsets):
        q_segment = q[:, start:stop]
        k_segment = k[:, start:stop]
        v_segment = value[:, start:stop]
        g_segment = g[:, start:stop]
        beta_segment = beta[:, start:stop]
        state = torch.zeros(
            1,
            q.shape[2],
            q.shape[3],
            value.shape[3],
            device=q.device,
            dtype=torch.float32,
        )
        segment_outputs = []
        for index in range(stop - start):
            decay = g_segment[:, index].float().exp()
            if decay.ndim == 2:
                decay = decay.unsqueeze(-1)
            state = decay.unsqueeze(-1) * state
            q_t = q_segment[:, index].float()
            k_t = k_segment[:, index].float()
            v_t = v_segment[:, index].float()
            prediction = torch.einsum("bhk,bhkv->bhv", k_t, state)
            error = beta_segment[:, index].float().unsqueeze(-1) * (v_t - prediction)
            state = state + torch.einsum("bhk,bhv->bhkv", k_t, error)
            segment_outputs.append(torch.einsum("bhk,bhkv->bhv", q_t, state))
        outputs.append(torch.stack(segment_outputs, dim=1).to(value.dtype))
    return torch.cat(outputs, dim=1)


def _unwrap_output(value: Any, field: str) -> torch.Tensor:
    output = value[0] if isinstance(value, tuple) else value
    if not isinstance(output, torch.Tensor):
        raise FlaSemanticParityError(f"{field} operator output differs")
    return output


def _operator_name(operator: Any) -> str:
    module = getattr(operator, "__module__", "unknown")
    name = getattr(operator, "__qualname__", getattr(operator, "__name__", "unknown"))
    return f"{module}.{name}"


def _direct_conv_probe(
    operator: Any,
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    offsets: list[int],
    channels: int,
    label: str,
) -> dict[str, Any]:
    total = offsets[-1]
    base_x = _cpu_random(generator, (1, total, channels), scale=0.2)
    base_weight = _cpu_random(generator, (channels, 4), scale=0.2)
    upstream_x = _leaf(base_x, device=device, dtype=dtype)
    upstream_weight = _leaf(base_weight, device=device, dtype=dtype)
    reference_x = _leaf(base_x, device=device, dtype=dtype)
    reference_weight = _leaf(base_weight, device=device, dtype=dtype)
    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.long)
    upstream_y = _unwrap_output(
        operator(
            x=upstream_x,
            weight=upstream_weight,
            bias=None,
            activation="silu",
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        ),
        "convolution",
    )
    reference_y = _reference_conv(reference_x, reference_weight, offsets)
    upstream_do = _cpu_random(generator, tuple(upstream_y.shape), scale=0.2).to(
        device=device, dtype=dtype
    )
    reference_do = upstream_do.detach().clone()
    (upstream_y * upstream_do).sum().backward()
    (reference_y * reference_do).sum().backward()
    gradients = (
        upstream_x.grad,
        reference_x.grad,
        upstream_weight.grad,
        reference_weight.grad,
    )
    if any(gradient is None for gradient in gradients):
        raise FlaSemanticParityError("convolution gradient is absent")
    metrics = {
        "y": _metric(upstream_y, reference_y, threshold=CONV_THRESHOLDS["y"]),
        "dx": _metric(
            upstream_x.grad, reference_x.grad, threshold=CONV_THRESHOLDS["dx"]
        ),
        "dw": _metric(
            upstream_weight.grad,
            reference_weight.grad,
            threshold=CONV_THRESHOLDS["dw"],
        ),
    }
    return {
        "label": label,
        "metrics": metrics,
        "passed": set(metrics) == set(CONV_THRESHOLDS)
        and all(bool(metric["passed"]) for metric in metrics.values()),
    }


def _direct_recurrence_probe(
    operator: Any,
    *,
    family: str,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    offsets: list[int],
    heads: int,
    key_dim: int,
) -> dict[str, Any]:
    total = offsets[-1]
    shape = (1, total, heads)
    base_q = F.normalize(_cpu_random(generator, (*shape, key_dim)).float(), dim=-1)
    base_k = F.normalize(_cpu_random(generator, (*shape, key_dim)).float(), dim=-1)
    base_v = _cpu_random(generator, (*shape, key_dim), scale=0.2)
    gate_tail = (key_dim,) if family == "kda" else ()
    base_g = -0.05 - 0.15 * F.softplus(_cpu_random(generator, (*shape, *gate_tail)))
    base_beta = torch.sigmoid(_cpu_random(generator, shape))

    def leaves() -> dict[str, torch.Tensor]:
        return {
            "q": _leaf(base_q, device=device, dtype=dtype),
            "k": _leaf(base_k, device=device, dtype=dtype),
            "v": _leaf(base_v, device=device, dtype=dtype),
            "g": _leaf(base_g, device=device, dtype=torch.float32),
            "beta": _leaf(base_beta, device=device, dtype=dtype),
        }

    upstream = leaves()
    reference = leaves()
    kwargs: dict[str, Any] = {
        **upstream,
        "scale": 1.0,
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": False,
        "cu_seqlens": torch.tensor(offsets, device=device, dtype=torch.long),
        "transpose_state_layout": False,
    }
    if family == "kda":
        kwargs.update(
            {
                "use_gate_in_kernel": False,
                "safe_gate": False,
                "disable_recompute": False,
            }
        )
    upstream_o = _unwrap_output(operator(**kwargs), "recurrence")
    reference_o = _reference_recurrence(
        reference["q"],
        reference["k"],
        reference["v"],
        reference["g"],
        reference["beta"],
        offsets,
    )
    upstream_do = _cpu_random(generator, tuple(upstream_o.shape), scale=0.2).to(
        device=device, dtype=dtype
    )
    reference_do = upstream_do.detach().clone()
    (upstream_o * upstream_do).sum().backward()
    (reference_o * reference_do).sum().backward()
    gradient_names = {
        "dq": "q",
        "dk": "k",
        "dv": "v",
        "dg": "g",
        "dbeta": "beta",
    }
    if any(
        upstream[source].grad is None or reference[source].grad is None
        for source in gradient_names.values()
    ):
        raise FlaSemanticParityError("recurrence gradient is absent")
    metrics = {
        "o": _metric(upstream_o, reference_o, threshold=RECURRENCE_THRESHOLDS["o"])
    }
    for metric_name, source in gradient_names.items():
        metrics[metric_name] = _metric(
            upstream[source].grad,
            reference[source].grad,
            threshold=RECURRENCE_THRESHOLDS[metric_name],
        )
    return {
        "metrics": metrics,
        "passed": set(metrics) == set(RECURRENCE_THRESHOLDS)
        and all(bool(metric["passed"]) for metric in metrics.values()),
    }


def _structural_recorder(base: FlaBackendOperators) -> _StructuralRecorder:
    conv_calls: list[dict[str, Any]] = []
    delta_calls: list[dict[str, Any]] = []

    def conv(**kwargs: Any) -> torch.Tensor:
        conv_calls.append(kwargs)
        return kwargs["x"]

    def delta(**kwargs: Any) -> tuple[torch.Tensor, None]:
        delta_calls.append(kwargs)
        return kwargs["v"], None

    return _StructuralRecorder(
        operators=FlaBackendOperators(
            gated_delta_chunk=delta,
            kda_chunk=delta,
            causal_conv1d=conv,
            version=base.version,
        ),
        conv_calls=conv_calls,
        delta_calls=delta_calls,
    )


def _structural_probe(
    base: FlaBackendOperators,
    *,
    family: str,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
    segment_ids: torch.Tensor,
    heads: int,
    key_dim: int,
) -> dict[str, Any]:
    recorder = _structural_recorder(base)
    batch, sequence = segment_ids.shape
    width = heads * key_dim
    value = _cpu_random(generator, (batch, sequence, width), scale=0.2).to(
        device=device, dtype=dtype
    )
    weight = _cpu_random(generator, (width, 1, 4), scale=0.2).to(
        device=device, dtype=torch.float32
    )
    conv_output = fla_causal_conv1d(
        value, weight, segment_ids, operators=recorder.operators
    )
    q = _cpu_random(generator, (batch, sequence, heads, key_dim), scale=0.2).to(
        device=device, dtype=dtype
    )
    k = _cpu_random(generator, (batch, sequence, heads, key_dim), scale=0.2).to(
        device=device, dtype=dtype
    )
    v = _cpu_random(generator, (batch, sequence, heads, key_dim), scale=0.2).to(
        device=device, dtype=dtype
    )
    alpha_shape = (
        (batch, sequence, heads, key_dim)
        if family == "kda"
        else (batch, sequence, heads, 1)
    )
    alpha = torch.sigmoid(
        _cpu_random(generator, alpha_shape).to(device=device, dtype=dtype)
    )
    beta = torch.sigmoid(
        _cpu_random(generator, (batch, sequence, heads, 1)).to(
            device=device, dtype=dtype
        )
    )
    delta_output = fla_delta_recurrence(
        q,
        k,
        v,
        alpha,
        beta,
        segment_ids,
        channel_wise_decay=family == "kda",
        operators=recorder.operators,
    )
    if len(recorder.conv_calls) != 1 or len(recorder.delta_calls) != 1:
        raise FlaSemanticParityError("structural adapter dispatch count differs")
    conv_call = recorder.conv_calls[0]
    delta_call = recorder.delta_calls[0]
    offsets = packed_cu_seqlens(segment_ids).tolist()
    flattened_tokens = batch * sequence
    common = [1, flattened_tokens, heads]
    expected_g_shape = common + ([key_dim] if family == "kda" else [])
    expected_q = (
        F.normalize(q.float(), dim=-1)
        .to(q.dtype)
        .reshape(1, flattened_tokens, heads, key_dim)
    )
    expected_k = (
        F.normalize(k.float(), dim=-1)
        .to(k.dtype)
        .reshape(1, flattened_tokens, heads, key_dim)
    )
    expected_v = v.reshape(1, flattened_tokens, heads, key_dim)
    expected_g = alpha.float().log()
    if family == "gdn":
        expected_g = expected_g.squeeze(-1).reshape(1, flattened_tokens, heads)
    else:
        expected_g = expected_g.reshape(1, flattened_tokens, heads, key_dim)
    expected_beta = beta.squeeze(-1).reshape(1, flattened_tokens, heads)
    if family == "kda":
        family_flags_passed = (
            delta_call.get("use_gate_in_kernel") is False
            and delta_call.get("safe_gate") is False
            and delta_call.get("disable_recompute") is False
        )
    else:
        family_flags_passed = all(
            field not in delta_call
            for field in ("use_gate_in_kernel", "safe_gate", "disable_recompute")
        )
    flags_passed = (
        delta_call.get("transpose_state_layout") is False and family_flags_passed
    )
    passed = (
        torch.equal(conv_output, value)
        and torch.equal(delta_output, v)
        and conv_call["activation"] == "silu"
        and conv_call["bias"] is None
        and conv_call["cu_seqlens"].tolist() == offsets
        and list(conv_call["x"].shape) == [1, flattened_tokens, width]
        and list(conv_call["weight"].shape) == [width, 4]
        and conv_call["weight"].dtype == dtype
        and torch.equal(conv_call["x"], value.reshape(1, flattened_tokens, width))
        and torch.equal(conv_call["weight"], weight[:, 0, :].to(dtype=dtype))
        and delta_call["scale"] == 1.0
        and delta_call["output_final_state"] is False
        and delta_call["use_qk_l2norm_in_kernel"] is False
        and delta_call["cu_seqlens"].tolist() == offsets
        and list(delta_call["q"].shape) == common + [key_dim]
        and list(delta_call["k"].shape) == common + [key_dim]
        and list(delta_call["v"].shape) == common + [key_dim]
        and list(delta_call["g"].shape) == expected_g_shape
        and list(delta_call["beta"].shape) == common
        and torch.equal(delta_call["q"], expected_q)
        and torch.equal(delta_call["k"], expected_k)
        and torch.equal(delta_call["v"], expected_v)
        and torch.equal(delta_call["g"], expected_g)
        and torch.equal(delta_call["beta"], expected_beta)
        and flags_passed
    )
    return {
        "packed_cu_seqlens": offsets,
        "equal_id_across_row_boundary": bool(segment_ids[0, -1] == segment_ids[1, 0]),
        "explicit_scale_one": delta_call["scale"] == 1.0,
        "external_qk_normalization": delta_call["use_qk_l2norm_in_kernel"] is False,
        "family_flags_passed": flags_passed,
        "passed": passed,
    }


def _run_case(
    base: FlaBackendOperators,
    *,
    family: str,
    sequence_length: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    family_offset = 10_000 if family == "kda" else 0
    generator = torch.Generator(device="cpu").manual_seed(
        seed * 100 + family_offset + sequence_length
    )
    segment_ids = _segments(sequence_length, device)
    offsets = packed_cu_seqlens(segment_ids).tolist()
    structural = _structural_probe(
        base,
        family=family,
        generator=generator,
        device=device,
        dtype=dtype,
        segment_ids=segment_ids,
        heads=2,
        key_dim=16,
    )
    convolution = [
        _direct_conv_probe(
            base.causal_conv1d,
            generator=generator,
            device=device,
            dtype=dtype,
            offsets=offsets,
            channels=32,
            label=label,
        )
        for label in ("q", "k", "v")
    ]
    recurrence = _direct_recurrence_probe(
        base.kda_chunk if family == "kda" else base.gated_delta_chunk,
        family=family,
        generator=generator,
        device=device,
        dtype=dtype,
        offsets=offsets,
        heads=2,
        key_dim=16,
    )
    passed = (
        structural["passed"]
        and all(probe["passed"] for probe in convolution)
        and recurrence["passed"]
    )
    return {
        "family": family,
        "sequence_length": sequence_length,
        "structural_mapping": structural,
        "causal_convolution": convolution,
        "packed_recurrence": recurrence,
        "passed": passed,
    }


def run_semantic_parity(
    *,
    seed: int,
    device: torch.device | str = "cuda",
    operators: FlaBackendOperators | None = None,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FlaSemanticParityError("semantic parity seed differs")
    resolved_device = torch.device(device)
    production = operators is None
    if production:
        if seed not in PRODUCTION_SEEDS:
            raise FlaSemanticParityError("production seed is not prospectively frozen")
        if resolved_device.type != "cuda":
            raise FlaSemanticParityError("production semantic parity requires CUDA")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise FlaSemanticParityError("production CUDA BF16 is unavailable")
        operators = load_fla_backend_operators()
    if operators is None or operators.version != FLA_VERSION:
        raise FlaSemanticParityError("semantic parity FLA version differs")
    dtype = torch.bfloat16 if production else torch.float32
    cases = [
        _run_case(
            operators,
            family=family,
            sequence_length=sequence_length,
            seed=seed,
            device=resolved_device,
            dtype=dtype,
        )
        for family in FAMILIES
        for sequence_length in SEQUENCE_LENGTHS
    ]
    family_results = {}
    for family in FAMILIES:
        family_cases = [case for case in cases if case["family"] == family]
        passed = len(family_cases) == len(SEQUENCE_LENGTHS) and all(
            case["passed"] for case in family_cases
        )
        qualified = production and passed
        family_results[family] = {
            "status": (
                "production_semantics_qualified"
                if qualified
                else ("test_oracle_passed" if passed else "semantic_parity_failed")
            ),
            "production_semantics_qualified": qualified,
            "passed_cases": sum(bool(case["passed"]) for case in family_cases),
            "required_cases": len(SEQUENCE_LENGTHS),
        }
    all_families_qualified = production and all(
        result["production_semantics_qualified"] for result in family_results.values()
    )
    cuda_name = None
    cuda_capability = None
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(resolved_device)
        cuda_capability = list(torch.cuda.get_device_capability(resolved_device))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "all_families_production_qualified"
            if all_families_qualified
            else (
                "test_oracle_passed"
                if not production and all(case["passed"] for case in cases)
                else "one_or_more_families_failed"
            )
        ),
        "production_cuda_qualified": all_families_qualified,
        "scope": "direct_packed_fla_semantic_forward_backward_parity",
        "fla_version": operators.version,
        "seed": seed,
        "production_seed_allowlist": list(PRODUCTION_SEEDS),
        "excluded_calibration_seeds": list(CALIBRATION_SEEDS),
        "dtype": "torch.bfloat16" if production else "torch.float32",
        "environment": {
            "device_type": resolved_device.type,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_bf16_supported": (
                torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
            ),
            "cuda_device_name": cuda_name,
            "cuda_capability": cuda_capability,
        },
        "operators": {
            "causal_conv1d": _operator_name(operators.causal_conv1d),
            "gated_delta_chunk": _operator_name(operators.gated_delta_chunk),
            "kda_chunk": _operator_name(operators.kda_chunk),
        },
        "thresholds": {
            "strict_less_than": True,
            "causal_convolution": dict(CONV_THRESHOLDS),
            "packed_recurrence": dict(RECURRENCE_THRESHOLDS),
        },
        "family_results": family_results,
        "cases": cases,
        "checks": {
            "no_cross_case_averaging": True,
            "all_tensors_finite_required": True,
            "structural_mapping_required": True,
            "every_family_case_required": True,
        },
        "optimizer_steps": 0,
        "training_gpu_jobs_submitted": 0,
        "training_authorized": False,
        "architecture_promoted": False,
        "four_b_training_authorized": False,
        "limitations": [
            "semantic_kernel_mapping_only_not_model_quality_evidence",
            "bounded_lengths_1_63_64_65_not_exact_b8_x_2048",
            "one_seed_per_receipt_requires_all_frozen_seeds_separately",
            "does_not_reinterpret_or_replace_v1_receipts",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FlaSemanticParityError("semantic parity output already exists")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise FlaSemanticParityError("semantic parity output parent is unsafe")
    stage = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
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
    parser.add_argument("--seed", type=int, choices=PRODUCTION_SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_semantic_parity(seed=args.seed, device="cuda")
    _write_create_only(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["production_cuda_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
