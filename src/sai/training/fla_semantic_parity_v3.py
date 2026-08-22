"""Prospective dual-lane FLA 0.4.2 mechanics qualification.

V3 preserves every v1/v2 byte and receipt.  Lane A applies the acceptance
protocol supported by upstream commit ca910f8 to FP32/FP16 convolution.  Lane B
tests Sai's actual BF16 convolution against both segmented Torch BF16 and a
pure FP64 mathematical oracle, without fitting a new scalar error threshold.
The unchanged v2 packed-recurrence and structural gates remain mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sai.data.token_stream import canonical_sha256
from sai.model.fla_backend import (
    FLA_VERSION,
    FlaBackendOperators,
    load_fla_backend_operators,
    packed_cu_seqlens,
)
from sai.training.fla_semantic_parity import (
    FAMILIES,
    RECURRENCE_THRESHOLDS,
    SEQUENCE_LENGTHS,
    FlaSemanticParityError,
    _cpu_random,
    _direct_recurrence_probe,
    _leaf,
    _metric,
    _operator_name,
    _segments,
    _spans,
    _structural_probe,
    _unwrap_output,
)

SCHEMA = "sai-fla-semantic-parity-v3"
PINNED_UPSTREAM_COMMIT = "ca910f8"
PRODUCTION_SEEDS = (20260827, 20260828, 20260829)
LANE_A_THRESHOLD = 0.001
QKV_PATHS = ("q", "k", "v")
TENSORS = ("y", "dx", "dw")


def _production_fla_provenance(
    root: Path, operators: FlaBackendOperators
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise FlaSemanticParityError("v3 FLA source root is unsafe")
    resolved = root.resolve(strict=True)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", "-C", str(resolved), *arguments),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise FlaSemanticParityError("v3 FLA source identity is unavailable")
        return completed.stdout.strip()

    head = git("rev-parse", "HEAD")
    if not head.startswith(PINNED_UPSTREAM_COMMIT):
        raise FlaSemanticParityError("v3 FLA source commit differs")
    if git("status", "--short"):
        raise FlaSemanticParityError("v3 FLA source tree is not clean")
    sources = {}
    named_operators = {
        "causal_conv1d": operators.causal_conv1d,
        "gated_delta_chunk": operators.gated_delta_chunk,
        "kda_chunk": operators.kda_chunk,
    }
    for name, operator in named_operators.items():
        source_name = inspect.getsourcefile(operator)
        if source_name is None:
            raise FlaSemanticParityError("v3 FLA operator source is unavailable")
        source = Path(source_name).resolve(strict=True)
        try:
            relative = source.relative_to(resolved)
        except ValueError as error:
            raise FlaSemanticParityError(
                "v3 FLA operator is outside the pinned source root"
            ) from error
        sources[name] = {
            "relative_path": relative.as_posix(),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    return {
        "root": str(resolved),
        "git_head": head,
        "git_tree_clean": True,
        "operator_sources": sources,
    }


def _reference_conv_activation(
    value: torch.Tensor,
    weight: torch.Tensor,
    offsets: list[int],
    *,
    activation: str | None,
) -> torch.Tensor:
    outputs = []
    for start, stop in _spans(offsets):
        segment = value[:, start:stop]
        convolved = F.conv1d(
            segment.transpose(1, 2),
            weight.unsqueeze(1),
            padding=weight.shape[-1] - 1,
            groups=value.shape[-1],
        )[..., : stop - start].transpose(1, 2)
        outputs.append(
            F.silu(convolved) if activation in ("silu", "swish") else convolved
        )
    return torch.cat(outputs, dim=1)


def _require_gradients(*values: torch.Tensor | None) -> tuple[torch.Tensor, ...]:
    if any(value is None for value in values):
        raise FlaSemanticParityError("v3 convolution gradient is absent")
    return tuple(value for value in values if value is not None)


def _lane_a_probe(
    operator: Any,
    *,
    base_x: torch.Tensor,
    base_weight: torch.Tensor,
    base_do: torch.Tensor,
    device: torch.device,
    offsets: list[int],
    dtype: torch.dtype,
    activation: str | None,
) -> dict[str, Any]:
    upstream_x = _leaf(base_x, device=device, dtype=dtype)
    upstream_weight = _leaf(base_weight, device=device, dtype=dtype)
    torch_x = _leaf(base_x, device=device, dtype=dtype)
    torch_weight = _leaf(base_weight, device=device, dtype=dtype)
    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.long)
    upstream_y = _unwrap_output(
        operator(
            x=upstream_x,
            weight=upstream_weight,
            bias=None,
            activation=activation,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        ),
        "v3 lane A convolution",
    )
    torch_y = _reference_conv_activation(
        torch_x, torch_weight, offsets, activation=activation
    )
    upstream_do = base_do.to(device=device, dtype=dtype)
    torch_do = upstream_do.detach().clone()
    (upstream_y * upstream_do).sum().backward()
    (torch_y * torch_do).sum().backward()
    upstream_dx, torch_dx, upstream_dw, torch_dw = _require_gradients(
        upstream_x.grad, torch_x.grad, upstream_weight.grad, torch_weight.grad
    )
    metrics = {
        "y": _metric(upstream_y, torch_y, threshold=LANE_A_THRESHOLD),
        "dx": _metric(upstream_dx, torch_dx, threshold=LANE_A_THRESHOLD),
        "dw": _metric(upstream_dw, torch_dw, threshold=LANE_A_THRESHOLD),
    }
    return {
        "dtype": str(dtype),
        "activation": activation,
        "upstream_protocol": {
            "commit": PINNED_UPSTREAM_COMMIT,
            "strict_less_than": True,
            "relative_rmse_threshold": LANE_A_THRESHOLD,
            "bias": None,
            "output_final_state": False,
        },
        "bf16_qualification": False,
        "metrics": metrics,
        "passed": set(metrics) == set(TENSORS)
        and all(bool(metric["passed"]) for metric in metrics.values()),
    }


def _ordered_bf16(value: torch.Tensor) -> torch.Tensor:
    """Map finite BF16 values to consecutive monotone integers.

    The duplicate signed-zero encoding is collapsed.  Consecutive negative
    subnormal, zero, and positive subnormal values consequently remain one
    representable rounding step apart.
    """

    if value.dtype != torch.bfloat16:
        raise FlaSemanticParityError("ordered-ULP input is not BF16")
    cpu = value.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(cpu).all().item()):
        raise FlaSemanticParityError("ordered-ULP input is not finite")
    raw = cpu.view(torch.int16).to(torch.int32).bitwise_and(0xFFFF)
    magnitude = raw.bitwise_and(0x7FFF)
    negative = raw.bitwise_and(0x8000).ne(0)
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def _bf16_half_ulp(oracle: torch.Tensor) -> torch.Tensor:
    """Return a conservative half-ULP around each finite FP64 oracle value."""

    if oracle.dtype != torch.float64:
        raise FlaSemanticParityError("ULP oracle is not FP64")
    if not bool(torch.isfinite(oracle).all().item()):
        raise FlaSemanticParityError("ULP oracle is not finite")
    absolute = oracle.abs()
    minimum_normal = float(torch.finfo(torch.bfloat16).tiny)
    minimum_subnormal = math.ldexp(1.0, -133)
    safe = absolute.clamp_min(minimum_normal)
    normal_ulp = torch.pow(
        torch.tensor(2.0, dtype=torch.float64), torch.floor(torch.log2(safe)) - 7
    )
    ulp = torch.where(
        absolute < minimum_normal,
        torch.full_like(absolute, minimum_subnormal),
        normal_ulp,
    )
    return ulp * 0.5


def _noninferiority_metric(
    fla: torch.Tensor,
    torch_value: torch.Tensor,
    oracle: torch.Tensor,
) -> dict[str, Any]:
    if fla.shape != torch_value.shape or fla.shape != oracle.shape or not fla.numel():
        raise FlaSemanticParityError("v3 BF16 comparison geometry differs")
    if fla.dtype != torch.bfloat16 or torch_value.dtype != torch.bfloat16:
        raise FlaSemanticParityError("v3 execution comparison is not BF16")
    if oracle.dtype != torch.float64:
        raise FlaSemanticParityError("v3 mathematical oracle is not FP64")
    fla_cpu = fla.detach().to(device="cpu")
    torch_cpu = torch_value.detach().to(device="cpu")
    oracle_cpu = oracle.detach().to(device="cpu")
    finite = all(
        bool(torch.isfinite(value).all().item())
        for value in (fla_cpu, torch_cpu, oracle_cpu)
    )
    if not finite:
        return {
            "elements_compared": fla.numel(),
            "all_finite": False,
            "elementwise_envelope_passed": False,
            "rms_envelope_passed": False,
            "ordered_bf16_ulp_passed": False,
            "passed": False,
        }
    fla_error = (fla_cpu.double() - oracle_cpu).abs()
    torch_error = (torch_cpu.double() - oracle_cpu).abs()
    half_ulp = _bf16_half_ulp(oracle_cpu)
    # This machine-epsilon term only covers FP64 arithmetic in the comparison;
    # it is not a fitted kernel tolerance and is many orders below a BF16 ULP.
    arithmetic_slack = torch.finfo(torch.float64).eps * oracle_cpu.abs().clamp_min(1.0)
    elementwise = fla_error <= torch_error + half_ulp + arithmetic_slack
    fla_rms = float(fla_error.square().mean().sqrt().item())
    torch_rms = float(torch_error.square().mean().sqrt().item())
    half_ulp_rms = float(half_ulp.square().mean().sqrt().item())
    arithmetic_slack_rms = float(arithmetic_slack.square().mean().sqrt().item())
    rms_passed = fla_rms <= torch_rms + half_ulp_rms + arithmetic_slack_rms
    ideal = oracle_cpu.to(torch.bfloat16)
    fla_distance = (_ordered_bf16(fla_cpu) - _ordered_bf16(ideal)).abs()
    torch_distance = (_ordered_bf16(torch_cpu) - _ordered_bf16(ideal)).abs()
    ulp_passed = fla_distance <= torch_distance + 1
    passed = (
        bool(elementwise.all().item()) and rms_passed and bool(ulp_passed.all().item())
    )
    return {
        "elements_compared": fla.numel(),
        "all_finite": True,
        "max_absolute_error_fla": float(fla_error.max().item()),
        "max_absolute_error_torch": float(torch_error.max().item()),
        "root_mean_square_error_fla": fla_rms,
        "root_mean_square_error_torch": torch_rms,
        "root_mean_square_half_bf16_ulp": half_ulp_rms,
        "max_ordered_bf16_ulp_fla": int(fla_distance.max().item()),
        "max_ordered_bf16_ulp_torch": int(torch_distance.max().item()),
        "elementwise_envelope_passed": bool(elementwise.all().item()),
        "rms_envelope_passed": rms_passed,
        "ordered_bf16_ulp_passed": bool(ulp_passed.all().item()),
        "passed": passed,
    }


def _lane_b_probe(
    operator: Any,
    *,
    generator: torch.Generator,
    device: torch.device,
    offsets: list[int],
    channels: int,
) -> dict[str, Any]:
    total = offsets[-1]
    # Generation and the sole precision reduction happen on CPU.  Every lane
    # below starts from exact clones of these immutable BF16 values.
    base_x = _cpu_random(generator, (1, total, channels), scale=0.2).to(torch.bfloat16)
    base_weight = _cpu_random(generator, (channels, 4), scale=0.2).to(torch.bfloat16)
    base_do = _cpu_random(generator, (1, total, channels), scale=0.2).to(torch.bfloat16)

    fla_x = _leaf(base_x, device=device, dtype=torch.bfloat16)
    fla_weight = _leaf(base_weight, device=device, dtype=torch.bfloat16)
    torch_x = _leaf(base_x, device=device, dtype=torch.bfloat16)
    torch_weight = _leaf(base_weight, device=device, dtype=torch.bfloat16)
    oracle_x = base_x.double().detach().requires_grad_(True)
    oracle_weight = base_weight.double().detach().requires_grad_(True)

    cu_seqlens = torch.tensor(offsets, device=device, dtype=torch.long)
    fla_y = _unwrap_output(
        operator(
            x=fla_x,
            weight=fla_weight,
            bias=None,
            activation="silu",
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        ),
        "v3 lane B convolution",
    )
    torch_y = _reference_conv_activation(
        torch_x, torch_weight, offsets, activation="silu"
    )
    oracle_y = _reference_conv_activation(
        oracle_x, oracle_weight, offsets, activation="silu"
    )
    fla_do = base_do.to(device=device)
    torch_do = fla_do.detach().clone()
    oracle_do = base_do.double()
    (fla_y * fla_do).sum().backward()
    (torch_y * torch_do).sum().backward()
    (oracle_y * oracle_do).sum().backward()
    fla_dx, torch_dx, oracle_dx, fla_dw, torch_dw, oracle_dw = _require_gradients(
        fla_x.grad,
        torch_x.grad,
        oracle_x.grad,
        fla_weight.grad,
        torch_weight.grad,
        oracle_weight.grad,
    )
    metrics = {
        "y": _noninferiority_metric(fla_y, torch_y, oracle_y),
        "dx": _noninferiority_metric(fla_dx, torch_dx, oracle_dx),
        "dw": _noninferiority_metric(fla_dw, torch_dw, oracle_dw),
    }
    return {
        "generated_on": "cpu",
        "quantized_once_to": "torch.bfloat16",
        "execution_dtype": "torch.bfloat16",
        "oracle_dtype": "torch.float64",
        "activation": "silu",
        "offsets": offsets,
        "acceptance": {
            "elementwise": (
                "abs(fla-oracle) <= abs(torch-oracle) + 0.5*bf16_ulp(oracle)"
            ),
            "rms": "rms(fla-oracle) <= rms(torch-oracle) + rms(0.5*bf16_ulp(oracle))",
            "ordered_bf16_ulp": (
                "distance(fla,round_bf16(oracle)) <= "
                "distance(torch,round_bf16(oracle)) + 1"
            ),
            "fitted_scalar_threshold": None,
        },
        "metrics": metrics,
        "passed": set(metrics) == set(TENSORS)
        and all(bool(metric["passed"]) for metric in metrics.values()),
    }


def _run_case(
    operators: FlaBackendOperators,
    *,
    family: str,
    sequence_length: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    family_offset = 10_000 if family == "kda" else 0
    generator = torch.Generator(device="cpu").manual_seed(
        seed * 100 + family_offset + sequence_length
    )
    segment_ids = _segments(sequence_length, device)
    offsets = packed_cu_seqlens(segment_ids).tolist()
    structural = _structural_probe(
        operators,
        family=family,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
        segment_ids=segment_ids,
        heads=2,
        key_dim=16,
    )
    lane_a = []
    lane_b = []
    for label in QKV_PATHS:
        base_x = _cpu_random(generator, (1, offsets[-1], 32), scale=0.2)
        base_weight = _cpu_random(generator, (32, 4), scale=0.2)
        base_do = _cpu_random(generator, (1, offsets[-1], 32), scale=0.2)
        probes = [
            _lane_a_probe(
                operators.causal_conv1d,
                base_x=base_x,
                base_weight=base_weight,
                base_do=base_do,
                device=device,
                offsets=offsets,
                dtype=torch.float32,
                activation="swish",
            ),
            _lane_a_probe(
                operators.causal_conv1d,
                base_x=base_x,
                base_weight=base_weight,
                base_do=base_do,
                device=device,
                offsets=offsets,
                dtype=torch.float16,
                activation=None,
            ),
        ]
        lane_a.append(
            {
                "label": label,
                "probes": probes,
                "passed": all(probe["passed"] for probe in probes),
            }
        )
        lane_b.append(
            {
                "label": label,
                **_lane_b_probe(
                    operators.causal_conv1d,
                    generator=generator,
                    device=device,
                    offsets=offsets,
                    channels=32,
                ),
            }
        )
    recurrence = _direct_recurrence_probe(
        operators.kda_chunk if family == "kda" else operators.gated_delta_chunk,
        family=family,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
        offsets=offsets,
        heads=2,
        key_dim=16,
    )
    passed = (
        bool(structural["passed"])
        and all(probe["passed"] for probe in lane_a)
        and all(probe["passed"] for probe in lane_b)
        and bool(recurrence["passed"])
    )
    return {
        "family": family,
        "sequence_length": sequence_length,
        "qkv_paths": list(QKV_PATHS),
        "structural_mapping": structural,
        "lane_a_upstream_fp32_fp16": lane_a,
        "lane_b_bf16_noninferiority": lane_b,
        "packed_recurrence": recurrence,
        "passed": passed,
    }


def run_semantic_parity_v3(
    *,
    seed: int,
    device: torch.device | str = "cuda",
    operators: FlaBackendOperators | None = None,
    fla_root: Path | None = None,
) -> dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FlaSemanticParityError("v3 semantic parity seed differs")
    resolved_device = torch.device(device)
    production = operators is None
    if production:
        if seed not in PRODUCTION_SEEDS:
            raise FlaSemanticParityError(
                "v3 production seed is not prospectively frozen"
            )
        if resolved_device.type != "cuda":
            raise FlaSemanticParityError("v3 production semantic parity requires CUDA")
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise FlaSemanticParityError("v3 production CUDA BF16 is unavailable")
        operators = load_fla_backend_operators()
    if operators is None or operators.version != FLA_VERSION:
        raise FlaSemanticParityError("v3 semantic parity FLA version differs")
    if production:
        if fla_root is None:
            raise FlaSemanticParityError("v3 production FLA source root is required")
        fla_provenance = _production_fla_provenance(fla_root, operators)
    else:
        fla_provenance = None
    cases = [
        _run_case(
            operators,
            family=family,
            sequence_length=sequence_length,
            seed=seed,
            device=resolved_device,
        )
        for family in FAMILIES
        for sequence_length in SEQUENCE_LENGTHS
    ]
    family_results: dict[str, dict[str, Any]] = {}
    for family in FAMILIES:
        family_cases = [case for case in cases if case["family"] == family]
        passed = len(family_cases) == len(SEQUENCE_LENGTHS) and all(
            bool(case["passed"]) for case in family_cases
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
    all_qualified = production and all(
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
            if all_qualified
            else (
                "test_oracle_passed"
                if not production and all(bool(case["passed"]) for case in cases)
                else "one_or_more_families_failed"
            )
        ),
        "production_cuda_qualified": all_qualified,
        "scope": "dual_lane_packed_fla_mechanics_only",
        "seed": seed,
        "production_seed_allowlist": list(PRODUCTION_SEEDS),
        "fla_version": operators.version,
        "pinned_upstream_commit": PINNED_UPSTREAM_COMMIT,
        "fla_source_provenance": fla_provenance,
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
        "acceptance": {
            "lane_a": {
                "strict_relative_rmse_less_than": LANE_A_THRESHOLD,
                "fp32_activation": "swish",
                "fp16_activation": None,
                "bf16_qualification": False,
            },
            "lane_b": {
                "elementwise_noninferiority": True,
                "rms_noninferiority": True,
                "maximum_extra_ordered_bf16_rounding_steps": 1,
                "fitted_scalar_threshold": None,
            },
            "packed_recurrence": dict(RECURRENCE_THRESHOLDS),
            "no_cross_case_or_seed_averaging": True,
        },
        "family_results": family_results,
        "cases": cases,
        "checks": {
            "all_tensors_finite_required": True,
            "every_lane_tensor_qkv_length_family_seed_required": True,
            "structural_mapping_required": True,
            "packed_resets_required": True,
            "equal_ids_across_row_boundary_included": True,
            "explicit_scale_one_required": True,
        },
        "optimizer_steps": 0,
        "training_gpu_jobs_submitted": 0,
        "training_authorized": False,
        "architecture_promoted": False,
        "four_b_training_authorized": False,
        "mechanics_canary_admitted": all_qualified,
        "mechanics_canary_scope": "exact_b8_x_2048_one_update_hybrid_only",
        "long_training_authorized": False,
        "limitations": [
            "mechanics_only_not_model_quality_or_benchmark_evidence",
            "lane_a_fp32_fp16_is_not_bf16_qualification",
            "bounded_lengths_1_63_64_65_precede_exact_b8_x_2048_canary",
            "each_frozen_seed_requires_a_separate_nonreplaceable_receipt",
            "does_not_reinterpret_or_replace_v1_or_v2_receipts",
            "v3_pass_admits_only_the_exact_one_update_hybrid_mechanics_canary",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FlaSemanticParityError("v3 semantic parity output already exists")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise FlaSemanticParityError("v3 semantic parity output parent is unsafe")
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
    parser.add_argument("--fla-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_semantic_parity_v3(
        seed=args.seed, device="cuda", fla_root=args.fla_root
    )
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
