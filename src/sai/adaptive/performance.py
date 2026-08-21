"""No-training mechanics and diagnostic timing for the latent workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

from sai.adaptive.config import (
    WorkspaceConfig,
    workspace_forward_flop_ledger,
    workspace_parameter_ledger,
)
from sai.adaptive.reference import LatentWorkspace
from sai.model.reference import exact_parameter_count

SCHEMA = "sai-workspace-performance-receipt-v1"


class WorkspacePerformanceError(RuntimeError):
    """A no-training mechanics measurement differs or overclaims qualification."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    payload = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "byteorder": sys.byteorder,
        "raw_sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
    }
    return canonical_sha256(payload)


def state_sha256(module: torch.nn.Module) -> str:
    rows = []
    for name, tensor in sorted(module.state_dict().items()):
        rows.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "raw_sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
            }
        )
    return canonical_sha256(rows)


def _versions(module: torch.nn.Module) -> dict[str, int]:
    return {name: parameter._version for name, parameter in module.named_parameters()}


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def run_cpu_mechanics(
    config: WorkspaceConfig,
    *,
    sequence_length: int,
    iterations: int,
    warmups: int = 2,
    samples: int = 5,
    seed: int = 20260821,
) -> dict[str, Any]:
    """Measure the incremental reference workspace without training or CUDA."""

    for value, field, minimum in (
        (sequence_length, "sequence length", 1),
        (iterations, "iterations", 1),
        (warmups, "warmups", 0),
        (samples, "samples", 3),
        (seed, "seed", 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise WorkspacePerformanceError(f"{field} differs")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        workspace = LatentWorkspace(config).eval()
    for parameter in workspace.parameters():
        parameter.requires_grad_(False)
    context = torch.randn(
        1,
        sequence_length,
        config.hidden_size,
        dtype=torch.float32,
        generator=generator,
    )
    mask = torch.ones(1, sequence_length, dtype=torch.bool)
    state_before = state_sha256(workspace)
    versions_before = _versions(workspace)
    rng_before = tensor_sha256(torch.random.get_rng_state())
    call_counts = {"compiler": 0, "reactor": 0, "reader": 0}

    def count(name: str):
        def hook(_module: torch.nn.Module, _inputs: Any, _output: Any) -> None:
            call_counts[name] += 1

        return hook

    hooks = [
        workspace.compiler.register_forward_hook(count("compiler")),
        *[block.register_forward_hook(count("reactor")) for block in workspace.reactor],
        workspace.reader.register_forward_hook(count("reader")),
    ]
    raw_ns = []
    output_hashes = []
    try:
        with torch.inference_mode():
            for _ in range(warmups):
                workspace(
                    context,
                    iterations=iterations,
                    context_mask=mask,
                    return_diagnostics=False,
                )
            warmup_counts = dict(call_counts)
            call_counts = {key: 0 for key in call_counts}
            for _ in range(samples):
                started = time.perf_counter_ns()
                output = workspace(
                    context,
                    iterations=iterations,
                    context_mask=mask,
                    return_diagnostics=False,
                )
                raw_ns.append(time.perf_counter_ns() - started)
                output_hashes.append(tensor_sha256(output))
    finally:
        for hook in hooks:
            hook.remove()
    state_after = state_sha256(workspace)
    versions_after = _versions(workspace)
    rng_after = tensor_sha256(torch.random.get_rng_state())
    expected_counts = {
        "compiler": samples,
        "reactor": samples * iterations * config.reactor_layers,
        "reader": samples,
    }
    expected_warmup_counts = {
        "compiler": warmups,
        "reactor": warmups * iterations * config.reactor_layers,
        "reader": warmups,
    }
    valid = (
        len(set(output_hashes)) == 1
        and call_counts == expected_counts
        and warmup_counts == expected_warmup_counts
        and state_before == state_after
        and versions_before == versions_after
        and rng_before == rng_after
        and all(parameter.grad is None for parameter in workspace.parameters())
        and all(math.isfinite(value) and value > 0 for value in raw_ns)
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete" if valid else "failed_mechanics",
        "measurement_receipt_valid": valid,
        "design_performance_gate_pass": None,
        "production_qualified": False,
        "scope": "incremental_workspace_reference_cpu_diagnostic",
        "end_to_end_production_latency_qualified": False,
        "dram_traffic_measured": False,
        "memory_traffic_qualified": False,
        "training_hold": True,
        "training_authorized": False,
        "gpu_jobs_submitted": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "device": "cpu",
        "dtype": "torch.float32",
        "seed": seed,
        "config": config.as_dict(),
        "config_sha256": canonical_sha256(config.as_dict()),
        "sequence_length": sequence_length,
        "iterations": iterations,
        "warmups": warmups,
        "samples": samples,
        "context_sha256": tensor_sha256(context),
        "mask_sha256": tensor_sha256(mask),
        "output_sha256": output_hashes[0] if output_hashes else None,
        "state_before_sha256": state_before,
        "state_after_sha256": state_after,
        "model_state_unchanged": state_before == state_after,
        "parameter_versions_unchanged": versions_before == versions_after,
        "rng_state_unchanged": rng_before == rng_after,
        "no_parameter_gradients": all(
            parameter.grad is None for parameter in workspace.parameters()
        ),
        "module_parameters": exact_parameter_count(workspace),
        "analytical_parameters": workspace_parameter_ledger(config)["total"],
        "analytical_flops": workspace_forward_flop_ledger(
            config, sequence_length, iterations
        ),
        "call_counts": call_counts,
        "expected_call_counts": expected_counts,
        "raw_latency_ns": raw_ns,
        "latency_ns": {
            "median": statistics.median(raw_ns),
            "p05": _percentile(raw_ns, 0.05),
            "p95": _percentile(raw_ns, 0.95),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        },
        "limitations": [
            "diagnostic_cpu_timing_not_production_latency",
            "reference_backbone_has_no_decode_cache",
            "cuda_allocator_peak_not_measured",
            "hbm_traffic_requires_external_hardware_counters",
            "performance_thresholds_not_yet_predeclared",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def validate_receipt(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise WorkspacePerformanceError("performance receipt must be an object")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "complete"
        or payload.get("measurement_receipt_valid") is not True
        or payload.get("design_performance_gate_pass") is not None
        or payload.get("production_qualified") is not False
        or payload.get("scope") != "incremental_workspace_reference_cpu_diagnostic"
        or payload.get("end_to_end_production_latency_qualified") is not False
        or payload.get("dram_traffic_measured") is not False
        or payload.get("memory_traffic_qualified") is not False
        or payload.get("training_hold") is not True
        or payload.get("training_authorized") is not False
        or payload.get("gpu_jobs_submitted") != 0
        or payload.get("optimizer_steps") != 0
        or payload.get("backward_calls") != 0
        or payload.get("model_state_unchanged") is not True
        or payload.get("parameter_versions_unchanged") is not True
        or payload.get("rng_state_unchanged") is not True
        or payload.get("no_parameter_gradients") is not True
        or payload.get("module_parameters") != payload.get("analytical_parameters")
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise WorkspacePerformanceError("performance receipt boundary differs")
    samples = payload.get("samples")
    raw = payload.get("raw_latency_ns")
    counts = payload.get("call_counts")
    expected = payload.get("expected_call_counts")
    try:
        config = WorkspaceConfig(**payload.get("config", {}))
    except (TypeError, ValueError) as error:
        raise WorkspacePerformanceError("performance configuration differs") from error
    sequence_length = payload.get("sequence_length")
    iterations = payload.get("iterations")
    warmups = payload.get("warmups")
    for value, field, minimum in (
        (sequence_length, "sequence length", 1),
        (iterations, "iterations", 1),
        (warmups, "warmups", 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise WorkspacePerformanceError(f"{field} differs")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples < 3
        or not isinstance(raw, list)
        or len(raw) != samples
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in raw
        )
        or counts != expected
        or payload.get("config_sha256") != canonical_sha256(config.as_dict())
        or payload.get("analytical_parameters")
        != workspace_parameter_ledger(config)["total"]
        or payload.get("analytical_flops")
        != workspace_forward_flop_ledger(config, sequence_length, iterations)
        or expected
        != {
            "compiler": samples,
            "reactor": samples * iterations * config.reactor_layers,
            "reader": samples,
        }
        or payload.get("latency_ns")
        != {
            "median": statistics.median(raw),
            "p05": _percentile(raw, 0.05),
            "p95": _percentile(raw, 0.95),
        }
        or any(
            not isinstance(payload.get(field), str) or len(payload[field]) != 64
            for field in (
                "context_sha256",
                "mask_sha256",
                "output_sha256",
                "state_before_sha256",
                "state_after_sha256",
            )
        )
        or not isinstance(payload.get("limitations"), list)
        or "hbm_traffic_requires_external_hardware_counters"
        not in payload["limitations"]
    ):
        raise WorkspacePerformanceError("performance samples or limitations differ")
    return payload


def write_receipt(payload: dict[str, Any], output: Path) -> None:
    validate_receipt(payload)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--workspace-size", type=int, required=True)
    parser.add_argument("--num-slots", type=int, default=16)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--reactor-layers", type=int, default=4)
    parser.add_argument("--reactor-intermediate-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = WorkspaceConfig(
        hidden_size=args.hidden_size,
        workspace_size=args.workspace_size,
        num_slots=args.num_slots,
        num_heads=args.num_heads,
        reactor_layers=args.reactor_layers,
        reactor_intermediate_size=args.reactor_intermediate_size,
    )
    payload = run_cpu_mechanics(
        config,
        sequence_length=args.sequence_length,
        iterations=args.iterations,
        warmups=args.warmups,
        samples=args.samples,
        seed=args.seed,
    )
    write_receipt(payload, args.output)
    print(
        json.dumps(
            {
                "receipt_sha256": payload["receipt_sha256"],
                "measurement_receipt_valid": payload["measurement_receipt_valid"],
                "production_qualified": payload["production_qualified"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
