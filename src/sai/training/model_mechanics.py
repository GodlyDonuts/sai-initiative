"""Bounded full-model H100 optimizer mechanics for the two Sai delta families."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sai.model.config import SaiModelConfig
from sai.model.reference import SaiCausalLM, exact_parameter_count
from sai.training.fla_parity import canonical_sha256, validate_receipt
from sai.training.runner import TrainingRunConfig, build_adamw

SCHEMA = "sai-100m-fla-model-mechanics-v1"
FAMILIES = ("gdn_hybrid", "kda_mla_hybrid")


class ModelMechanicsError(RuntimeError):
    """The full-model mechanics execution or evidence differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(model: SaiCausalLM) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        header = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def load_100m_configs(path: Path) -> dict[str, SaiModelConfig]:
    if not path.is_file() or path.is_symlink():
        raise ModelMechanicsError("geometry artifact is missing or unsafe")
    payload = json.loads(path.read_text())
    rows = payload.get("geometries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ModelMechanicsError("geometry artifact differs")
    selected: dict[str, SaiModelConfig] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("scale") != "100m":
            continue
        family = row.get("mixer_family")
        config = row.get("config")
        if family in FAMILIES and isinstance(config, dict):
            if family in selected:
                raise ModelMechanicsError("100M family geometry is duplicated")
            selected[family] = SaiModelConfig(**config)
    if tuple(selected) != FAMILIES:
        raise ModelMechanicsError("100M delta-family geometry set differs")
    return selected


def _run_family(
    family: str,
    config: SaiModelConfig,
    *,
    seed: int,
    sequence_length: int,
    optimizer_steps: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SaiCausalLM(config, delta_backend="fla").to(
        device="cuda", dtype=torch.bfloat16
    )
    model.train()
    initial_state = _state_sha256(model)
    optimizer = build_adamw(
        model,
        TrainingRunConfig(
            optimizer_steps=optimizer_steps,
            learning_rate=1e-4,
            warmup_steps=0,
            weight_decay=0.1,
            gradient_clip_norm=1.0,
        ),
    )
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, sequence_length),
        generator=generator,
        device="cuda",
    )
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    split = sequence_length // 2
    segment_ids = torch.cat(
        (
            torch.zeros((1, split), dtype=torch.long, device="cuda"),
            torch.ones((1, sequence_length - split), dtype=torch.long, device="cuda"),
        ),
        dim=1,
    )
    target_mask = torch.ones_like(input_ids, dtype=torch.bool)
    target_mask[:, split - 1] = False
    target_mask[:, -1] = False
    losses = []
    gradient_norms = []
    all_parameters_received_gradients = True
    torch.cuda.reset_peak_memory_stats()
    for _ in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, segment_ids)
        loss = F.cross_entropy(
            logits[target_mask].float(), target_ids[target_mask], reduction="mean"
        )
        if not bool(torch.isfinite(loss).item()):
            raise ModelMechanicsError("full-model mechanics loss is nonfinite")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        all_parameters_received_gradients &= bool(gradients) and all(
            gradient is not None for gradient in gradients
        )
        if not all_parameters_received_gradients or not all(
            bool(torch.isfinite(gradient).all().item())
            for gradient in gradients
            if gradient is not None
        ):
            raise ModelMechanicsError("full-model mechanics gradients differ")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(norm).item()):
            raise ModelMechanicsError("full-model gradient norm is nonfinite")
        optimizer.step()
        losses.append(float(loss.detach()))
        gradient_norms.append(float(norm.detach()))
    torch.cuda.synchronize()
    final_state = _state_sha256(model)
    result = {
        "family": family,
        "config": config.as_dict(),
        "parameters": exact_parameter_count(model),
        "sequence_length": sequence_length,
        "segments": 2,
        "valid_targets_per_step": int(target_mask.sum().item()),
        "optimizer_steps": optimizer_steps,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "all_parameters_received_gradients": all_parameters_received_gradients,
        "all_values_finite": all(
            math.isfinite(value) for value in (*losses, *gradient_norms)
        ),
        "initial_state_sha256": initial_state,
        "final_state_sha256": final_state,
        "state_changed": initial_state != final_state,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    result["passed"] = bool(
        result["all_parameters_received_gradients"]
        and result["all_values_finite"]
        and result["state_changed"]
    )
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def run(
    geometry: Path,
    parity_receipt: Path,
    *,
    source_commit: str,
    seed: int = 20260821,
    sequence_length: int = 128,
    optimizer_steps: int = 2,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ModelMechanicsError("H100 CUDA BF16 mechanics are unavailable")
    if not parity_receipt.is_file() or parity_receipt.is_symlink():
        raise ModelMechanicsError("FLA parity receipt is missing or unsafe")
    parity = validate_receipt(json.loads(parity_receipt.read_text()))
    if parity.get("parity_qualified") is not True:
        raise ModelMechanicsError("FLA parity is not qualified")
    configs = load_100m_configs(geometry)
    cases = [
        _run_family(
            family,
            configs[family],
            seed=seed + index,
            sequence_length=sequence_length,
            optimizer_steps=optimizer_steps,
        )
        for index, family in enumerate(FAMILIES)
    ]
    passed = all(case["passed"] for case in cases)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "mechanics_qualified": passed,
        "scope": "synthetic_full_100m_model_optimizer_mechanics_only",
        "scientific_result": False,
        "architecture_promoted": False,
        "four_b_training_authorized": False,
        "source_commit": source_commit,
        "geometry_path": str(geometry.resolve()),
        "geometry_sha256": sha256_file(geometry),
        "parity_receipt_path": str(parity_receipt.resolve()),
        "parity_receipt_file_sha256": sha256_file(parity_receipt),
        "parity_receipt_sha256": parity["receipt_sha256"],
        "seed": seed,
        "sequence_length": sequence_length,
        "optimizer_steps_per_family": optimizer_steps,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_capability": list(torch.cuda.get_device_capability()),
        },
        "cases": cases,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--parity-receipt", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ModelMechanicsError("mechanics output already exists")
    payload = run(args.geometry, args.parity_receipt, source_commit=args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = args.output.with_name(f".{args.output.name}.partial.{os.getpid()}")
    stage.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(stage, args.output)


if __name__ == "__main__":
    main()
