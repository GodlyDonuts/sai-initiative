"""Qualify the exact immutable SmolLM3-3B cross-family parent on one H100."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import platform
import uuid
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sai.data.external_hf_snapshot import (
    ExternalSnapshotSpec,
    validate_external_snapshot,
)
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-smollm3-3b-text-mechanics-v1"
EXPECTED_MODEL_CLASS = "SmolLM3ForCausalLM"
EXPECTED_PARAMETER_COUNT = 3_075_098_624
EXPECTED_VOCAB_SIZE = 128_256
SNAPSHOT_SPEC = ExternalSnapshotSpec(
    repository="HuggingFaceTB/SmolLM3-3B",
    revision="a07cc9a04f16550a088caea529712d1d335b0ac1",
    tree_sha256="6badcd593aee3052e3d66afb315b979e2cc62c4a61f9cef31c07203912478a0f",
    manifest_sha256="e689bcce197b02c4d2e8b600696ec3137b1e1724104954cc1735d5d8848e6945",
    receipt_sha256="4672fc549809d89f0489a5e82045d54d3b5580718dcf40631a31807fd7415c85",
)


class SmolParentError(RuntimeError):
    """The exact Smol parent, CUDA load, or mechanics evidence differs."""


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise SmolParentError("Smol runtime artifact is missing or unsafe")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_versions(module: nn.Module) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, value._version)
        for name, value in sorted(module.state_dict(keep_vars=True).items())
    )


def _cuda_residency(module: nn.Module) -> dict[str, Any]:
    parameters = list(module.named_parameters())
    buffers = list(module.named_buffers())
    wrong = [
        name
        for name, value in [*parameters, *buffers]
        if value.device.type != "cuda" or value.device.index not in {0, None}
    ]
    if not parameters or wrong:
        raise SmolParentError("Smol parent is not entirely resident on CUDA:0")
    return {
        "parameter_tensors": len(parameters),
        "buffer_tensors": len(buffers),
        "parameter_count": sum(value.numel() for _, value in parameters),
        "parameter_dtypes": sorted({str(value.dtype) for _, value in parameters}),
        "all_parameters_and_buffers_cuda_zero": True,
    }


def load_smol_parent(
    model_root: Path,
    *,
    manifest_path: Path,
    restoration_receipt_path: Path,
) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load only the causal text path and bind the sealed external snapshot."""

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
    ):
        raise SmolParentError("exactly one CUDA BF16 GPU is required")
    snapshot = validate_external_snapshot(
        model_root,
        manifest_path=manifest_path,
        receipt_path=restoration_receipt_path,
        spec=SNAPSHOT_SPEC,
    )
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SmolParentError("Transformers Smol runtime is unavailable") from error
    tokenizer = AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, trust_remote_code=False
    )
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise SmolParentError("Smol tokenizer vocabulary differs")
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
        output_loading_info=True,
    )
    if type(model).__name__ != EXPECTED_MODEL_CLASS:
        raise SmolParentError("Smol text model class differs")
    if not isinstance(loading, dict) or any(
        loading.get(field)
        for field in (
            "missing_keys",
            "mismatched_keys",
            "unexpected_keys",
            "error_msgs",
        )
    ):
        raise SmolParentError("Smol text model load is incomplete")
    residency = _cuda_residency(model)
    if residency["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise SmolParentError("Smol text parameter count differs")
    source = Path(inspect.getsourcefile(type(model)) or "")
    return (
        model,
        tokenizer,
        {
            "snapshot": snapshot,
            "model_class": type(model).__name__,
            "transformers_version": transformers.__version__,
            "transformers_model_source_sha256": _sha256_file(source),
            **residency,
        },
    )


def qualify_smol_parent(
    model_root: Path,
    *,
    manifest_path: Path,
    restoration_receipt_path: Path,
) -> dict[str, Any]:
    """Run a deterministic text-only forward without training or mutation."""

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if "H100" not in device_name:
        raise SmolParentError("Smol mechanics requires an H100")
    model, tokenizer, runtime = load_smol_parent(
        model_root,
        manifest_path=manifest_path,
        restoration_receipt_path=restoration_receipt_path,
    )
    prompt = "Sai must transfer only benchmark-proven mechanisms across model families"
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids or any(not isinstance(value, int) for value in token_ids):
        raise SmolParentError("Smol mechanics tokenization differs")
    inputs = torch.tensor([token_ids], dtype=torch.long, device="cuda:0")
    versions_before = _tensor_versions(model)
    original_training = model.training
    torch.cuda.reset_peak_memory_stats(0)
    model.eval()
    try:
        with torch.inference_mode():
            output = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                use_cache=False,
                logits_to_keep=1,
            )
            logits = getattr(output, "logits", None)
            if (
                not isinstance(logits, torch.Tensor)
                or logits.shape != (1, 1, EXPECTED_VOCAB_SIZE)
                or not bool(torch.isfinite(logits).all().item())
            ):
                raise SmolParentError("Smol mechanics logits differ")
            checksum = float(logits.float().sum().item())
            maximum = float(logits.float().max().item())
            if not math.isfinite(checksum) or not math.isfinite(maximum):
                raise SmolParentError("Smol mechanics output is nonfinite")
            argmax = int(logits[0, -1].argmax().item())
    finally:
        model.train(original_training)
    if _tensor_versions(model) != versions_before:
        raise SmolParentError("Smol mechanics mutated model state")
    receipt = {
        "schema": SCHEMA,
        "status": "pass",
        "model_root": str(Path(model_root).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "restoration_receipt_path": str(Path(restoration_receipt_path).resolve()),
        "runtime": runtime,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu_name": device_name,
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        },
        "forward": {
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_tokens": len(token_ids),
            "logits_shape": list(logits.shape),
            "logits_sum_fp32": checksum,
            "logits_max_fp32": maximum,
            "argmax_token_id": argmax,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "finite": True,
        },
        "training_executed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "model_state_unchanged": True,
        "architecture_result": False,
        "four_b_training_executed": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def write_receipt(output: Path, receipt: dict[str, Any]) -> None:
    output = Path(output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise SmolParentError("Smol mechanics output boundary differs")
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--restoration-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify_smol_parent(
        args.model_root,
        manifest_path=args.manifest,
        restoration_receipt_path=args.restoration_receipt,
    )
    write_receipt(args.output, receipt)
    print(json.dumps({"status": "pass", "receipt_sha256": receipt["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
