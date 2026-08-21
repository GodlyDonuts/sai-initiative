"""Load and qualify the exact pretrained Qwen3.5-0.8B text-only parent."""

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

from sai.data.hf_model_snapshot import validate_snapshot
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-qwen35-0p8b-text-mechanics-v1"
EXPECTED_MODEL_CLASS = "Qwen3_5ForCausalLM"
EXPECTED_PARAMETER_COUNT = 752_393_024
EXPECTED_VOCAB_SIZE = 248_320


class HFParentError(RuntimeError):
    """The exact parent, CUDA load, or text-only forward differs."""


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise HFParentError(f"runtime artifact is missing or unsafe: {path}")
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
    if not parameters:
        raise HFParentError("parent contains no parameters")
    wrong = [
        name
        for name, value in [*parameters, *buffers]
        if value.device.type != "cuda" or value.device.index not in {0, None}
    ]
    if wrong:
        raise HFParentError("parent parameter or buffer is not resident on CUDA:0")
    return {
        "parameter_tensors": len(parameters),
        "buffer_tensors": len(buffers),
        "parameter_count": sum(value.numel() for _, value in parameters),
        "parameter_dtypes": sorted({str(value.dtype) for _, value in parameters}),
        "all_parameters_and_buffers_cuda_zero": True,
    }


class HFTextLogitAdapter(nn.Module):
    """Expose the standard Sai evaluator surface over one HF causal LM."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, segment_ids: torch.Tensor
    ) -> torch.Tensor:
        if (
            input_ids.ndim != 2
            or segment_ids.shape != input_ids.shape
            or bool(segment_ids.ne(0).any().item())
        ):
            raise HFParentError("HF parent evaluator requires one unsegmented sequence")
        output = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=0,
        )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise HFParentError("HF parent did not return logits")
        return logits


def load_text_parent(model_root: Path) -> tuple[nn.Module, Any, dict[str, Any]]:
    """Load only the causal text path and prove exact CUDA residency."""

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
    ):
        raise HFParentError("exactly one CUDA BF16 GPU is required")
    snapshot = validate_snapshot(model_root)
    try:
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise HFParentError("Transformers parent runtime is unavailable") from error

    tokenizer = AutoTokenizer.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
    )
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise HFParentError("parent tokenizer vocabulary differs")
    model, loading = AutoModelForCausalLM.from_pretrained(
        model_root,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
        output_loading_info=True,
    )
    if type(model).__name__ != EXPECTED_MODEL_CLASS:
        raise HFParentError("parent text model class differs")
    if not isinstance(loading, dict) or any(
        loading.get(field)
        for field in ("missing_keys", "mismatched_keys", "error_msgs")
    ):
        raise HFParentError("parent text model load is incomplete")
    unexpected = loading.get("unexpected_keys", [])
    if not isinstance(unexpected, list) or any(
        not isinstance(name, str)
        or not (name.startswith("model.visual.") or name.startswith("mtp."))
        for name in unexpected
    ):
        raise HFParentError("parent text model unexpected weights differ")
    residency = _cuda_residency(model)
    if residency["parameter_count"] != EXPECTED_PARAMETER_COUNT:
        raise HFParentError("parent text parameter count differs")
    model_source = Path(inspect.getsourcefile(type(model)) or "")
    runtime = {
        "snapshot_receipt_sha256": snapshot["receipt_sha256"],
        "snapshot_tree_sha256": snapshot["tree_sha256"],
        "model_class": type(model).__name__,
        "transformers_version": transformers.__version__,
        "transformers_model_source_sha256": _sha256_file(model_source),
        "unexpected_weight_count": len(unexpected),
        "unexpected_weight_names_sha256": canonical_sha256(sorted(unexpected)),
        **residency,
    }
    return model, tokenizer, runtime


def qualify_parent(model_root: Path) -> dict[str, Any]:
    """Run one deterministic text-only forward without mutating the parent."""

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if "H100" not in device_name:
        raise HFParentError("parent mechanics requires an H100")
    model, tokenizer, runtime = load_text_parent(model_root)
    prompt = "Sai should improve a small language model by measuring"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids or any(not isinstance(value, int) for value in input_ids):
        raise HFParentError("parent mechanics tokenization differs")
    inputs = torch.tensor([input_ids], dtype=torch.long, device="cuda:0")
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
                raise HFParentError("parent mechanics logits differ")
            checksum = float(logits.float().sum().item())
            maximum = float(logits.float().max().item())
            if not math.isfinite(checksum) or not math.isfinite(maximum):
                raise HFParentError("parent mechanics output is nonfinite")
            argmax = int(logits[0, -1].argmax().item())
    finally:
        model.train(original_training)
    if _tensor_versions(model) != versions_before:
        raise HFParentError("parent mechanics mutated model state")
    receipt = {
        "schema": SCHEMA,
        "status": "pass",
        "model_root": str(Path(model_root).resolve()),
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
            "input_tokens": len(input_ids),
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
    """Atomically publish one new mechanics receipt."""

    output = Path(output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise HFParentError("mechanics output path differs")
    encoded = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def validate_mechanics_receipt(
    path: Path, *, expected_file_sha256: str, model_root: Path
) -> dict[str, Any]:
    """Replay the immutable mechanics receipt before an expensive evaluation."""

    if _sha256_file(path) != expected_file_sha256:
        raise HFParentError("mechanics receipt file identity differs")
    try:
        receipt = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFParentError("mechanics receipt is unreadable") from error
    if not isinstance(receipt, dict):
        raise HFParentError("mechanics receipt must be an object")
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    runtime = receipt.get("runtime")
    environment = receipt.get("environment")
    forward = receipt.get("forward")
    snapshot = validate_snapshot(model_root)
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != "pass"
        or claimed != canonical_sha256(unsigned)
        or receipt.get("model_root") != str(Path(model_root).resolve())
        or not isinstance(runtime, dict)
        or runtime.get("snapshot_receipt_sha256") != snapshot["receipt_sha256"]
        or runtime.get("snapshot_tree_sha256") != snapshot["tree_sha256"]
        or runtime.get("model_class") != EXPECTED_MODEL_CLASS
        or runtime.get("parameter_count") != EXPECTED_PARAMETER_COUNT
        or runtime.get("all_parameters_and_buffers_cuda_zero") is not True
        or not isinstance(environment, dict)
        or "H100" not in str(environment.get("gpu_name"))
        or not isinstance(forward, dict)
        or forward.get("finite") is not True
        or forward.get("logits_shape") != [1, 1, EXPECTED_VOCAB_SIZE]
        or receipt.get("training_executed") is not False
        or receipt.get("optimizer_steps") != 0
        or receipt.get("backward_calls") != 0
        or receipt.get("model_state_unchanged") is not True
        or receipt.get("architecture_result") is not False
        or receipt.get("four_b_training_executed") is not False
    ):
        raise HFParentError("mechanics receipt evidence differs")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = qualify_parent(args.model_root)
    write_receipt(args.output, receipt)
    print(json.dumps({"status": "pass", "receipt_sha256": receipt["receipt_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
