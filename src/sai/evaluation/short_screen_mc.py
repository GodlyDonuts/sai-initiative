"""Read-only development-MC evaluation of one completed Sai scale checkpoint.

This runner reconstructs an exact frozen 100M, 300M, or 1B model and loads
only its validated model state.  It never constructs an optimizer, runs
backward, or authorizes a 4B experiment or a public benchmark claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from contextlib import contextmanager
from io import BufferedReader
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sai.data.token_stream import (
    TokenStreamError,
    canonical_sha256,
    sha256_file,
    sha256_tree,
    validate_frozen_stream,
)
from sai.evaluation.development_mc import (
    DevelopmentMCError,
    evaluate_development_mc,
    write_development_mc,
)
from sai.evaluation.scale_checkpoint import EVALUATION_SCALES, load_evaluation_config
from sai.model.initialization import POLICY_SHA256
from sai.model.reference import SaiCausalLM, exact_parameter_count
from sai.training.checkpoint import (
    CHECKPOINT_SCHEMA,
    MANIFEST_SCHEMA,
    CheckpointBindings,
    MechanicsCheckpointError,
    TrainingCounters,
)
from sai.training.short_screen import SCHEMA as SHORT_SCREEN_SCHEMA
from sai.training.stream import StreamCursor, TrainingStreamError

SCALE_TRAINING_SCHEMA = "sai-sub-4b-scale-training-v1"


class ShortScreenMCError(RuntimeError):
    """A completed screen, model state, or evaluation binding differs."""


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ShortScreenMCError(f"{field} must be a lowercase SHA256")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ShortScreenMCError(f"{field} must be a positive integer")
    return value


@contextmanager
def _regular_handle(path: Path, field: str):
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ShortScreenMCError(f"{field} is missing or unsafe") from error
    handle = os.fdopen(descriptor, "rb")
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        handle.close()
        raise ShortScreenMCError(f"{field} is not a unique regular file")
    try:
        yield handle, metadata
    finally:
        handle.close()


def _sha256_handle(handle: BufferedReader) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _load_json(path: Path, field: str) -> tuple[dict[str, Any], str]:
    with _regular_handle(path, field) as (handle, _):
        encoded = handle.read()
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShortScreenMCError(f"{field} is unreadable") from error
    if not isinstance(value, dict):
        raise ShortScreenMCError(f"{field} must be an object")
    return value, hashlib.sha256(encoded).hexdigest()


def _state_sha256(model: nn.Module) -> str:
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


def _short_screen_specification(
    result: dict[str, Any], *, scale: str
) -> dict[str, Any]:
    fields = {
        "schema",
        "evidence_class",
        "scientific_promotion_authorized",
        "four_b_training_authorized",
        "config",
        "config_sha256",
        "model_sha256",
        "delta_backend",
        "initialization_policy_sha256",
        "initialization_seed",
        "training_stream_identity_sha256",
        "development_stream_identity_sha256",
        "code_sha256",
        "environment_sha256",
        "optimizer",
        "precision",
        "micro_batch_size_sequences",
        "sequences_per_update",
        "training_sequences",
        "training_utf8_bytes",
        "development_sequences",
        "development_batch_size_sequences",
        "checkpoint_interval_steps",
        "mechanics_only",
    }
    if scale != "100m":
        fields.update({"scale", "promotion_receipt_sha256"})
    if "milestone_steps" in result:
        fields.add("milestone_steps")
    if not fields.issubset(result):
        raise ShortScreenMCError("short-screen specification is incomplete")
    return {field: result[field] for field in fields}


def validate_short_screen_result(
    result_path: Path,
    *,
    expected_sha256: str,
    config: Any,
    family: str,
    geometry_parameter_count: int,
    scale: str = "100m",
) -> tuple[dict[str, Any], CheckpointBindings]:
    """Validate the immutable terminal receipt and derive checkpoint bindings."""

    if scale not in EVALUATION_SCALES:
        raise ShortScreenMCError("evaluation scale must be 100m, 300m, or 1b")
    result, observed_sha256 = _load_json(result_path, "short-screen result")
    if observed_sha256 != _sha256(expected_sha256, "short-screen result SHA256"):
        raise ShortScreenMCError("short-screen result bytes differ")
    unsigned = dict(result)
    claimed_receipt = unsigned.pop("receipt_sha256", None)
    if claimed_receipt != canonical_sha256(unsigned):
        raise ShortScreenMCError("short-screen result receipt differs")
    backend = "fla" if family in {"gdn_hybrid", "kda_mla_hybrid"} else "reference"
    config_sha256 = canonical_sha256(config.as_dict())
    model_sha256 = canonical_sha256(
        {
            "config_sha256": config_sha256,
            "delta_backend": backend,
            "initialization_policy_sha256": POLICY_SHA256,
            "initialization_seed": result.get("initialization_seed"),
        }
    )
    expected_schema = SHORT_SCREEN_SCHEMA if scale == "100m" else SCALE_TRAINING_SCHEMA
    if scale != "100m":
        _sha256(
            result.get("promotion_receipt_sha256"),
            "scale promotion receipt SHA256",
        )
    if (
        result.get("schema") != expected_schema
        or result.get("status") != "complete"
        or result.get("config") != config.as_dict()
        or result.get("config_sha256") != config_sha256
        or result.get("model_sha256") != model_sha256
        or result.get("delta_backend") != backend
        or result.get("initialization_policy_sha256") != POLICY_SHA256
        or result.get("parameter_count") != geometry_parameter_count
        or result.get("scientific_promotion_authorized") is not False
        or result.get("four_b_training_authorized") is not False
        or (scale != "100m" and result.get("scale") != scale)
        or (
            scale == "100m"
            and ("scale" in result or "promotion_receipt_sha256" in result)
        )
    ):
        raise ShortScreenMCError("short-screen model identity differs")
    specification = _short_screen_specification(result, scale=scale)
    run_sha256 = canonical_sha256(specification)
    if result.get("run_sha256") != run_sha256:
        raise ShortScreenMCError("short-screen run identity differs")
    try:
        bindings = CheckpointBindings(
            model_sha256=model_sha256,
            config_sha256=config_sha256,
            ordered_stream_identity_sha256=_sha256(
                result.get("training_stream_identity_sha256"),
                "training stream identity",
            ),
            code_sha256=_sha256(result.get("code_sha256"), "code identity"),
            environment_sha256=_sha256(
                result.get("environment_sha256"), "environment identity"
            ),
            run_sha256=run_sha256,
        )
    except MechanicsCheckpointError as error:
        raise ShortScreenMCError("short-screen checkpoint bindings differ") from error
    return result, bindings


def load_validated_model_state(
    checkpoint_path: Path,
    manifest_path: Path,
    *,
    model: nn.Module,
    expected_bindings: CheckpointBindings,
    expected_descriptor: dict[str, Any],
    expected_counters: dict[str, Any],
    expected_cursor: dict[str, Any],
    expected_final_state_sha256: str,
) -> dict[str, Any]:
    """Load only a fully validated model state; never restore optimizer or RNG."""

    manifest, manifest_sha256 = _load_json(manifest_path, "checkpoint manifest")
    if set(manifest) != {"schema", "checkpoint", "bindings", "counters", "cursor"}:
        raise ShortScreenMCError("checkpoint manifest membership differs")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ShortScreenMCError("checkpoint manifest schema differs")
    descriptor = manifest.get("checkpoint")
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"path", "bytes", "sha256"}
        or descriptor != expected_descriptor
        or descriptor.get("path") != Path(checkpoint_path).name
    ):
        raise ShortScreenMCError("checkpoint descriptor differs")
    with _regular_handle(checkpoint_path, "checkpoint") as (handle, metadata):
        if metadata.st_size != descriptor.get("bytes") or _sha256_handle(
            handle
        ) != _sha256(descriptor.get("sha256"), "checkpoint SHA256"):
            raise ShortScreenMCError("checkpoint bytes differ from manifest")
        try:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ShortScreenMCError("checkpoint payload is unreadable") from error
    try:
        bindings = CheckpointBindings.from_dict(manifest.get("bindings"))
        counters = TrainingCounters.from_dict(manifest.get("counters"))
        cursor = StreamCursor.from_dict(manifest.get("cursor"))
    except (MechanicsCheckpointError, TrainingStreamError) as error:
        raise ShortScreenMCError("checkpoint lineage differs") from error
    if (
        bindings != expected_bindings
        or counters.as_dict() != expected_counters
        or cursor.as_dict() != expected_cursor
        or counters.sequences != cursor.next_sequence
        or cursor.ordered_stream_identity_sha256
        != bindings.ordered_stream_identity_sha256
    ):
        raise ShortScreenMCError("checkpoint lineage differs")
    expected_payload_keys = {
        "schema",
        "bindings",
        "counters",
        "cursor",
        "model_state_dict",
        "optimizer_state_dict",
        "cpu_rng_state",
        "cuda_available",
        "cuda_device_count",
        "cuda_rng_states",
    }
    if not isinstance(payload, dict) or set(payload) != expected_payload_keys:
        raise ShortScreenMCError("checkpoint payload membership differs")
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("bindings") != manifest["bindings"]
        or payload.get("counters") != manifest["counters"]
        or payload.get("cursor") != manifest["cursor"]
        or not isinstance(payload.get("optimizer_state_dict"), dict)
        or set(payload["optimizer_state_dict"]) != {"state", "param_groups"}
    ):
        raise ShortScreenMCError("checkpoint payload lineage differs")
    saved = payload.get("model_state_dict")
    current = model.state_dict()
    if not isinstance(saved, dict) or set(saved) != set(current):
        raise ShortScreenMCError("checkpoint model state membership differs")
    for name, target in current.items():
        source = saved[name]
        if (
            not isinstance(source, torch.Tensor)
            or source.shape != target.shape
            or source.dtype != target.dtype
        ):
            raise ShortScreenMCError(f"checkpoint model tensor {name} differs")
    model.load_state_dict(saved, strict=True)
    final_state_sha256 = _state_sha256(model)
    if final_state_sha256 != _sha256(
        expected_final_state_sha256, "final model state SHA256"
    ):
        raise ShortScreenMCError("checkpoint final model state differs")
    return {
        "checkpoint_sha256": descriptor["sha256"],
        "checkpoint_bytes": descriptor["bytes"],
        "manifest_sha256": manifest_sha256,
        "final_state_sha256": final_state_sha256,
    }


def _tokenizer_files(root: Path) -> list[Path]:
    paths = [path for path in sorted(Path(root).rglob("*")) if path.is_file()]
    if not paths:
        raise ShortScreenMCError("tokenizer tree contains no files")
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate all inputs and execute one immutable development-only score."""

    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.device_count() != 1
    ):
        raise ShortScreenMCError("exactly one CUDA BF16 GPU is required")
    if args.output.exists() or args.output.is_symlink():
        raise ShortScreenMCError("development output already exists")
    if sha256_file(args.benchmark_source) != _sha256(
        args.benchmark_source_sha256, "benchmark source SHA256"
    ) or sha256_file(args.disjoint_receipt) != _sha256(
        args.disjoint_receipt_sha256, "disjoint receipt SHA256"
    ):
        raise ShortScreenMCError("benchmark or disjoint receipt bytes differ")
    config, geometry_row = load_evaluation_config(
        args.geometry, args.family, args.scale
    )
    if sha256_file(args.geometry) != _sha256(args.geometry_sha256, "geometry SHA256"):
        raise ShortScreenMCError("geometry bytes differ")
    result, bindings = validate_short_screen_result(
        args.short_screen_result,
        expected_sha256=args.short_screen_result_sha256,
        config=config,
        family=args.family,
        geometry_parameter_count=geometry_row["parameter_ledger"]["total"],
        scale=args.scale,
    )
    try:
        training_stream = validate_frozen_stream(
            args.training_stream, verify_sources=True
        )
        tokenizer_sha256 = sha256_tree(args.tokenizer_root)
    except TokenStreamError as error:
        raise ShortScreenMCError("training stream or tokenizer differs") from error
    if (
        training_stream["ordered_stream_identity_sha256"]
        != _sha256(args.training_stream_identity, "training stream identity")
        or training_stream["ordered_stream_identity_sha256"]
        != result["training_stream_identity_sha256"]
        or training_stream["tokenizer_identity_sha256"]
        != _sha256(args.tokenizer_sha256, "tokenizer SHA256")
        or tokenizer_sha256 != args.tokenizer_sha256
        or training_stream["vocab_size"] != config.vocab_size
    ):
        raise ShortScreenMCError("training stream/tokenizer identity differs")

    model = SaiCausalLM(config, delta_backend=result["delta_backend"])
    checkpoint_observation = load_validated_model_state(
        args.checkpoint,
        args.checkpoint_manifest,
        model=model,
        expected_bindings=bindings,
        expected_descriptor=result.get("checkpoint"),
        expected_counters=result.get("counters"),
        expected_cursor=result.get("stream_cursor"),
        expected_final_state_sha256=result.get("final_state_sha256"),
    )
    if exact_parameter_count(model) != geometry_row["parameter_ledger"]["total"]:
        raise ShortScreenMCError("instantiated parameter count differs")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ShortScreenMCError("Transformers is required") from error
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ShortScreenMCError("development evaluation requires a fast tokenizer")
    model = model.to(device="cuda")
    try:
        payload = evaluate_development_mc(
            model,
            tokenizer,
            benchmark=args.benchmark,
            source_path=args.benchmark_source,
            disjoint_receipt_path=args.disjoint_receipt,
            training_source_sha256=args.training_source_sha256,
            checkpoint_paths=[args.checkpoint, args.checkpoint_manifest],
            config_paths=[args.geometry, args.short_screen_result],
            tokenizer_paths=_tokenizer_files(args.tokenizer_root),
            runtime_paths=args.runtime_path,
            expected_rows=args.expected_rows,
            expected_identity_order_sha256=args.expected_identity_order_sha256,
            max_sequence_tokens=args.max_sequence_tokens,
            autocast_dtype=torch.bfloat16,
        )
    except DevelopmentMCError as error:
        raise ShortScreenMCError("development-MC evaluation failed") from error
    if (
        sha256_file(args.checkpoint) != checkpoint_observation["checkpoint_sha256"]
        or sha256_file(args.checkpoint_manifest)
        != checkpoint_observation["manifest_sha256"]
    ):
        raise ShortScreenMCError("checkpoint changed during evaluation")
    if (
        payload["bindings"]["benchmark_source_sha256"] != args.benchmark_source_sha256
        or payload["bindings"]["source_disjoint_receipt_sha256"]
        != args.disjoint_receipt_sha256
        or payload["bindings"]["checkpoint_sha256"]
        != canonical_sha256(
            [
                {
                    "name": args.checkpoint.name,
                    "bytes": args.checkpoint.stat().st_size,
                    "sha256": checkpoint_observation["checkpoint_sha256"],
                },
                {
                    "name": args.checkpoint_manifest.name,
                    "bytes": args.checkpoint_manifest.stat().st_size,
                    "sha256": checkpoint_observation["manifest_sha256"],
                },
            ]
        )
    ):
        raise ShortScreenMCError("development output binding differs")
    payload["short_screen_lineage"] = {
        "scale": args.scale,
        "run_sha256": result["run_sha256"],
        "result_sha256": args.short_screen_result_sha256,
        "model_sha256": result["model_sha256"],
        **checkpoint_observation,
    }
    unsigned = dict(payload)
    unsigned.pop("receipt_sha256")
    payload["receipt_sha256"] = canonical_sha256(unsigned)
    write_development_mc(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--scale", choices=EVALUATION_SCALES, default="100m")
    parser.add_argument("--short-screen-result", type=Path, required=True)
    parser.add_argument("--short-screen-result-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--training-stream", type=Path, required=True)
    parser.add_argument("--training-stream-identity", required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--benchmark", choices=("mmlu_pro", "musr"), required=True)
    parser.add_argument("--benchmark-source", type=Path, required=True)
    parser.add_argument("--benchmark-source-sha256", required=True)
    parser.add_argument("--disjoint-receipt", type=Path, required=True)
    parser.add_argument("--disjoint-receipt-sha256", required=True)
    parser.add_argument("--training-source-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-identity-order-sha256", required=True)
    parser.add_argument("--max-sequence-tokens", type=int, required=True)
    parser.add_argument("--runtime-path", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _positive_integer(args.expected_rows, "expected rows")
    _positive_integer(args.max_sequence_tokens, "maximum sequence tokens")
    payload = run(args)
    print(
        json.dumps(
            {
                "benchmark": payload["benchmark"],
                "receipt_sha256": payload["receipt_sha256"],
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
