"""Evaluate one completed matched SmolLM3 workspace arm on a Sai dev board."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sai.adaptive.hf_workspace import FrozenHFWorkspaceSystem
from sai.adaptive.reference import LatentWorkspace
from sai.data.token_stream import canonical_sha256
from sai.evaluation.development_mc import (
    DevelopmentMCError,
    evaluate_development_mc,
    write_development_mc,
)
from sai.evaluation.hf_smol_parent import (
    load_smol_parent,
    validate_smol_mechanics_receipt,
)
from sai.evaluation.hf_workspace_mc import (
    HFWorkspaceChoiceAdapter,
    HFWorkspaceEvaluationError,
)
from sai.training.checkpoint import load_mechanics_checkpoint
from sai.training.hf_smol_workspace_screen import (
    EXPECTED_WORKSPACE_PARAMETERS,
    SEED,
    WORKSPACE_CONFIG,
    _state_sha256,
    make_bindings,
)
from sai.training.hf_smol_workspace_screen import SCHEMA as TRAINING_SCHEMA
from sai.training.hf_workspace_screen import (
    ITERATIONS,
    KL_COEFFICIENT,
    SEQUENCES_PER_UPDATE,
)
from sai.training.runner import TrainingRunConfig, build_adamw


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise HFWorkspaceEvaluationError("evaluation artifact is missing or unsafe")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_result(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _sha256_file(path) != expected_sha256:
        raise HFWorkspaceEvaluationError("training result file identity differs")
    try:
        result = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFWorkspaceEvaluationError("training result is unreadable") from error
    if not isinstance(result, dict):
        raise HFWorkspaceEvaluationError("training result differs")
    unsigned = dict(result)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    if (
        receipt_sha256 != canonical_sha256(unsigned)
        or result.get("schema") != TRAINING_SCHEMA
        or result.get("status") != "complete"
        or result.get("training_sequences") != 61_035
        or result.get("workspace_parameter_count") != EXPECTED_WORKSPACE_PARAMETERS
        or result.get("workspace_config") != WORKSPACE_CONFIG.as_dict()
        or result.get("workspace_initialization_seed") != SEED
        or result.get("iterations") != ITERATIONS
        or result.get("objective", {}).get("frozen_parent_kl") != KL_COEFFICIENT
        or result.get("sequences_per_update") != SEQUENCES_PER_UPDATE
        or result.get("parent_state_unchanged") is not True
        or result.get("four_b_training_executed") is not False
        or result.get("architecture_improvement_demonstrated") is not False
    ):
        raise HFWorkspaceEvaluationError("training result evidence differs")
    return result


def load_completed_workspace(
    *, result_path: Path, result_sha256: str, checkpoint_path: Path
) -> tuple[LatentWorkspace, dict[str, Any]]:
    """Reopen a completed Smol run receipt and exact workspace checkpoint."""

    result = _load_result(result_path, result_sha256)
    state_mode = result.get("state_mode")
    if state_mode not in {"recurrent", "reset_average"}:
        raise HFWorkspaceEvaluationError("training state mode differs")
    optimizer_payload = result.get("optimizer")
    if not isinstance(optimizer_payload, dict):
        raise HFWorkspaceEvaluationError("training optimizer receipt differs")
    optimizer_payload = dict(optimizer_payload)
    betas = optimizer_payload.get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        raise HFWorkspaceEvaluationError("training Adam betas differ")
    optimizer_payload["betas"] = tuple(betas)
    optimizer_config = TrainingRunConfig(**optimizer_payload)
    bindings, specification = make_bindings(
        state_mode=state_mode,
        snapshot_tree_sha256=result["parent"]["snapshot_tree_sha256"],
        mechanics_file_sha256=result["mechanics_receipt_file_sha256"],
        stream_identity_sha256=result["training_stream_identity_sha256"],
        source_manifest_sha256=result["training_source_manifest_sha256"],
        training_sequences=result["training_sequences"],
        training_utf8_bytes=result["training_utf8_bytes"],
        optimizer=optimizer_config,
        code_sha256=result["code_sha256"],
        environment_sha256=result["environment_sha256"],
    )
    for key, value in specification.items():
        if canonical_sha256(result.get(key)) != canonical_sha256(value):
            raise HFWorkspaceEvaluationError(f"training specification {key} differs")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    workspace = LatentWorkspace(WORKSPACE_CONFIG).to("cuda:0")
    optimizer = build_adamw(workspace, optimizer_config)
    restored = load_mechanics_checkpoint(
        checkpoint_path,
        model=workspace,
        optimizer=optimizer,
        expected_bindings=bindings,
    )
    if (
        restored.counters.as_dict() != result.get("counters")
        or restored.cursor.as_dict() != result.get("stream_cursor")
        or restored.checkpoint_sha256 != result.get("checkpoint", {}).get("sha256")
        or restored.checkpoint_bytes != result.get("checkpoint", {}).get("bytes")
        or _state_sha256(workspace) != result.get("workspace_final_state_sha256")
    ):
        raise HFWorkspaceEvaluationError("workspace checkpoint evidence differs")
    return workspace, result


def _runtime_paths(model: nn.Module, tokenizer: object) -> list[Path]:
    paths = [Path(__file__)]
    for value in (
        type(model),
        type(tokenizer),
        LatentWorkspace,
        FrozenHFWorkspaceSystem,
        HFWorkspaceChoiceAdapter,
    ):
        source = inspect.getsourcefile(value)
        if not source:
            raise HFWorkspaceEvaluationError("workspace runtime source is unavailable")
        paths.append(Path(source))
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate one full Smol arm and emit a standard development result."""

    result_sha256 = _sha256_file(args.training_result)
    if result_sha256 != args.training_result_sha256:
        raise HFWorkspaceEvaluationError("training result file identity differs")
    workspace, training = load_completed_workspace(
        result_path=args.training_result,
        result_sha256=result_sha256,
        checkpoint_path=args.checkpoint,
    )
    mechanics = validate_smol_mechanics_receipt(
        args.mechanics_receipt,
        expected_file_sha256=args.mechanics_receipt_sha256,
        model_root=args.model_root,
        manifest_path=args.model_manifest,
        restoration_receipt_path=args.restoration_receipt,
    )
    if training["mechanics_receipt_file_sha256"] != args.mechanics_receipt_sha256:
        raise HFWorkspaceEvaluationError("training and evaluation mechanics differ")
    parent, tokenizer, runtime = load_smol_parent(
        args.model_root,
        manifest_path=args.model_manifest,
        restoration_receipt_path=args.restoration_receipt,
    )
    if runtime != mechanics["runtime"]:
        raise HFWorkspaceEvaluationError("live Smol parent differs from mechanics")
    adapter = HFWorkspaceChoiceAdapter(
        FrozenHFWorkspaceSystem(parent, workspace),
        state_mode=training["state_mode"],
    )
    try:
        result = evaluate_development_mc(
            adapter,
            tokenizer,
            benchmark=args.benchmark,
            source_path=args.benchmark_source,
            disjoint_receipt_path=args.disjoint_receipt,
            training_source_sha256=args.training_source_sha256,
            checkpoint_paths=[
                args.model_root / "model-00001-of-00002.safetensors",
                args.model_root / "model-00002-of-00002.safetensors",
                args.model_manifest,
                args.restoration_receipt,
                args.mechanics_receipt,
                args.training_result,
                args.checkpoint,
                args.checkpoint.with_name(f"{args.checkpoint.name}.manifest.json"),
            ],
            config_paths=[
                args.model_root / "config.json",
                args.model_root / "model.safetensors.index.json",
                args.training_result,
            ],
            tokenizer_paths=[
                args.model_root / "tokenizer.json",
                args.model_root / "tokenizer_config.json",
                args.model_root / "special_tokens_map.json",
            ],
            runtime_paths=_runtime_paths(parent, tokenizer),
            expected_rows=args.expected_rows,
            expected_identity_order_sha256=args.expected_identity_order_sha256,
            max_sequence_tokens=args.max_sequence_tokens,
            autocast_dtype=torch.bfloat16,
        )
    except DevelopmentMCError as error:
        raise HFWorkspaceEvaluationError(
            "Smol workspace development scoring failed"
        ) from error
    result["workspace_evidence"] = {
        "training_result_file_sha256": result_sha256,
        "training_receipt_sha256": training["receipt_sha256"],
        "training_run_sha256": training["run_sha256"],
        "parent_snapshot_tree_sha256": training["parent"]["snapshot_tree_sha256"],
        "workspace_config_sha256": training["workspace_config_sha256"],
        "workspace_parameter_count": training["workspace_parameter_count"],
        "workspace_initial_state_sha256": training["workspace_initial_state_sha256"],
        "workspace_final_state_sha256": training["workspace_final_state_sha256"],
        "training_stream_identity_sha256": training["training_stream_identity_sha256"],
        "training_source_manifest_sha256": training["training_source_manifest_sha256"],
        "training_sequences": training["training_sequences"],
        "training_utf8_bytes": training["training_utf8_bytes"],
        "optimizer": training["optimizer"],
        "code_sha256": training["code_sha256"],
        "environment_sha256": training["environment_sha256"],
        "state_mode": training["state_mode"],
        "matched_comparison": True,
        "cross_family_confirmation": True,
        "source_disjoint_from_factor_training": True,
        "four_b_training_executed": False,
    }
    unsigned = dict(result)
    unsigned.pop("receipt_sha256")
    result["receipt_sha256"] = canonical_sha256(unsigned)
    write_development_mc(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--restoration-receipt", type=Path, required=True)
    parser.add_argument("--mechanics-receipt", type=Path, required=True)
    parser.add_argument("--mechanics-receipt-sha256", required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-result-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("mmlu_pro", "musr"), required=True)
    parser.add_argument("--benchmark-source", type=Path, required=True)
    parser.add_argument("--disjoint-receipt", type=Path, required=True)
    parser.add_argument("--training-source-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-identity-order-sha256", required=True)
    parser.add_argument("--max-sequence-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(result["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
