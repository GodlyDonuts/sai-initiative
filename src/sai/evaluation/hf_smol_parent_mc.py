"""Evaluate the exact unchanged SmolLM3-3B parent on a Sai dev board."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch
from torch import nn

from sai.data.token_stream import canonical_sha256
from sai.evaluation.development_mc import (
    evaluate_development_mc,
    write_development_mc,
)
from sai.evaluation.hf_smol_parent import (
    SmolParentError,
    load_smol_parent,
    validate_smol_mechanics_receipt,
)


class SmolTextLogitAdapter(nn.Module):
    """Expose the standard Sai likelihood surface over unchanged SmolLM3."""

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
            raise SmolParentError("Smol evaluator requires one unsegmented sequence")
        output = self.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=0,
        )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor):
            raise SmolParentError("Smol evaluator did not return logits")
        return logits


def _runtime_paths(model: nn.Module, tokenizer: object) -> list[Path]:
    paths = [Path(__file__)]
    for value in (type(model), type(tokenizer), SmolTextLogitAdapter):
        source = inspect.getsourcefile(value)
        if not source:
            raise SmolParentError("Smol evaluator runtime source is unavailable")
        paths.append(Path(source))
    return paths


def run(args: argparse.Namespace) -> dict:
    """Validate the exact mechanics boundary and emit one standard result."""

    mechanics = validate_smol_mechanics_receipt(
        args.mechanics_receipt,
        expected_file_sha256=args.mechanics_receipt_sha256,
        model_root=args.model_root,
        manifest_path=args.model_manifest,
        restoration_receipt_path=args.restoration_receipt,
    )
    model, tokenizer, runtime = load_smol_parent(
        args.model_root,
        manifest_path=args.model_manifest,
        restoration_receipt_path=args.restoration_receipt,
    )
    if runtime != mechanics["runtime"]:
        raise SmolParentError("live Smol runtime differs from mechanics")
    result = evaluate_development_mc(
        SmolTextLogitAdapter(model),
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
        ],
        config_paths=[
            args.model_root / "config.json",
            args.model_root / "model.safetensors.index.json",
        ],
        tokenizer_paths=[
            args.model_root / "tokenizer.json",
            args.model_root / "tokenizer_config.json",
            args.model_root / "special_tokens_map.json",
        ],
        runtime_paths=_runtime_paths(model, tokenizer),
        expected_rows=args.expected_rows,
        expected_identity_order_sha256=args.expected_identity_order_sha256,
        max_sequence_tokens=args.max_sequence_tokens,
        autocast_dtype=torch.bfloat16,
    )
    result["parent_evidence"] = {
        "model": "HuggingFaceTB/SmolLM3-3B",
        "revision": "a07cc9a04f16550a088caea529712d1d335b0ac1",
        "parameter_count": 3_075_098_624,
        "unchanged_parent": True,
        "snapshot_tree_sha256": runtime["snapshot"]["tree_sha256"],
        "upstream_pretraining_contamination_status": "not_auditable_from_public_model",
        "source_disjoint_from_sai_factor_training": True,
        "mechanics_receipt_sha256": mechanics["receipt_sha256"],
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
