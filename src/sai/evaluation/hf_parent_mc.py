"""Evaluate the exact unchanged Qwen3.5-0.8B text parent on a Sai dev board."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import torch

from sai.evaluation.development_mc import (
    evaluate_development_mc,
    write_development_mc,
)
from sai.evaluation.hf_parent import (
    HFTextLogitAdapter,
    load_text_parent,
    validate_mechanics_receipt,
)


def _runtime_paths(model: torch.nn.Module, tokenizer: object) -> list[Path]:
    paths = [Path(__file__)]
    for value in (type(model), type(tokenizer)):
        source = inspect.getsourcefile(value)
        if not source:
            raise RuntimeError("HF parent runtime source is unavailable")
        paths.append(Path(source))
    return paths


def run(args: argparse.Namespace) -> dict:
    """Validate the mechanics boundary and emit one standard MC result."""

    mechanics = validate_mechanics_receipt(
        args.mechanics_receipt,
        expected_file_sha256=args.mechanics_receipt_sha256,
        model_root=args.model_root,
    )
    model, tokenizer, runtime = load_text_parent(args.model_root)
    if runtime != mechanics["runtime"]:
        raise RuntimeError("HF parent live runtime differs from mechanics")
    adapter = HFTextLogitAdapter(model)
    result = evaluate_development_mc(
        adapter,
        tokenizer,
        benchmark=args.benchmark,
        source_path=args.benchmark_source,
        disjoint_receipt_path=args.disjoint_receipt,
        training_source_sha256=args.training_source_sha256,
        checkpoint_paths=[
            args.model_root / "model.safetensors-00001-of-00001.safetensors",
            args.model_root / "snapshot.json",
            args.mechanics_receipt,
        ],
        config_paths=[
            args.model_root / "config.json",
            args.model_root / "model.safetensors.index.json",
        ],
        tokenizer_paths=[
            args.model_root / "tokenizer.json",
            args.model_root / "tokenizer_config.json",
            args.model_root / "vocab.json",
            args.model_root / "merges.txt",
        ],
        runtime_paths=_runtime_paths(model, tokenizer),
        expected_rows=args.expected_rows,
        expected_identity_order_sha256=args.expected_identity_order_sha256,
        max_sequence_tokens=args.max_sequence_tokens,
        autocast_dtype=torch.bfloat16,
    )
    result["parent_evidence"] = {
        "model": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "unchanged_parent": True,
        "upstream_pretraining_contamination_status": "not_auditable_from_public_model",
        "source_disjoint_from_sai_factor_training": True,
        "mechanics_receipt_sha256": mechanics["receipt_sha256"],
    }
    from sai.data.token_stream import canonical_sha256

    unsigned = dict(result)
    unsigned.pop("receipt_sha256")
    result["receipt_sha256"] = canonical_sha256(unsigned)
    write_development_mc(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
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
