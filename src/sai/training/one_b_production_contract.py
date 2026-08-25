"""Build Sai's exact, non-launching OLMo2-derived 1B production contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.one_b_spiral_contract import SEQUENCE_LENGTH, TOTAL_TOKENS
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-1b-production-model-contract-v1"
OLMO_COMMIT = "090253dac6688f2532509daa7aa2eb5fae50e956"
OLMO_CORE_COMMIT = "7899e7cefaae44e30766ee654bd177f1e1474bc7"


class OneBProductionContractError(RuntimeError):
    """The model ledger, optimizer, trainer, or token geometry differs."""


def _parameter_ledger() -> dict[str, int]:
    vocabulary = 48_000
    width = 2_048
    layers = 16
    hidden = 5_504
    embeddings = 2 * vocabulary * width
    attention_per_layer = 4 * width * width
    feed_forward_per_layer = 3 * width * hidden
    norms_per_layer = 4 * width
    final_norm = width
    total = (
        embeddings
        + layers * (attention_per_layer + feed_forward_per_layer + norms_per_layer)
        + final_norm
    )
    return {
        "untied_embeddings": embeddings,
        "attention": layers * attention_per_layer,
        "feed_forward": layers * feed_forward_per_layer,
        "block_and_qk_norms": layers * norms_per_layer,
        "final_norm": final_norm,
        "total": total,
    }


def build_contract() -> dict[str, Any]:
    """Return the complete prospective production model and trainer identity."""

    ledger = _parameter_ledger()
    ordinary_batch_sequences = 512
    ordinary_batch_tokens = ordinary_batch_sequences * SEQUENCE_LENGTH
    full_steps, terminal_tokens = divmod(TOTAL_TOKENS, ordinary_batch_tokens)
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_1b_production_contract",
        "architecture_basis": "published_olmo2_1b_conservative_baseline",
        "architecture_novelty_claimed": False,
        "upstream": {
            "allenai_olmo_commit": OLMO_COMMIT,
            "allenai_olmo_core_commit": OLMO_CORE_COMMIT,
        },
        "model": {
            "vocab_size": 48_000,
            "embedding_size": 48_000,
            "d_model": 2_048,
            "n_layers": 16,
            "n_heads": 16,
            "n_kv_heads": 16,
            "mlp_hidden_size": 5_504,
            "max_sequence_length": SEQUENCE_LENGTH,
            "weight_tying": False,
            "block_type": "reordered_norm",
            "qk_norm": True,
            "rope_theta": 500_000,
            "activation": "swiglu",
            "attention_backend": "flash_attention",
            "dropout": 0.0,
            "bias": False,
            "rms_norm_eps": 1e-6,
        },
        "parameter_ledger": ledger,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 4e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "scheduler": "cosine_with_token_warmup",
            "warmup_tokens": 8_388_608_000,
            "final_learning_rate_ratio": 0.1,
        },
        "distributed": {
            "precision": "amp_bf16",
            "fsdp_sharding_strategy": "SHARD_GRAD_OP",
            "ordinary_global_batch_sequences": ordinary_batch_sequences,
            "ordinary_global_batch_tokens": ordinary_batch_tokens,
            "full_ordinary_steps": full_steps,
            "terminal_partial_batch_tokens": terminal_tokens,
            "terminal_partial_batch_sequences": terminal_tokens // SEQUENCE_LENGTH,
            "exact_stage_boundary_partial_batches": True,
        },
        "target_tokens": TOTAL_TOKENS,
        "six_parameter_token_flops": 6 * ledger["total"] * TOTAL_TOKENS,
        "public_benchmark_improvement_claimed": False,
        "model_training_started": False,
        "one_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def validate_contract(value: Any) -> dict[str, Any]:
    expected = build_contract()
    if value != expected:
        raise OneBProductionContractError("1B production contract differs")
    if (
        value["parameter_ledger"]["total"] != 1_006_241_792
        or value["distributed"]["terminal_partial_batch_sequences"] != 324
        or value["target_tokens"] != TOTAL_TOKENS
    ):
        raise OneBProductionContractError("1B production arithmetic differs")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise OneBProductionContractError("production contract output exists")
    payload = validate_contract(build_contract())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
