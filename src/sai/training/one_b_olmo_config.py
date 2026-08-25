"""Generate hash-bound OLMo configs for Sai's exact non-launched 1B schedule."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.one_b_stage_schedule import SCHEMA as SCHEDULE_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file, sha256_tree
from sai.tokenizer.production_qualification import SCHEMA as TOKENIZER_SCHEMA
from sai.training.one_b_production_contract import (
    OLMO_COMMIT,
    OLMO_CORE_COMMIT,
    build_contract,
)

SCHEMA = "sai-1b-olmo-config-bundle-v1"


class OneBOlmoConfigError(RuntimeError):
    """The schedule, tokenizer, model, or exact OLMo config differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBOlmoConfigError("signed config input differs") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or value.get("schema") != schema
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise OneBOlmoConfigError("signed config input differs")
    return value


def _paths(entries: list[dict[str, Any]]) -> list[str]:
    paths = []
    for entry in entries:
        path = Path(entry["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256_file(path) != entry["sha256"]
        ):
            raise OneBOlmoConfigError("scheduled memmap differs")
        paths.extend([str(path.resolve())] * entry["repeat"])
    return paths


def _special_ids(tokenizer_root: Path) -> tuple[int, int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise OneBOlmoConfigError("transformers is required") from error
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if (
        tokenizer.vocab_size != 48_000
        or not isinstance(tokenizer.eos_token_id, int)
        or not isinstance(tokenizer.pad_token_id, int)
    ):
        raise OneBOlmoConfigError("tokenizer special IDs differ")
    return tokenizer.eos_token_id, tokenizer.pad_token_id


def _model(eos_token_id: int, pad_token_id: int) -> dict[str, Any]:
    return {
        "d_model": 2_048,
        "n_heads": 16,
        "n_kv_heads": 16,
        "n_layers": 16,
        "mlp_hidden_size": 11_008,
        "weight_tying": False,
        "alibi": False,
        "rope": True,
        "rope_theta": 500_000,
        "flash_attention": True,
        "attention_dropout": 0.0,
        "include_bias": False,
        "block_type": "sequential",
        "layer_norm_type": "rms",
        "layer_norm_with_affine": True,
        "layer_norm_eps": 1e-6,
        "bias_for_layer_norm": False,
        "attention_layer_norm": True,
        "attention_layer_norm_with_affine": True,
        "norm_after": True,
        "activation_type": "swiglu",
        "residual_dropout": 0.0,
        "embedding_dropout": 0.0,
        "max_sequence_length": 4_096,
        "vocab_size": 48_000,
        "embedding_size": 48_000,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "init_device": "meta",
        "init_fn": "normal",
        "init_std": 0.02,
        "init_cutoff_factor": 3,
    }


def _config(
    *,
    stage: dict[str, Any],
    phase: str,
    paths: list[str],
    tokenizer_json: Path,
    eos_token_id: int,
    pad_token_id: int,
    max_duration_tokens: int,
    load_path: str | None,
) -> dict[str, Any]:
    boundary = phase == "boundary"
    global_batch = stage["boundary_batch_sequences"] if boundary else 512
    return {
        "run_name": f"sai-1b-{stage['index']}-{stage['stage']}-{phase}",
        "seed": 6_198,
        "dry_run": False,
        "wandb": None,
        "model": _model(eos_token_id, pad_token_id),
        "softmax_auxiliary_loss": True,
        "auxiliary_loss_multiplier": 1e-5,
        "fused_loss": True,
        "compile": None,
        "optimizer": {
            "name": "adamw",
            "learning_rate": 4e-4,
            "weight_decay": 0.1,
            "eps": 1e-8,
            "decay_norm_and_bias": True,
            "decay_embeddings": False,
            "betas": [0.9, 0.95],
            "metrics_log_interval": 1,
        },
        "scheduler": {
            "name": "cosine_with_warmup",
            "units": "tokens",
            "t_warmup": 8_388_608_000,
            "t_max": 4_000_000_000_000,
            "alpha_f": 0.1,
            "warmup_min_lr": 0.0,
        },
        "tokenizer": {
            "identifier": str(tokenizer_json.resolve()),
            "truncate_direction": "right",
        },
        "save_folder": (
            "/lustre/fs1/home/sa305415/sai_checkpoints/"
            f"sai-1b-{stage['index']}-{stage['stage']}-{phase}"
        ),
        "save_overwrite": False,
        "save_interval": 1_000,
        "save_interval_ephemeral": 1_000,
        "save_num_checkpoints_to_keep": 2,
        "sharded_checkpointer": "olmo_core",
        "save_interval_unsharded": 1_000,
        "save_num_unsharded_checkpoints_to_keep": 1,
        "load_path": load_path,
        "max_duration": f"{max_duration_tokens}T",
        "global_train_batch_size": global_batch,
        "device_train_microbatch_size": 1 if boundary else 4,
        "precision": "amp_bf16",
        "fsdp": {
            "wrapping_strategy": None,
            "sharding_strategy": "SHARD_GRAD_OP",
            "precision": "mixed",
        },
        "max_grad_norm": 1.0,
        "max_grad_norm_ratio": None,
        "speed_monitor": {"window_size": 1},
        "gen1_gc_interval": 10,
        "eval_interval": 1_000,
        "eval_subset_num_batches": -1,
        "device_eval_batch_size": 1 if boundary else 4,
        "evaluators": [],
        "data": {
            "pad_direction": "right",
            "generate_doc_lengths": True,
            "num_workers": 0 if boundary else 32,
            "drop_last": True,
            "pin_memory": True,
            "prefetch_factor": None if boundary else 8,
            "persistent_workers": not boundary,
            "memmap_dtype": "uint16",
            "timeout": 0,
            "instance_filter": {
                "repetition_max_period": 13,
                "repetition_min_period": 1,
                "repetition_max_count": 32,
            },
            "paths": paths,
        },
    }


def build(
    schedule_path: Path,
    qualification_path: Path,
    tokenizer_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Write exact JSON-as-YAML configs; never invoke the training entry point."""

    if output_root.exists() or output_root.is_symlink():
        raise OneBOlmoConfigError("OLMo config output exists")
    schedule = _load_signed(schedule_path, SCHEDULE_SCHEMA)
    qualification = _load_signed(qualification_path, TOKENIZER_SCHEMA)
    if (
        qualification.get("status") != "qualified_production_48k"
        or sha256_tree(tokenizer_root) != qualification.get("tokenizer_identity_sha256")
        or schedule.get("tokenizer_identity_sha256")
        != qualification.get("tokenizer_identity_sha256")
    ):
        raise OneBOlmoConfigError("production tokenizer gate differs")
    tokenizer_json = tokenizer_root / "tokenizer.json"
    if not tokenizer_json.is_file() or tokenizer_json.is_symlink():
        raise OneBOlmoConfigError("production tokenizer JSON differs")
    eos_token_id, pad_token_id = _special_ids(tokenizer_root)
    stage_root = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage_root.mkdir(parents=True)
    descriptors = []
    cumulative_tokens = 0
    try:
        for stage in schedule["stages"]:
            body_tokens = stage["body_sequences"] * 4_096
            boundary_tokens = stage["boundary_batch_sequences"] * 4_096
            body_end = cumulative_tokens + body_tokens
            boundary_end = body_end + boundary_tokens
            if stage["index"] == 0:
                body_load = None
            else:
                body_load = "__REQUIRED_PREVIOUS_STAGE_BOUNDARY_CHECKPOINT__"
            phases = (
                (
                    "body",
                    stage["body_entries"],
                    body_end,
                    body_load,
                ),
                (
                    "boundary",
                    stage["boundary_entries"],
                    boundary_end,
                    "__REQUIRED_CURRENT_STAGE_BODY_CHECKPOINT__",
                ),
            )
            for phase, entries, max_tokens, load_path in phases:
                config = _config(
                    stage=stage,
                    phase=phase,
                    paths=_paths(entries),
                    tokenizer_json=tokenizer_json,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    max_duration_tokens=max_tokens,
                    load_path=load_path,
                )
                path = stage_root / f"stage-{stage['index']}-{phase}.json"
                _atomic_create(path, config)
                descriptors.append(
                    {
                        "stage": stage["index"],
                        "phase": phase,
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                        "global_train_batch_size": config["global_train_batch_size"],
                        "max_duration": config["max_duration"],
                        "load_path_requires_resolution": load_path is not None,
                    }
                )
            cumulative_tokens = boundary_end
        contract = build_contract()
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_1b_olmo_config_bundle",
            "schedule_receipt_sha256": schedule["receipt_sha256"],
            "tokenizer_qualification_receipt_sha256": qualification["receipt_sha256"],
            "tokenizer_identity_sha256": qualification["tokenizer_identity_sha256"],
            "production_contract_receipt_sha256": contract["receipt_sha256"],
            "upstream": {
                "allenai_olmo_commit": OLMO_COMMIT,
                "allenai_olmo_core_commit": OLMO_CORE_COMMIT,
            },
            "configs": descriptors,
            "configs_sha256": canonical_sha256(descriptors),
            "exact_body_and_boundary_batches": True,
            "document_boundary_isolation_enabled": True,
            "unresolved_checkpoint_placeholders_expected": 9,
            "model_training_started": False,
            "one_b_training_authorized": False,
        }
        if cumulative_tokens != contract["target_tokens"]:
            raise OneBOlmoConfigError("OLMo config token horizon differs")
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage_root / "receipt.json", payload)
        os.replace(stage_root, output_root)
        return payload
    except BaseException:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    value = build(
        args.schedule, args.qualification, args.tokenizer_root, args.output_root
    )
    print(json.dumps({"receipt_sha256": value["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
