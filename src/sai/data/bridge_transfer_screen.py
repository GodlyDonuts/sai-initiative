"""Run an equal-token proxy transfer screen for verified connection lessons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from array import array
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.grounded_bridge_verification_population import SCHEMA as POPULATION_SCHEMA
from sai.data.practical_bridge_reconcile import SCHEMA as RECONCILIATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-bridge-transfer-proxy-arm-v1"
MODEL_REPOSITORY = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
ARMS = ("unchanged", "source_control", "connections")
SEED = 20_260_826
BLOCK_SIZE = 512
MAXIMUM_TRAIN_TOKENS = 2_000_000
MAXIMUM_EVALUATION_TOKENS_PER_STREAM = 500_000
MICRO_BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.1
WARMUP_FRACTION = 0.1


class BridgeTransferScreenError(RuntimeError):
    """A transfer input, runtime, or equal-compute invariant differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise BridgeTransferScreenError("signed input is unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeTransferScreenError("signed input differs") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != schema
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise BridgeTransferScreenError("signed input differs")
    return payload


def _load_jsonl(path: Path, descriptor: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise BridgeTransferScreenError("bound JSONL differs")
    try:
        rows = [json.loads(line) for line in path.open()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BridgeTransferScreenError("bound JSONL rows differ") from error
    if len(rows) != descriptor.get("rows"):
        raise BridgeTransferScreenError("bound JSONL coverage differs")
    return rows


def build_text_sets(
    reconciliation_root: Path,
    population_root: Path,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Replay train/dev custody and build treatment, control, and eval texts."""

    reconciliation = _load_signed(
        reconciliation_root / "receipt.json", RECONCILIATION_SCHEMA
    )
    if (
        reconciliation.get("status")
        != "complete_practical_bridge_foundation_reconciliation"
        or reconciliation.get("global_exact_content_deduplication_complete") is not True
        or reconciliation.get("development_source_disjoint_against_foundation_complete")
        is not True
        or reconciliation.get("transfer_ablation_complete") is not False
        or reconciliation.get("training_ready") is not False
    ):
        raise BridgeTransferScreenError("reconciliation receipt differs")
    outputs = reconciliation.get("outputs")
    if not isinstance(outputs, dict):
        raise BridgeTransferScreenError("reconciliation outputs differ")
    train_rows = _load_jsonl(
        reconciliation_root / str(outputs.get("train", {}).get("path")),
        outputs.get("train", {}),
    )
    development_rows = _load_jsonl(
        reconciliation_root / str(outputs.get("development", {}).get("path")),
        outputs.get("development", {}),
    )
    if (
        not train_rows
        or not development_rows
        or any(row.get("corpus_split") != "train" for row in train_rows)
        or any(row.get("corpus_split") != "development" for row in development_rows)
    ):
        raise BridgeTransferScreenError("reconciled split differs")
    train_pairs = {row.get("pair_identity_sha256") for row in train_rows}
    development_pairs = {row.get("pair_identity_sha256") for row in development_rows}
    if (
        None in train_pairs
        or None in development_pairs
        or not development_pairs
        or train_pairs & development_pairs
    ):
        raise BridgeTransferScreenError("reconciled pair split differs")

    population = _load_signed(population_root / "receipt.json", POPULATION_SCHEMA)
    descriptor = population.get("candidates")
    if (
        population.get("status")
        != "complete_nontraining_bridge_verification_population"
        or population.get("source_disjoint_pairs") is not True
        or not isinstance(descriptor, dict)
    ):
        raise BridgeTransferScreenError("verification population differs")
    population_rows = _load_jsonl(
        population_root / str(descriptor.get("path")), descriptor
    )
    by_pair = {}
    ordered_identities = []
    for row in population_rows:
        pair = row.get("pair_identity_sha256")
        identity = row.get("candidate_identity_sha256")
        anchor_a = row.get("anchor_a_text")
        anchor_b = row.get("anchor_b_text")
        if (
            not isinstance(pair, str)
            or pair in by_pair
            or not isinstance(identity, str)
            or not isinstance(anchor_a, str)
            or not anchor_a.strip()
            or not isinstance(anchor_b, str)
            or not anchor_b.strip()
            or hashlib.sha256(anchor_a.encode()).hexdigest()
            != row.get("anchor_a_source_content_sha256")
            or hashlib.sha256(anchor_b.encode()).hexdigest()
            != row.get("anchor_b_source_content_sha256")
        ):
            raise BridgeTransferScreenError("verification anchor differs")
        by_pair[pair] = (anchor_a, anchor_b)
        ordered_identities.append(identity)
    if canonical_sha256(ordered_identities) != descriptor.get(
        "ordered_identities_sha256"
    ):
        raise BridgeTransferScreenError("verification population order differs")
    if not (train_pairs | development_pairs) <= set(by_pair):
        raise BridgeTransferScreenError("reconciled pair anchor coverage differs")

    connection_train = [row.get("text") for row in train_rows]
    connection_development = [row.get("text") for row in development_rows]
    if any(not isinstance(text, str) or not text.strip() for text in connection_train):
        raise BridgeTransferScreenError("connection train text differs")
    if any(
        not isinstance(text, str) or not text.strip() for text in connection_development
    ):
        raise BridgeTransferScreenError("connection development text differs")
    source_control = []
    for pair in sorted(train_pairs):
        anchor_a, anchor_b = by_pair[pair]
        source_control.extend(
            [f"Source document A\n\n{anchor_a}", f"Source document B\n\n{anchor_b}"]
        )
    development_anchors = []
    for pair in sorted(development_pairs):
        anchor_a, anchor_b = by_pair[pair]
        development_anchors.extend(
            [f"Source document A\n\n{anchor_a}", f"Source document B\n\n{anchor_b}"]
        )
    sets = {
        "connection_train": connection_train,
        "source_control_train": source_control,
        "connection_development": connection_development,
        "source_development": development_anchors,
    }
    lineage = {
        "reconciliation_receipt_sha256": reconciliation["receipt_sha256"],
        "verification_population_receipt_sha256": population["receipt_sha256"],
        "train_pairs": len(train_pairs),
        "development_pairs": len(development_pairs),
        "connection_train_documents": len(connection_train),
        "source_control_documents": len(source_control),
        "connection_development_documents": len(connection_development),
        "source_development_documents": len(development_anchors),
        "ordered_text_sha256": {
            key: canonical_sha256(
                [hashlib.sha256(text.encode()).hexdigest() for text in values]
            )
            for key, values in sets.items()
        },
    }
    return sets, lineage


def _tokenize(texts: list[str], tokenizer: Any) -> list[int]:
    tokens = []
    eos = tokenizer.eos_token_id
    if not isinstance(eos, int) or eos < 0:
        raise BridgeTransferScreenError("tokenizer EOS differs")
    for text in texts:
        values = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise BridgeTransferScreenError("tokenizer output differs")
        tokens.extend(values)
        tokens.append(eos)
    return tokens


def _token_sha256(tokens: list[int]) -> str:
    if sys.byteorder != "little" or any(not 0 <= token < 2**32 for token in tokens):
        raise BridgeTransferScreenError("token hash encoding differs")
    return hashlib.sha256(array("I", tokens).tobytes()).hexdigest()


def _chunks(tokens: list[int], maximum_tokens: int) -> tuple[list[list[int]], int]:
    usable = min(len(tokens), maximum_tokens)
    usable -= usable % BLOCK_SIZE
    if usable < BLOCK_SIZE:
        raise BridgeTransferScreenError("token stream is too small")
    values = [
        tokens[index : index + BLOCK_SIZE] for index in range(0, usable, BLOCK_SIZE)
    ]
    return values, usable


def _evaluate(model: Any, chunks: list[list[int]], torch: Any) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    predicted_tokens = 0
    with torch.no_grad():
        for index in range(0, len(chunks), MICRO_BATCH_SIZE):
            batch = torch.tensor(
                chunks[index : index + MICRO_BATCH_SIZE],
                dtype=torch.long,
                device="cuda",
            )
            output = model(input_ids=batch, labels=batch)
            count = batch.shape[0] * (batch.shape[1] - 1)
            total_loss += float(output.loss.detach().float().item()) * count
            predicted_tokens += count
    nll = total_loss / predicted_tokens
    return {
        "predicted_tokens": predicted_tokens,
        "mean_nll": nll,
        "perplexity": math.exp(min(nll, 80.0)),
    }


def _model_sha256(model: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run(
    arm: str,
    reconciliation_root: Path,
    population_root: Path,
    output: Path,
    token: str,
) -> dict[str, Any]:
    """Train/evaluate one arm with matched model, token, seed, and score custody."""

    if arm not in ARMS or output.exists() or output.is_symlink() or not token:
        raise BridgeTransferScreenError("screen arguments differ")
    try:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise BridgeTransferScreenError("screen runtime is incomplete") from error
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise BridgeTransferScreenError("H100 BF16 runtime is unavailable")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    sets, lineage = build_text_sets(reconciliation_root, population_root)
    snapshot = Path(
        snapshot_download(
            MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            token=token,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
    )
    model_files = []
    for path in sorted(snapshot.iterdir()):
        if path.is_file():
            model_files.append(
                {
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    connection_tokens = _tokenize(sets["connection_train"], tokenizer)
    control_tokens = _tokenize(sets["source_control_train"], tokenizer)
    budget = min(len(connection_tokens), len(control_tokens), MAXIMUM_TRAIN_TOKENS)
    budget -= budget % BLOCK_SIZE
    if budget < 100 * BLOCK_SIZE:
        raise BridgeTransferScreenError("matched training budget is too small")
    selected = connection_tokens if arm == "connections" else control_tokens
    train_chunks, used_train_tokens = _chunks(selected, budget)
    if used_train_tokens != budget:
        raise BridgeTransferScreenError("matched training budget differs")
    connection_eval_tokens = _tokenize(sets["connection_development"], tokenizer)
    source_eval_tokens = _tokenize(sets["source_development"], tokenizer)
    connection_eval, connection_eval_used = _chunks(
        connection_eval_tokens, MAXIMUM_EVALUATION_TOKENS_PER_STREAM
    )
    source_eval, source_eval_used = _chunks(
        source_eval_tokens, MAXIMUM_EVALUATION_TOKENS_PER_STREAM
    )
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    initial_model_sha256 = _model_sha256(model, torch)
    optimizer_steps = 0
    examples_seen = 0
    started = time.monotonic()
    if arm != "unchanged":
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            fused=True,
        )
        batches = math.ceil(len(train_chunks) / MICRO_BATCH_SIZE)
        planned_steps = math.ceil(batches / GRADIENT_ACCUMULATION)
        warmup_steps = max(1, round(planned_steps * WARMUP_FRACTION))
        generator = random.Random(SEED)
        order = list(range(len(train_chunks)))
        generator.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        for batch_index in range(0, len(order), MICRO_BATCH_SIZE):
            indexes = order[batch_index : batch_index + MICRO_BATCH_SIZE]
            batch = torch.tensor(
                [train_chunks[index] for index in indexes],
                dtype=torch.long,
                device="cuda",
            )
            loss = model(input_ids=batch, labels=batch).loss / GRADIENT_ACCUMULATION
            loss.backward()
            examples_seen += len(indexes)
            final_microbatch = batch_index + MICRO_BATCH_SIZE >= len(order)
            if (
                (batch_index // MICRO_BATCH_SIZE) + 1
            ) % GRADIENT_ACCUMULATION == 0 or final_microbatch:
                optimizer_steps += 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if optimizer_steps <= warmup_steps:
                    scale = optimizer_steps / warmup_steps
                else:
                    progress = (optimizer_steps - warmup_steps) / max(
                        1, planned_steps - warmup_steps
                    )
                    scale = 0.5 * (1.0 + math.cos(math.pi * progress))
                for group in optimizer.param_groups:
                    group["lr"] = LEARNING_RATE * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    trained_model_sha256 = _model_sha256(model, torch)
    evaluations = {
        "connection_development": _evaluate(model, connection_eval, torch),
        "source_development": _evaluate(model, source_eval, torch),
    }
    elapsed = time.monotonic() - started
    payload = {
        "schema": SCHEMA,
        "status": "complete_bridge_transfer_proxy_arm",
        "arm": arm,
        "lineage": lineage,
        "model": {
            "repository": MODEL_REPOSITORY,
            "revision": MODEL_REVISION,
            "files": model_files,
            "ordered_files_sha256": canonical_sha256(model_files),
            "initial_state_sha256": initial_model_sha256,
            "final_state_sha256": trained_model_sha256,
        },
        "tokenizer": {
            "vocabulary_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
        },
        "training": {
            "seed": SEED,
            "block_size": BLOCK_SIZE,
            "matched_token_budget": budget,
            "used_train_tokens": 0 if arm == "unchanged" else used_train_tokens,
            "selected_stream_sha256": (
                None
                if arm == "unchanged"
                else _token_sha256(selected[:used_train_tokens])
            ),
            "micro_batch_size": MICRO_BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "optimizer_steps": optimizer_steps,
            "examples_seen": examples_seen,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_fraction": WARMUP_FRACTION,
            "elapsed_seconds": elapsed,
        },
        "evaluation_streams": {
            "connection_development": {
                "tokens": connection_eval_used,
                "sha256": _token_sha256(connection_eval_tokens[:connection_eval_used]),
            },
            "source_development": {
                "tokens": source_eval_used,
                "sha256": _token_sha256(source_eval_tokens[:source_eval_used]),
            },
        },
        "evaluations": evaluations,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "screen_is_proxy_not_4b_capability_claim": True,
        "transfer_ablation_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--reconciliation-root", type=Path, required=True)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    result = run(
        args.arm,
        args.reconciliation_root,
        args.population_root,
        args.output,
        os.environ.get(args.token_env, ""),
    )
    print(
        json.dumps(
            {"arm": result["arm"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
