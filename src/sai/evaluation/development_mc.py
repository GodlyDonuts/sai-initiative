"""Fail-closed normalized-choice-likelihood development evaluation.

This module deliberately does not implement an official MMLU-Pro or MuSR
terminal score.  It accepts one canonical, explicitly versioned row shape for
each benchmark and emits development-only evidence that cannot authorize model
promotion by itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import uuid
from collections import defaultdict
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = "sai-development-mc-likelihood-v1"
DISJOINT_RECEIPT_SCHEMA = "sai-development-mc-source-disjoint-v1"
DECODING_CONTRACT = {
    "schema": "sai-development-mc-decoding-contract-v1",
    "mode": "teacher-forced-choice-continuation-likelihood",
    "generation": False,
    "sampling": False,
    "temperature": None,
    "max_generated_tokens": 0,
}
SCORING_CONTRACT = {
    "schema": "sai-development-mc-scoring-contract-v1",
    "purpose": "source-disjoint-development-screen",
    "choice_score": "sum_natural_log_probability_divided_by_choice_token_count",
    "selection": "highest_normalized_choice_log_likelihood_argmax_first_on_tie",
    "correctness": "selected_choice_text_equals_answer_choice_text",
    "prompt_special_tokens": False,
    "choice_special_tokens": False,
    "prompt_choice_boundary": "prompt_token_ids_must_prefix_prompt_plus_choice_ids",
    "mmlu_pro_prompt": "Question: {question}\\nAnswer:",
    "musr_prompt": "Passage: {context}\\nQuestion: {question}\\nAnswer:",
    "choice_continuation": " {choice}",
    "official_benchmark_result": False,
    "public_terminal_result": False,
    "architecture_promotion_allowed": False,
}
SUPPORTED_KEYS = {
    "mmlu_pro": {
        "benchmark",
        "row_id",
        "domain",
        "question",
        "choices",
        "answer_index",
    },
    "musr": {
        "benchmark",
        "row_id",
        "domain",
        "context",
        "question",
        "choices",
        "answer_index",
    },
}


class DevelopmentMCError(RuntimeError):
    """The development population, artifact binding, or score differs."""


class TokenizerProtocol(Protocol):
    """Minimal tokenizer surface required by the evaluator."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SCORING_CONTRACT_SHA256 = _canonical_sha256(SCORING_CONTRACT)
DECODING_CONTRACT_SHA256 = _canonical_sha256(DECODING_CONTRACT)


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DevelopmentMCError(f"{field} must be a lowercase SHA256")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DevelopmentMCError(f"{field} must be a positive integer")
    return value


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise DevelopmentMCError(f"artifact is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_sha256(paths: Sequence[Path], field: str) -> str:
    if not paths:
        raise DevelopmentMCError(f"{field} bundle is empty")
    rows = []
    resolved_seen: set[str] = set()
    for path_value in paths:
        path = Path(path_value)
        resolved = str(path.resolve())
        if resolved in resolved_seen:
            raise DevelopmentMCError(f"{field} bundle repeats an artifact")
        resolved_seen.add(resolved)
        rows.append(
            {
                "name": path.name,
                "bytes": (
                    path.stat().st_size
                    if path.is_file() and not path.is_symlink()
                    else None
                ),
                "sha256": _sha256_file(path),
            }
        )
    return _canonical_sha256(rows)


def _load_rows(source_path: Path) -> tuple[list[dict[str, Any]], str]:
    source_path = Path(source_path)
    source_sha256 = _sha256_file(source_path)
    rows: list[dict[str, Any]] = []
    try:
        with source_path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    raise DevelopmentMCError(f"blank source row at line {line_number}")
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise DevelopmentMCError(
                        f"source row {line_number} must be an object"
                    )
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DevelopmentMCError("benchmark source is unreadable") from error
    if not rows:
        raise DevelopmentMCError("benchmark source is empty")
    return rows, source_sha256


def _validate_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DevelopmentMCError(f"{field} must be a nonempty stripped string")
    return value


def _validate_row(row: dict[str, Any], benchmark: str) -> dict[str, Any]:
    if benchmark not in SUPPORTED_KEYS or set(row) != SUPPORTED_KEYS[benchmark]:
        raise DevelopmentMCError("benchmark row schema is unsupported")
    if row.get("benchmark") != benchmark:
        raise DevelopmentMCError("benchmark row identity differs")
    for field in ("row_id", "domain", "question"):
        _validate_text(row.get(field), field)
    if benchmark == "musr":
        _validate_text(row.get("context"), "context")
    choices = row.get("choices")
    if (
        not isinstance(choices, list)
        or not 2 <= len(choices) <= 16
        or any(
            not isinstance(choice, str) or not choice or choice != choice.strip()
            for choice in choices
        )
    ):
        raise DevelopmentMCError("choices must contain 2-16 nonempty stripped strings")
    answer_index = row.get("answer_index")
    if (
        isinstance(answer_index, bool)
        or not isinstance(answer_index, int)
        or not 0 <= answer_index < len(choices)
    ):
        raise DevelopmentMCError("answer_index differs")
    return row


def _validate_disjoint_receipt(
    receipt_path: Path,
    *,
    benchmark: str,
    benchmark_source_sha256: str,
    training_source_sha256: str,
) -> str:
    receipt_sha256 = _sha256_file(receipt_path)
    try:
        receipt = json.loads(Path(receipt_path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DevelopmentMCError("source-disjoint receipt is unreadable") from error
    expected_keys = {
        "schema",
        "benchmark",
        "benchmark_source_sha256",
        "training_source_sha256",
        "source_disjoint",
        "method",
        "evidence_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise DevelopmentMCError("source-disjoint receipt schema differs")
    if (
        receipt.get("schema") != DISJOINT_RECEIPT_SCHEMA
        or receipt.get("benchmark") != benchmark
        or receipt.get("benchmark_source_sha256") != benchmark_source_sha256
        or receipt.get("training_source_sha256") != training_source_sha256
        or receipt.get("source_disjoint") is not True
        or receipt.get("method") != "identity-and-contamination-audit"
    ):
        raise DevelopmentMCError("source-disjoint evidence differs")
    _sha256(receipt.get("evidence_sha256"), "disjoint evidence SHA256")
    return receipt_sha256


def _model_versions(model: nn.Module) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, tensor._version)
        for name, tensor in sorted(model.state_dict(keep_vars=True).items())
    )


def _prompt(row: dict[str, Any], benchmark: str) -> str:
    if benchmark == "mmlu_pro":
        return f"Question: {row['question']}\nAnswer:"
    return f"Passage: {row['context']}\nQuestion: {row['question']}\nAnswer:"


def _tokenize(tokenizer: TokenizerProtocol, text: str) -> list[int]:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except (AttributeError, TypeError, ValueError) as error:
        raise DevelopmentMCError("tokenizer encode contract differs") from error
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in tokens
        )
    ):
        raise DevelopmentMCError("tokenizer returned invalid token IDs")
    return tokens


def _score_choice(
    model: nn.Module,
    tokenizer: TokenizerProtocol,
    prompt: str,
    choice: str,
    *,
    device: torch.device,
    max_sequence_tokens: int,
) -> dict[str, Any]:
    prompt_ids = _tokenize(tokenizer, prompt)
    combined_ids = _tokenize(tokenizer, f"{prompt} {choice}")
    if combined_ids[: len(prompt_ids)] != prompt_ids:
        raise DevelopmentMCError(
            "tokenizer prompt/choice boundary is not prefix-stable"
        )
    choice_ids = combined_ids[len(prompt_ids) :]
    if not choice_ids or len(combined_ids) > max_sequence_tokens:
        raise DevelopmentMCError("choice token coverage differs")
    input_ids = torch.tensor([combined_ids[:-1]], dtype=torch.long, device=device)
    segment_ids = torch.zeros_like(input_ids)
    choice_logits = getattr(model, "choice_logits", None)
    if callable(choice_logits):
        selected_logits = choice_logits(
            input_ids,
            segment_ids,
            start_position=len(prompt_ids) - 1,
            token_count=len(choice_ids),
        )
        if (
            not isinstance(selected_logits, torch.Tensor)
            or selected_logits.ndim != 2
            or selected_logits.shape[0] != len(choice_ids)
        ):
            raise DevelopmentMCError("specialized choice-logit geometry differs")
        vocab_size = selected_logits.shape[-1]
    else:
        logits = model(input_ids, segment_ids)
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.shape[:2] != input_ids.shape
        ):
            raise DevelopmentMCError("model logit geometry differs")
        vocab_size = logits.shape[-1]
        start = len(prompt_ids) - 1
        selected_logits = logits[0, start : start + len(choice_ids)]
    targets = torch.tensor(choice_ids, dtype=torch.long, device=device)
    if bool((targets >= vocab_size).any().item()):
        raise DevelopmentMCError("choice token exceeds model vocabulary")
    selected = F.log_softmax(selected_logits.float(), dim=-1)
    log_probability_sum = float(selected.gather(1, targets[:, None]).sum())
    normalized = log_probability_sum / len(choice_ids)
    if not math.isfinite(log_probability_sum) or not math.isfinite(normalized):
        raise DevelopmentMCError("choice likelihood is nonfinite")
    return {
        "choice_token_count": len(choice_ids),
        "log_probability_sum": log_probability_sum,
        "normalized_log_likelihood": normalized,
    }


def evaluate_development_mc(
    model: nn.Module,
    tokenizer: TokenizerProtocol,
    *,
    benchmark: str,
    source_path: Path,
    disjoint_receipt_path: Path,
    training_source_sha256: str,
    checkpoint_paths: Sequence[Path],
    config_paths: Sequence[Path],
    tokenizer_paths: Sequence[Path],
    runtime_paths: Sequence[Path],
    expected_rows: int,
    expected_identity_order_sha256: str,
    max_sequence_tokens: int,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Score an exact development population without mutating model state."""

    if not isinstance(model, nn.Module) or benchmark not in SUPPORTED_KEYS:
        raise DevelopmentMCError("model or benchmark contract differs")
    expected_rows = _positive_integer(expected_rows, "expected rows")
    max_sequence_tokens = _positive_integer(
        max_sequence_tokens, "maximum sequence tokens"
    )
    if autocast_dtype is not None and autocast_dtype is not torch.bfloat16:
        raise DevelopmentMCError("evaluation autocast dtype differs")
    expected_identity_order_sha256 = _sha256(
        expected_identity_order_sha256, "expected identity-order SHA256"
    )
    training_source_sha256 = _sha256(training_source_sha256, "training source SHA256")
    rows, source_sha256 = _load_rows(source_path)
    if len(rows) != expected_rows:
        raise DevelopmentMCError("benchmark row coverage differs")
    validated = [_validate_row(row, benchmark) for row in rows]
    identities = [row["row_id"] for row in validated]
    if len(set(identities)) != len(identities):
        raise DevelopmentMCError("benchmark row identities are duplicated")
    identity_order_sha256 = _canonical_sha256(identities)
    if identity_order_sha256 != expected_identity_order_sha256:
        raise DevelopmentMCError("benchmark identity order differs")
    disjoint_receipt_sha256 = _validate_disjoint_receipt(
        disjoint_receipt_path,
        benchmark=benchmark,
        benchmark_source_sha256=source_sha256,
        training_source_sha256=training_source_sha256,
    )
    checkpoint_sha256 = _bundle_sha256(checkpoint_paths, "checkpoint")
    config_sha256 = _bundle_sha256(config_paths, "config")
    tokenizer_sha256 = _bundle_sha256(tokenizer_paths, "tokenizer")
    runtime_files_sha256 = _bundle_sha256(runtime_paths, "runtime")
    evaluator_code_sha256 = _sha256_file(Path(__file__))
    runtime_sha256 = _canonical_sha256(
        {
            "evaluator_code_sha256": evaluator_code_sha256,
            "runtime_files_sha256": runtime_files_sha256,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
        }
    )

    original_training = model.training
    versions_before = _model_versions(model)
    device = next(model.parameters()).device
    if autocast_dtype is not None and device.type != "cuda":
        raise DevelopmentMCError("evaluation autocast requires CUDA")
    scored_rows: list[dict[str, Any]] = []
    model.eval()
    try:
        autocast = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            for row in validated:
                prompt = _prompt(row, benchmark)
                choice_scores = [
                    _score_choice(
                        model,
                        tokenizer,
                        prompt,
                        choice,
                        device=device,
                        max_sequence_tokens=max_sequence_tokens,
                    )
                    for choice in row["choices"]
                ]
                prediction = max(
                    range(len(choice_scores)),
                    key=lambda index: choice_scores[index]["normalized_log_likelihood"],
                )
                scored_rows.append(
                    {
                        "row_id": row["row_id"],
                        "domain": row["domain"],
                        "answer_index": row["answer_index"],
                        "predicted_index": prediction,
                        "correct": (
                            row["choices"][prediction]
                            == row["choices"][row["answer_index"]]
                        ),
                        "choice_scores": choice_scores,
                    }
                )
    finally:
        model.train(original_training)
    if _model_versions(model) != versions_before:
        raise DevelopmentMCError("evaluation mutated model state")

    domains: dict[str, list[bool]] = defaultdict(list)
    for row in scored_rows:
        domains[row["domain"]].append(row["correct"])
    correct = sum(row["correct"] for row in scored_rows)
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "benchmark": benchmark,
        "development_only": True,
        "official_benchmark_result": False,
        "public_terminal_result": False,
        "architecture_promotion_allowed": False,
        "bindings": {
            "benchmark_source_sha256": source_sha256,
            "training_source_sha256": training_source_sha256,
            "source_disjoint_receipt_sha256": disjoint_receipt_sha256,
            "identity_order_sha256": identity_order_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "evaluator_code_sha256": evaluator_code_sha256,
            "runtime_files_sha256": runtime_files_sha256,
            "runtime_sha256": runtime_sha256,
            "decoding_contract_sha256": DECODING_CONTRACT_SHA256,
            "scoring_contract_sha256": SCORING_CONTRACT_SHA256,
        },
        "decoding_contract": DECODING_CONTRACT,
        "scoring_contract": SCORING_CONTRACT,
        "coverage": {"expected_rows": expected_rows, "scored_rows": len(scored_rows)},
        "aggregate": {
            "correct": correct,
            "rows": len(scored_rows),
            "accuracy": correct / len(scored_rows),
        },
        "domains": {
            domain: {
                "correct": sum(values),
                "rows": len(values),
                "accuracy": sum(values) / len(values),
            }
            for domain, values in sorted(domains.items())
        },
        "rows": scored_rows,
    }
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def write_development_mc(output_path: Path, payload: dict[str, Any]) -> None:
    """Create one immutable JSON result using an atomic same-directory rename."""

    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise DevelopmentMCError("development result already exists")
    if not output_path.parent.is_dir() or output_path.parent.is_symlink():
        raise DevelopmentMCError("development result parent is missing or unsafe")
    if payload.get("schema") != SCHEMA or payload.get("status") != "complete":
        raise DevelopmentMCError("development result payload differs")
    temporary = output_path.parent / f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output_path)
        temporary.unlink()
        directory = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise DevelopmentMCError("development result already exists") from error
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
