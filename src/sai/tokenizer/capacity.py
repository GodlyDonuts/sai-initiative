"""Measure unused multilingual vocabulary before any Sai tokenizer surgery."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

TEXT_FIELDS = (
    "question",
    "problem",
    "prompt",
    "instruction",
    "response",
    "solution",
    "completion",
    "output",
    "answer",
    "text",
)
UNSUPPORTED_SCRIPTS = (
    "CJK",
    "HIRAGANA",
    "KATAKANA",
    "HANGUL",
    "CYRILLIC",
    "ARABIC",
    "HEBREW",
    "DEVANAGARI",
    "BENGALI",
    "GURMUKHI",
    "GUJARATI",
    "ORIYA",
    "TAMIL",
    "TELUGU",
    "KANNADA",
    "MALAYALAM",
    "SINHALA",
    "THAI",
    "LAO",
    "MYANMAR",
    "GEORGIAN",
    "ARMENIAN",
)


class TokenizerAuditError(RuntimeError):
    """The tokenizer or admitted corpus cannot support a safe proposal."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_texts(row: Any) -> Iterable[str]:
    if isinstance(row, dict):
        for field in TEXT_FIELDS:
            value = row.get(field)
            if isinstance(value, str) and value:
                yield value


def unsupported_script(text: str) -> str | None:
    """Classify only tokens whose every letter belongs to one excluded script."""

    scripts: set[str] = set()
    letters = 0
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        letters += 1
        name = unicodedata.name(character, "")
        matches = [script for script in UNSUPPORTED_SCRIPTS if script in name]
        if len(matches) != 1:
            return None
        scripts.add(matches[0])
    return next(iter(scripts)) if letters and len(scripts) == 1 else None


def audit_files(tokenizer: Any, paths: list[Path]) -> tuple[dict[str, Any], Counter]:
    usage: Counter[int] = Counter()
    reports: dict[str, Any] = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise TokenizerAuditError(f"corpus is missing or unsafe: {path}")
        rows = texts = characters = tokens = malformed = roundtrip_failures = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                for text in row_texts(row):
                    texts += 1
                    characters += len(text)
                    token_ids = [
                        int(value)
                        for value in tokenizer.encode(text, add_special_tokens=False)
                    ]
                    if not token_ids:
                        raise TokenizerAuditError("nonempty text encoded to no tokens")
                    tokens += len(token_ids)
                    usage.update(token_ids)
                    roundtrip_failures += int(
                        tokenizer.decode(
                            token_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        != text
                    )
        if not rows or not texts or not characters or not tokens:
            raise TokenizerAuditError(f"corpus contains no auditable text: {path}")
        reports[str(path.resolve())] = {
            "sha256": sha256_file(path),
            "rows": rows,
            "texts": texts,
            "characters": characters,
            "tokens": tokens,
            "tokens_per_1k_characters": 1000.0 * tokens / characters,
            "characters_per_token": characters / tokens,
            "malformed_rows": malformed,
            "roundtrip_failures": roundtrip_failures,
        }
    return reports, usage


def propose_reduction(
    tokenizer: Any,
    usage: Counter,
    *,
    hidden_size: int,
    tied_embeddings: bool,
) -> dict[str, Any]:
    """Propose only zero-use, single-unsupported-script token removals."""

    vocabulary = tokenizer.get_vocab()
    if hidden_size <= 0 or not isinstance(vocabulary, dict) or not vocabulary:
        raise TokenizerAuditError("tokenizer capacity geometry differs")
    ids = [int(value) for value in vocabulary.values()]
    if len(ids) != len(set(ids)) or min(ids) < 0:
        raise TokenizerAuditError("tokenizer vocabulary IDs differ")
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    removable: list[dict[str, Any]] = []
    scripts: Counter[str] = Counter()
    for token_id in sorted(ids):
        if token_id in special_ids or usage[token_id] > 0:
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not decoded or "�" in decoded:
            continue
        script = unsupported_script(decoded)
        if script is None:
            continue
        removable.append(
            {
                "id": token_id,
                "token": tokenizer.convert_ids_to_tokens(token_id),
                "decoded": decoded,
                "script": script,
            }
        )
        scripts[script] += 1
    matrices = 1 if tied_embeddings else 2
    recovered = len(removable) * hidden_size * matrices
    return {
        "original_vocabulary_size": len(ids),
        "used_token_ids": len(usage),
        "special_token_ids": len(special_ids),
        "removable_token_count": len(removable),
        "candidate_vocabulary_size": len(ids) - len(removable),
        "removable_fraction": len(removable) / len(ids),
        "removable_by_script": dict(sorted(scripts.items())),
        "embedding_matrices": matrices,
        "hidden_size": hidden_size,
        "estimated_parameters_recovered": recovered,
        "estimated_bf16_bytes_recovered": recovered * 2,
        "removable_tokens": removable,
    }


def audit(
    tokenizer: Any,
    corpora: list[Path],
    evaluation_prompts: list[Path],
    *,
    hidden_size: int,
    tied_embeddings: bool,
) -> dict[str, Any]:
    """Bind admitted text use and return a proposal, never a trained tokenizer."""

    if not corpora or not evaluation_prompts:
        raise TokenizerAuditError("training and evaluation text are both required")
    corpus_reports, corpus_usage = audit_files(tokenizer, corpora)
    evaluation_reports, evaluation_usage = audit_files(tokenizer, evaluation_prompts)
    usage = corpus_usage + evaluation_usage
    proposal = propose_reduction(
        tokenizer,
        usage,
        hidden_size=hidden_size,
        tied_embeddings=tied_embeddings,
    )
    roundtrip_failures = sum(
        report["roundtrip_failures"]
        for report in (*corpus_reports.values(), *evaluation_reports.values())
    )
    return {
        "schema": "sai-4b-tokenizer-capacity-audit-v1",
        "status": "complete",
        "corpora": corpus_reports,
        "evaluation_prompts": evaluation_reports,
        "candidate": proposal,
        "checks": {
            "all_text_roundtrips_exactly": roundtrip_failures == 0,
            "all_evaluation_tokens_protected": all(
                usage[token_id] > 0 for token_id in evaluation_usage
            ),
            "all_special_tokens_protected": True,
        },
        "candidate_build_authorized": roundtrip_failures == 0
        and proposal["removable_token_count"] > 0,
        "scientific_training_authorized": False,
    }
