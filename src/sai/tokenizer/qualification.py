"""Qualify Sai tokenizer candidates on exact corpora and protected strings."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from sai.data.token_stream import (
    TokenStreamError,
    canonical_sha256,
    normalize_document,
    sha256_file,
    sha256_tree,
)

SCHEMA = "sai-tokenizer-tournament-report-v1"
RECEIPT_SCHEMA = "sai-tokenizer-qualification-receipt-v1"
PROTECTED_SCHEMA = "sai-tokenizer-protected-string-v1"
CANDIDATE_SIZES = {"64k": 64_000, "48k": 48_000, "32k": 32_000}
PROTECTED_CATEGORIES = {
    "ascii",
    "english",
    "code",
    "whitespace_and_indentation",
    "identifiers",
    "urls",
    "numbers_and_units",
    "greek",
    "math",
    "latex",
    "science_notation",
    "unicode_fallback",
}


class TokenizerQualificationError(RuntimeError):
    """A tokenizer candidate, corpus, protected suite, or selection differs."""


class TokenizerCandidate(Protocol):
    eos_token_id: int | None
    unk_token_id: int | None
    all_special_ids: list[int]
    byte_fallback: bool

    def get_vocab(self) -> dict[str, int]: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...

    def convert_ids_to_tokens(self, token_id: int) -> str: ...


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TokenizerQualificationError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise TokenizerQualificationError(f"{field} differs") from error
    return value


def _source_receipts(paths: list[Path]) -> list[dict[str, Any]]:
    if not paths:
        raise TokenizerQualificationError("at least one admitted corpus is required")
    receipts = []
    seen = set()
    for order, path in enumerate(paths):
        if not path.is_file() or path.is_symlink():
            raise TokenizerQualificationError(f"corpus is missing or unsafe: {path}")
        resolved = str(path.resolve())
        if resolved in seen or path.stat().st_size <= 0:
            raise TokenizerQualificationError(
                "corpus paths must be unique and nonempty"
            )
        seen.add(resolved)
        receipts.append(
            {
                "order": order,
                "path": resolved,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return receipts


def _load_corpus_texts(paths: list[Path]) -> tuple[list[tuple[str, str]], Counter[str]]:
    texts = []
    domains: Counter[str] = Counter()
    identities = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise TokenizerQualificationError(
                        "corpus row is malformed"
                    ) from error
                try:
                    normalized = normalize_document(row)
                except TokenStreamError as error:
                    raise TokenizerQualificationError(
                        "corpus row contract differs"
                    ) from error
                identity = normalized["identity_sha256"]
                if identity in identities:
                    raise TokenizerQualificationError(
                        "corpus document identities are not unique"
                    )
                identities.add(identity)
                domain = normalized["source"]["domain"]
                texts.append((domain, normalized["text"]))
                domains[domain] += 1
    if not texts:
        raise TokenizerQualificationError("admitted corpus contains no text")
    return texts, domains


def _load_protected_suite(
    path: Path,
) -> tuple[list[tuple[str, str, str]], dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise TokenizerQualificationError("protected suite is missing or unsafe")
    rows = []
    identities = set()
    categories = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TokenizerQualificationError(
                    "protected suite row is malformed"
                ) from error
            if (
                not isinstance(row, dict)
                or row.get("schema") != PROTECTED_SCHEMA
                or not isinstance(row.get("id"), str)
                or not row["id"]
                or row["id"] in identities
                or row.get("category") not in PROTECTED_CATEGORIES
                or not isinstance(row.get("text"), str)
                or not row["text"]
            ):
                raise TokenizerQualificationError(
                    "protected suite row contract differs"
                )
            identities.add(row["id"])
            categories.add(row["category"])
            rows.append((row["id"], row["category"], row["text"]))
    if not rows or categories != PROTECTED_CATEGORIES:
        raise TokenizerQualificationError("protected categories are incomplete")
    receipt = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "identity_sha256": canonical_sha256([row[0] for row in rows]),
        "categories": sorted(categories),
    }
    return rows, receipt


def _vocabulary(
    tokenizer: TokenizerCandidate, expected_size: int
) -> tuple[set[int], tuple[str, ...]]:
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict) or len(vocabulary) != expected_size:
        raise TokenizerQualificationError("candidate vocabulary size differs")
    ids = list(vocabulary.values())
    if (
        any(isinstance(value, bool) or not isinstance(value, int) for value in ids)
        or len(ids) != len(set(ids))
        or set(ids) != set(range(expected_size))
    ):
        raise TokenizerQualificationError("candidate vocabulary IDs are not contiguous")
    special_ids = getattr(tokenizer, "all_special_ids", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if (
        not isinstance(special_ids, list)
        or not special_ids
        or any(value not in ids for value in special_ids)
        or isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or eos_token_id not in special_ids
    ):
        raise TokenizerQualificationError("candidate special/EOS tokens differ")
    special_tokens = tuple(
        sorted(tokenizer.convert_ids_to_tokens(token_id) for token_id in special_ids)
    )
    if any(not isinstance(token, str) or not token for token in special_tokens) or len(
        special_tokens
    ) != len(set(special_tokens)):
        raise TokenizerQualificationError("candidate special-token strings differ")
    return set(ids), special_tokens


def _measure_texts(
    tokenizer: TokenizerCandidate,
    vocabulary_ids: set[int],
    rows: list[tuple[str, str]],
) -> dict[str, Any]:
    domain_bytes: Counter[str] = Counter()
    domain_tokens: Counter[str] = Counter()
    domain_texts: Counter[str] = Counter()
    roundtrip_failures = unknown_tokens = empty_encodings = 0
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for domain, text in rows:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(token_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in token_ids
        ):
            raise TokenizerQualificationError("candidate token IDs differ")
        if not token_ids:
            empty_encodings += 1
            continue
        if any(value not in vocabulary_ids for value in token_ids):
            raise TokenizerQualificationError("candidate emitted an invalid token ID")
        decoded = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        roundtrip_failures += int(decoded != text)
        if isinstance(unk_token_id, int):
            unknown_tokens += token_ids.count(unk_token_id)
        domain_texts[domain] += 1
        domain_bytes[domain] += len(text.encode("utf-8"))
        domain_tokens[domain] += len(token_ids)
    total_bytes = sum(domain_bytes.values())
    total_tokens = sum(domain_tokens.values())
    if total_bytes <= 0 or total_tokens <= 0:
        raise TokenizerQualificationError(
            "candidate measured no corpus bytes or tokens"
        )
    return {
        "texts": sum(domain_texts.values()),
        "utf8_bytes": total_bytes,
        "tokens": total_tokens,
        "tokens_per_1k_utf8_bytes": 1_000.0 * total_tokens / total_bytes,
        "utf8_bytes_per_token": total_bytes / total_tokens,
        "roundtrip_failures": roundtrip_failures,
        "unknown_tokens": unknown_tokens,
        "empty_encodings": empty_encodings,
        "by_domain": {
            domain: {
                "texts": domain_texts[domain],
                "utf8_bytes": domain_bytes[domain],
                "tokens": domain_tokens[domain],
                "tokens_per_1k_utf8_bytes": 1_000.0
                * domain_tokens[domain]
                / domain_bytes[domain],
            }
            for domain in sorted(domain_texts)
        },
    }


def qualify(
    candidates: dict[str, TokenizerCandidate],
    candidate_identities: dict[str, str],
    corpora: list[Path],
    protected_suite: Path,
) -> dict[str, Any]:
    """Measure exact candidate fertility and lossless protected behavior."""

    if set(candidates) != set(CANDIDATE_SIZES) or set(candidate_identities) != set(
        CANDIDATE_SIZES
    ):
        raise TokenizerQualificationError("exact 64K/48K/32K candidates are required")
    source_receipts = _source_receipts(corpora)
    corpus_rows, corpus_domains = _load_corpus_texts(corpora)
    protected_rows, protected_receipt = _load_protected_suite(protected_suite)
    protected_texts = [(category, text) for _, category, text in protected_rows]
    corpus_identity = canonical_sha256(
        {
            "sources": source_receipts,
            "protected_suite": protected_receipt,
            "domain_documents": dict(sorted(corpus_domains.items())),
        }
    )
    candidate_reports = {}
    special_contract: tuple[str, ...] | None = None
    for name in CANDIDATE_SIZES:
        tokenizer = candidates[name]
        identity = _sha256(candidate_identities[name], f"{name} tokenizer identity")
        vocabulary_ids, candidate_specials = _vocabulary(
            tokenizer, CANDIDATE_SIZES[name]
        )
        if special_contract is None:
            special_contract = candidate_specials
        elif candidate_specials != special_contract:
            raise TokenizerQualificationError(
                "candidate special-token contracts are not identical"
            )
        corpus_metrics = _measure_texts(tokenizer, vocabulary_ids, corpus_rows)
        protected_metrics = _measure_texts(tokenizer, vocabulary_ids, protected_texts)
        declared_byte_fallback = getattr(tokenizer, "byte_fallback", None) is True
        qualified = declared_byte_fallback and all(
            metrics[field] == 0
            for metrics in (corpus_metrics, protected_metrics)
            for field in ("roundtrip_failures", "unknown_tokens", "empty_encodings")
        )
        candidate_reports[name] = {
            "name": name,
            "vocab_size": CANDIDATE_SIZES[name],
            "tokenizer_identity_sha256": identity,
            "corpus": corpus_metrics,
            "protected_suite": protected_metrics,
            "byte_fallback": declared_byte_fallback,
            "special_tokens_preserved": True,
            "special_token_contract_sha256": canonical_sha256(candidate_specials),
            "qualified": qualified,
        }
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "all_candidates_qualified"
            if all(row["qualified"] for row in candidate_reports.values())
            else "candidate_rejected"
        ),
        "training_authorized": False,
        "candidate_build_authorized": False,
        "source_receipts": source_receipts,
        "protected_suite_receipt": protected_receipt,
        "corpus_identity_sha256": corpus_identity,
        "candidates": candidate_reports,
        "checks": {
            "same_corpus_and_protected_suite": True,
            "vocabulary_sizes_exact": True,
            "byte_normalized_metrics_reported": True,
            "selection_not_automatic": True,
            "training_not_authorized": True,
        },
    }
    payload["report_sha256"] = canonical_sha256(payload)
    return payload


def selected_receipt(
    report: dict[str, Any], candidate_name: str = "48k"
) -> dict[str, Any]:
    if (
        not isinstance(report, dict)
        or report.get("schema") != SCHEMA
        or report.get("status") != "all_candidates_qualified"
        or report.get("training_authorized") is not False
        or report.get("report_sha256")
        != canonical_sha256(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
        or candidate_name != "48k"
    ):
        raise TokenizerQualificationError("qualified 48K tournament report is required")
    candidate = report.get("candidates", {}).get(candidate_name)
    if (
        not isinstance(candidate, dict)
        or candidate.get("qualified") is not True
        or candidate.get("vocab_size") != 48_000
        or candidate.get("byte_fallback") is not True
        or candidate.get("special_tokens_preserved") is not True
        or candidate.get("corpus", {}).get("roundtrip_failures") != 0
        or candidate.get("protected_suite", {}).get("roundtrip_failures") != 0
    ):
        raise TokenizerQualificationError("selected tokenizer evidence differs")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "qualified",
        "training_authorized": False,
        "vocab_size": 48_000,
        "tokenizer_identity_sha256": _sha256(
            candidate.get("tokenizer_identity_sha256"), "selected tokenizer identity"
        ),
        "corpus_identity_sha256": _sha256(
            report.get("corpus_identity_sha256"), "selected tokenizer corpus identity"
        ),
        "tournament_report_sha256": report["report_sha256"],
        "byte_fallback": True,
        "roundtrip_failures": 0,
        "special_tokens_preserved": True,
        "corpus_tokens_per_1k_utf8_bytes": candidate["corpus"][
            "tokens_per_1k_utf8_bytes"
        ],
        "protected_tokens_per_1k_utf8_bytes": candidate["protected_suite"][
            "tokens_per_1k_utf8_bytes"
        ],
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _candidate_mapping(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or name not in CANDIDATE_SIZES or name in result or not path:
            raise TokenizerQualificationError(f"invalid candidate mapping: {value}")
        result[name] = Path(path)
    if set(result) != set(CANDIDATE_SIZES):
        raise TokenizerQualificationError("exact candidate roots are required")
    return result


def declares_byte_fallback(root: Path) -> bool:
    """Return whether an exact local tokenizer tree declares lossless bytes."""
    tokenizer_json = root / "tokenizer.json"
    if not tokenizer_json.is_file() or tokenizer_json.is_symlink():
        return False
    try:
        payload = json.loads(tokenizer_json.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    model = payload.get("model") if isinstance(payload, dict) else None
    if isinstance(model, dict) and model.get("byte_fallback") is True:
        return True

    def contains_byte_level(value: Any) -> bool:
        if isinstance(value, dict):
            return value.get("type") == "ByteLevel" or any(
                contains_byte_level(child) for child in value.values()
            )
        if isinstance(value, list):
            return any(contains_byte_level(child) for child in value)
        return False

    return (
        isinstance(payload, dict)
        and contains_byte_level(payload.get("pre_tokenizer"))
        and contains_byte_level(payload.get("decoder"))
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--protected-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-48k-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.selected_48k_output.exists():
        raise TokenizerQualificationError("tokenizer output already exists")
    roots = _candidate_mapping(args.candidate)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TokenizerQualificationError(
            "Transformers is required to load tokenizer candidates"
        ) from error
    candidates = {}
    identities = {}
    for name, root in roots.items():
        identities[name] = sha256_tree(root)
        candidates[name] = AutoTokenizer.from_pretrained(
            root,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        if not getattr(candidates[name], "is_fast", False):
            raise TokenizerQualificationError(f"{name} tokenizer is not fast")
        candidates[name].byte_fallback = declares_byte_fallback(root)
    report = qualify(candidates, identities, args.corpus, args.protected_suite)
    if report["status"] != "all_candidates_qualified":
        raise TokenizerQualificationError("one or more tokenizer candidates failed")
    selection = selected_receipt(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.selected_48k_output.parent.mkdir(parents=True, exist_ok=True)
    report_tmp = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    selected_tmp = args.selected_48k_output.with_name(
        f".{args.selected_48k_output.name}.tmp.{os.getpid()}"
    )
    try:
        report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        selected_tmp.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        os.replace(report_tmp, args.output)
        os.replace(selected_tmp, args.selected_48k_output)
    except BaseException:
        report_tmp.unlink(missing_ok=True)
        selected_tmp.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "report_sha256": report["report_sha256"],
                "selected_receipt_sha256": selection["receipt_sha256"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
