"""Audit what a Sai tokenizer tournament does and does not establish."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file
from sai.tokenizer.qualification import CANDIDATE_SIZES, PROTECTED_CATEGORIES, SCHEMA

EVIDENCE_SCHEMA = "sai-tokenizer-evidence-audit-v1"
REQUIRED_DOMAINS = ("english", "code", "math", "science", "technical")
DEFAULT_WIDTHS = (768, 2560)


class TokenizerEvidenceError(RuntimeError):
    """The tournament report or requested evidence audit differs."""


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TokenizerEvidenceError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise TokenizerEvidenceError(f"{field} differs") from error
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TokenizerEvidenceError(f"{field} differs")
    return value


def _positive_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise TokenizerEvidenceError(f"{field} differs")
    return float(value)


def _metric_block(value: object, field: str) -> dict[str, Any]:
    keys = {
        "texts",
        "utf8_bytes",
        "tokens",
        "tokens_per_1k_utf8_bytes",
        "utf8_bytes_per_token",
        "roundtrip_failures",
        "unknown_tokens",
        "empty_encodings",
        "by_domain",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise TokenizerEvidenceError(f"{field} contract differs")
    texts = _nonnegative_int(value["texts"], f"{field} texts")
    utf8_bytes = _nonnegative_int(value["utf8_bytes"], f"{field} bytes")
    tokens = _nonnegative_int(value["tokens"], f"{field} tokens")
    if texts <= 0 or utf8_bytes <= 0 or tokens <= 0:
        raise TokenizerEvidenceError(f"{field} is empty")
    tokens_per_1k = _positive_number(
        value["tokens_per_1k_utf8_bytes"], f"{field} tokens per byte"
    )
    bytes_per_token = _positive_number(
        value["utf8_bytes_per_token"], f"{field} bytes per token"
    )
    if not math.isclose(tokens_per_1k, 1_000.0 * tokens / utf8_bytes, rel_tol=1e-12):
        raise TokenizerEvidenceError(f"{field} fertility arithmetic differs")
    if not math.isclose(bytes_per_token, utf8_bytes / tokens, rel_tol=1e-12):
        raise TokenizerEvidenceError(f"{field} compression arithmetic differs")
    for key in ("roundtrip_failures", "unknown_tokens", "empty_encodings"):
        if _nonnegative_int(value[key], f"{field} {key}") != 0:
            raise TokenizerEvidenceError(f"{field} is not lossless")
    by_domain = value["by_domain"]
    if not isinstance(by_domain, dict) or not by_domain:
        raise TokenizerEvidenceError(f"{field} domains differ")
    domain_totals = {"texts": 0, "utf8_bytes": 0, "tokens": 0}
    for domain, row in by_domain.items():
        if not isinstance(domain, str) or not domain or not isinstance(row, dict):
            raise TokenizerEvidenceError(f"{field} domain differs")
        if set(row) != {
            "texts",
            "utf8_bytes",
            "tokens",
            "tokens_per_1k_utf8_bytes",
        }:
            raise TokenizerEvidenceError(f"{field} domain contract differs")
        domain_texts = _nonnegative_int(row["texts"], f"{field} domain texts")
        domain_bytes = _nonnegative_int(row["utf8_bytes"], f"{field} domain bytes")
        domain_tokens = _nonnegative_int(row["tokens"], f"{field} domain tokens")
        if domain_texts <= 0 or domain_bytes <= 0 or domain_tokens <= 0:
            raise TokenizerEvidenceError(f"{field} domain is empty")
        domain_fertility = _positive_number(
            row["tokens_per_1k_utf8_bytes"], f"{field} domain fertility"
        )
        if not math.isclose(
            domain_fertility, 1_000.0 * domain_tokens / domain_bytes, rel_tol=1e-12
        ):
            raise TokenizerEvidenceError(f"{field} domain arithmetic differs")
        domain_totals["texts"] += domain_texts
        domain_totals["utf8_bytes"] += domain_bytes
        domain_totals["tokens"] += domain_tokens
    if domain_totals != {
        "texts": texts,
        "utf8_bytes": utf8_bytes,
        "tokens": tokens,
    }:
        raise TokenizerEvidenceError(f"{field} domain totals differ")
    return value


def _validate_report(report: object) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise TokenizerEvidenceError("tournament report differs")
    if (
        report.get("schema") != SCHEMA
        or report.get("status") != "all_candidates_qualified"
        or report.get("training_authorized") is not False
        or report.get("candidate_build_authorized") is not False
        or report.get("report_sha256")
        != canonical_sha256(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
    ):
        raise TokenizerEvidenceError("qualified tournament report is required")
    _sha256(report.get("corpus_identity_sha256"), "corpus identity")
    candidates = report.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != set(CANDIDATE_SIZES):
        raise TokenizerEvidenceError("candidate set differs")
    shared_corpus_shape: dict[str, tuple[int, int]] | None = None
    special_contract: str | None = None
    for name, expected_size in CANDIDATE_SIZES.items():
        candidate = candidates[name]
        if (
            not isinstance(candidate, dict)
            or candidate.get("name") != name
            or candidate.get("vocab_size") != expected_size
            or candidate.get("byte_fallback") is not True
            or candidate.get("special_tokens_preserved") is not True
            or candidate.get("qualified") is not True
        ):
            raise TokenizerEvidenceError(f"{name} qualification differs")
        _sha256(candidate.get("tokenizer_identity_sha256"), f"{name} identity")
        current_special = _sha256(
            candidate.get("special_token_contract_sha256"),
            f"{name} special-token contract",
        )
        special_contract = special_contract or current_special
        if current_special != special_contract:
            raise TokenizerEvidenceError("special-token contracts differ")
        corpus = _metric_block(candidate.get("corpus"), f"{name} corpus")
        protected = _metric_block(
            candidate.get("protected_suite"), f"{name} protected suite"
        )
        if set(protected["by_domain"]) != PROTECTED_CATEGORIES:
            raise TokenizerEvidenceError("protected categories differ")
        corpus_shape = {
            domain: (row["texts"], row["utf8_bytes"])
            for domain, row in corpus["by_domain"].items()
        }
        shared_corpus_shape = shared_corpus_shape or corpus_shape
        if corpus_shape != shared_corpus_shape:
            raise TokenizerEvidenceError("candidate corpus populations differ")
    return report


def audit_report(
    report: dict[str, Any],
    *,
    report_file_sha256: str,
    model_widths: tuple[int, ...] = DEFAULT_WIDTHS,
) -> dict[str, Any]:
    """Produce a truthful evidence boundary without selecting a tokenizer."""

    report = _validate_report(report)
    report_file_sha256 = _sha256(report_file_sha256, "report file")
    if (
        not model_widths
        or len(set(model_widths)) != len(model_widths)
        or any(
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
            for width in model_widths
        )
    ):
        raise TokenizerEvidenceError("model widths differ")
    candidates = report["candidates"]
    measured_domains = tuple(sorted(candidates["48k"]["corpus"]["by_domain"]))
    complete_domains = set(measured_domains) == set(REQUIRED_DOMAINS)
    metrics = {}
    for name in sorted(CANDIDATE_SIZES, key=CANDIDATE_SIZES.get):
        candidate = candidates[name]
        metrics[name] = {
            "vocab_size": candidate["vocab_size"],
            "tokenizer_identity_sha256": candidate["tokenizer_identity_sha256"],
            "corpus_tokens_per_1k_utf8_bytes": candidate["corpus"][
                "tokens_per_1k_utf8_bytes"
            ],
            "corpus_utf8_bytes_per_token": candidate["corpus"]["utf8_bytes_per_token"],
            "protected_tokens_per_1k_utf8_bytes": candidate["protected_suite"][
                "tokens_per_1k_utf8_bytes"
            ],
            "protected_tokens_per_1k_utf8_bytes_by_category": {
                category: row["tokens_per_1k_utf8_bytes"]
                for category, row in sorted(
                    candidate["protected_suite"]["by_domain"].items()
                )
            },
            "tied_embedding_parameters": {
                str(width): candidate["vocab_size"] * width for width in model_widths
            },
        }
    comparisons = []
    for smaller, larger in (("32k", "48k"), ("48k", "64k"), ("32k", "64k")):
        small = metrics[smaller]
        large = metrics[larger]
        comparisons.append(
            {
                "smaller_candidate": smaller,
                "larger_candidate": larger,
                "smaller_vocab_parameters_saved": {
                    str(width): (large["vocab_size"] - small["vocab_size"]) * width
                    for width in model_widths
                },
                "smaller_corpus_token_increase_percent": 100.0
                * (
                    small["corpus_tokens_per_1k_utf8_bytes"]
                    / large["corpus_tokens_per_1k_utf8_bytes"]
                    - 1.0
                ),
                "smaller_protected_token_increase_percent": 100.0
                * (
                    small["protected_tokens_per_1k_utf8_bytes"]
                    / large["protected_tokens_per_1k_utf8_bytes"]
                    - 1.0
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "status": (
            "mechanically_qualified_capability_selection_pending"
            if complete_domains
            else "mechanically_qualified_domain_coverage_incomplete_capability_pending"
        ),
        "training_authorized": False,
        "production_tokenizer_selected": False,
        "empirical_winner": None,
        "fixed_geometry_default": "48k",
        "fixed_geometry_default_is_empirical_winner": False,
        "tournament_report": {
            "file_sha256": report_file_sha256,
            "report_sha256": report["report_sha256"],
            "corpus_identity_sha256": report["corpus_identity_sha256"],
        },
        "required_domains": list(REQUIRED_DOMAINS),
        "measured_domains": list(measured_domains),
        "all_required_domains_measured": complete_domains,
        "candidate_metrics": metrics,
        "compression_parameter_tradeoffs": comparisons,
        "pareto_conclusion": (
            "all_candidates_remain_nondominated_without_capability_evidence"
        ),
        "selection_requirements": {
            "broad_domain_tournament_population": True,
            "same_tokenizer_training_records": True,
            "same_initialization_and_model_body": True,
            "iso_utf8_byte_training_contrast": True,
            "iso_flop_training_contrast": True,
            "source_disjoint_heldout_likelihood": True,
            "source_disjoint_real_benchmarks": True,
            "paired_retention_and_capability_evidence": True,
            "numeric_pretokenization_ablation": True,
        },
        "checks": {
            "all_candidates_mechanically_qualified": True,
            "losslessness_is_not_capability": True,
            "fertility_is_not_capability": True,
            "fixed_default_is_not_selection": True,
            "no_architecture_or_training_authorization": True,
        },
    }
    payload["audit_sha256"] = canonical_sha256(payload)
    return payload


def audit_file(
    report_path: Path,
    output_path: Path,
    *,
    model_widths: tuple[int, ...] = DEFAULT_WIDTHS,
) -> dict[str, Any]:
    if (
        not report_path.is_file()
        or report_path.is_symlink()
        or report_path.stat().st_nlink != 1
    ):
        raise TokenizerEvidenceError("tournament report is missing or unsafe")
    if output_path.exists() or output_path.is_symlink():
        raise TokenizerEvidenceError("audit output already exists")
    encoded = report_path.read_bytes()
    try:
        report = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenizerEvidenceError("tournament report is malformed") from error
    report_hash = sha256_file(report_path)
    if report_hash != hashlib.sha256(encoded).hexdigest():
        raise TokenizerEvidenceError("tournament report changed while reading")
    payload = audit_report(
        report, report_file_sha256=report_hash, model_widths=model_widths
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tournament-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-width", type=int, action="append")
    args = parser.parse_args()
    widths = tuple(args.model_width) if args.model_width else DEFAULT_WIDTHS
    payload = audit_file(args.tournament_report, args.output, model_widths=widths)
    print(
        json.dumps(
            {"audit_sha256": payload["audit_sha256"], "status": payload["status"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
