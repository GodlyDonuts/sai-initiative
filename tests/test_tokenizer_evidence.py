from __future__ import annotations

import json
from pathlib import Path

import pytest

from sai.data.token_stream import canonical_sha256, sha256_file
from sai.tokenizer.evidence import (
    TokenizerEvidenceError,
    audit_file,
    audit_report,
)


def metric(domains: dict[str, tuple[int, int, int]]) -> dict:
    texts = sum(row[0] for row in domains.values())
    utf8_bytes = sum(row[1] for row in domains.values())
    tokens = sum(row[2] for row in domains.values())
    return {
        "texts": texts,
        "utf8_bytes": utf8_bytes,
        "tokens": tokens,
        "tokens_per_1k_utf8_bytes": 1_000.0 * tokens / utf8_bytes,
        "utf8_bytes_per_token": utf8_bytes / tokens,
        "roundtrip_failures": 0,
        "unknown_tokens": 0,
        "empty_encodings": 0,
        "by_domain": {
            name: {
                "texts": values[0],
                "utf8_bytes": values[1],
                "tokens": values[2],
                "tokens_per_1k_utf8_bytes": 1_000.0 * values[2] / values[1],
            }
            for name, values in domains.items()
        },
    }


def report(*, broad: bool = False) -> dict:
    source = (
        {
            "english": (10, 10_000, 2_200),
            "code": (10, 10_000, 2_000),
            "math": (10, 10_000, 2_100),
            "science": (10, 10_000, 2_050),
            "technical": (10, 10_000, 2_000),
        }
        if broad
        else {"english": (50, 50_000, 10_350)}
    )
    candidate_tokens = {"32k": 11_000, "48k": 10_500, "64k": 10_250}
    candidates = {}
    for index, (name, size) in enumerate(
        (("64k", 64_000), ("48k", 48_000), ("32k", 32_000))
    ):
        scale = candidate_tokens[name] / sum(row[2] for row in source.values())
        domains = {
            domain: (texts, utf8_bytes, round(tokens * scale))
            for domain, (texts, utf8_bytes, tokens) in source.items()
        }
        candidates[name] = {
            "name": name,
            "vocab_size": size,
            "tokenizer_identity_sha256": f"{index + 1:064x}",
            "corpus": metric(domains),
            "protected_suite": metric({"ascii": (5, 1_000, 300 + 20 * index)}),
            "byte_fallback": True,
            "special_tokens_preserved": True,
            "special_token_contract_sha256": "a" * 64,
            "qualified": True,
        }
    payload = {
        "schema": "sai-tokenizer-tournament-report-v1",
        "status": "all_candidates_qualified",
        "training_authorized": False,
        "candidate_build_authorized": False,
        "source_receipts": [],
        "protected_suite_receipt": {},
        "corpus_identity_sha256": "b" * 64,
        "candidates": candidates,
        "checks": {},
    }
    payload["report_sha256"] = canonical_sha256(payload)
    return payload


def test_english_only_mechanics_report_cannot_select_a_winner() -> None:
    payload = audit_report(report(), report_file_sha256="c" * 64)
    assert payload["status"].endswith("domain_coverage_incomplete_capability_pending")
    assert payload["measured_domains"] == ["english"]
    assert not payload["all_required_domains_measured"]
    assert not payload["production_tokenizer_selected"]
    assert payload["empirical_winner"] is None
    assert payload["fixed_geometry_default"] == "48k"
    assert not payload["fixed_geometry_default_is_empirical_winner"]
    assert set(payload["candidate_metrics"]) == {"32k", "48k", "64k"}
    assert payload["candidate_metrics"]["48k"]["tied_embedding_parameters"] == {
        "768": 36_864_000,
        "2560": 122_880_000,
    }


def test_broad_mechanics_evidence_still_requires_model_capability() -> None:
    payload = audit_report(report(broad=True), report_file_sha256="c" * 64)
    assert payload["status"] == "mechanically_qualified_capability_selection_pending"
    assert payload["all_required_domains_measured"]
    assert not payload["production_tokenizer_selected"]
    assert payload["pareto_conclusion"].startswith("all_candidates")


def test_resigned_tamper_and_cross_candidate_population_drift_fail() -> None:
    payload = report()
    payload["candidates"]["48k"]["corpus"]["tokens"] += 1
    payload["report_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )
    with pytest.raises(TokenizerEvidenceError, match="fertility arithmetic"):
        audit_report(payload, report_file_sha256="c" * 64)

    payload = report()
    payload["candidates"]["32k"]["corpus"]["by_domain"]["english"]["utf8_bytes"] += 1
    corpus = payload["candidates"]["32k"]["corpus"]
    corpus["utf8_bytes"] += 1
    corpus["tokens_per_1k_utf8_bytes"] = 1_000 * corpus["tokens"] / corpus["utf8_bytes"]
    corpus["utf8_bytes_per_token"] = corpus["utf8_bytes"] / corpus["tokens"]
    domain = corpus["by_domain"]["english"]
    domain["tokens_per_1k_utf8_bytes"] = 1_000 * domain["tokens"] / domain["utf8_bytes"]
    payload["report_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )
    with pytest.raises(TokenizerEvidenceError, match="populations differ"):
        audit_report(payload, report_file_sha256="c" * 64)


def test_atomic_file_audit_rejects_overwrite(tmp_path: Path) -> None:
    report_path = tmp_path / "tournament.json"
    report_path.write_text(json.dumps(report()) + "\n")
    output = tmp_path / "audit.json"
    payload = audit_file(report_path, output)
    assert payload["tournament_report"]["file_sha256"] == sha256_file(report_path)
    assert json.loads(output.read_text())["audit_sha256"] == payload["audit_sha256"]
    with pytest.raises(TokenizerEvidenceError, match="already exists"):
        audit_file(report_path, output)
