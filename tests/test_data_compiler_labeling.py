from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.data_compiler_labeling import (
    RUBRIC_SHA256,
    DataCompilerLabelingError,
    build_messages,
    evidence_quote_candidates,
    normalize_model_judgment,
    repair_evidence_quotes,
    validate_normalized_judgment,
)
from sai.data.nous_compiler_worker import (
    DEFAULT_COMPILER_CONCURRENCY,
    execute_one,
    run_shard,
)
from sai.data.token_stream import canonical_sha256


def _candidate(*, language: str = "english") -> dict:
    if language == "english":
        text = (
            "A historian compares irrigation records, grain prices, and letters from "
            "farmers to explain how climate, institutions, and household decisions "
            "interacted across several decades. The argument distinguishes direct "
            "archival evidence from later interpretation and preserves the voices of "
            "the people represented in the letters."
        )
    else:
        text = (
            "这份历史材料依据地方档案、私人书信和财政记录，讨论水利工程、粮价与普通家庭决策之间的关系。"
            "作者明确区分原始证据与后来的解释，并保留不同社会群体的观点。"
            "这些材料适合翻译成英文，同时需要保留其中国历史背景，避免把制度和文化概念简单地西方化。"
        )
    row = {
        "schema": "sai-agent-data-candidate-v1",
        "text": text,
        "source": {
            "dataset": "example/world-archive",
            "revision": "v1",
            "row_id": f"history-{language}",
            "license": "CC-BY-4.0",
            "source_type": "reference",
        },
        "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "provenance_sha256": "1" * 64,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def _judgment(candidate: dict, *, language: str = "english") -> dict:
    quote = (
        "A historian compares irrigation records"
        if language == "english"
        else "这份历史材料依据地方档案"
    )
    return {
        "verdict": "retain",
        "epistemic_functions": [
            "reality_anchor",
            "cross_domain_bridge",
            "human_expression",
        ],
        "domains": ["history", "economics_business", "practical_world"],
        "subdomains": ["agricultural history", "institutional history"],
        "difficulty": 2,
        "prerequisite_burden": 1,
        "curriculum_phase": "integration",
        "source_language": language,
        "translation_disposition": (
            "not_needed_english"
            if language == "english"
            else "translate_preserve_meaning"
        ),
        "translation_priority": 0 if language == "english" else 4,
        "preservation_policy": "preserve_plus_derivatives",
        "recommended_representations": (
            ["original_english", "conceptual_summary", "cross_domain_problems"]
            if language == "english"
            else ["english_translation", "conceptual_summary", "cross_domain_problems"]
        ),
        "style": "exposition",
        "likely_origin": "organic_human",
        "grounding_type": "direct_primary",
        "concepts_taught": ["irrigation", "grain prices", "institutions"],
        "prerequisites_assumed": ["historical evidence"],
        "cross_domain_bridges": ["history::economics", "climate::institutions"],
        "scores": {
            "writing_quality": 4,
            "information_density": 4,
            "educational_value": 3,
            "reasoning_density": 3,
            "factual_reference_value": 4,
            "source_reliability": 4,
            "technical_depth": 2,
            "coherence": 4,
            "formatting_quality": 4,
            "human_expression_value": 4,
            "cultural_context_value": 4,
            "cross_domain_bridge_value": 4,
            "novelty_potential": 3,
        },
        "risks": {
            "seo_or_content_farm": False,
            "incoherent_or_corrupted": False,
            "factual_unreliability": False,
            "duplicated_boilerplate": False,
            "answer_farm_without_teaching": False,
            "personal_or_secret_data": False,
            "ocr_or_extraction_damage": False,
            "translation_loss": language != "english",
            "cultural_flattening": language != "english",
            "weak_source_grounding": False,
            "generic_synthetic_style": False,
            "license_or_provenance_unclear": False,
        },
        "confidence_ppm": 950_000,
        "evidence_quotes": [quote],
        "transformation_brief": (
            "Preserve the archival distinctions and cultural context; derive an "
            "English explanation and a climate-economics bridge exercise."
        ),
        "rationale": (
            "The source contributes grounded history and useful cross-domain "
            "relationships."
        ),
    }


def test_compiler_prompt_is_global_source_aware_and_not_scalar() -> None:
    assert DEFAULT_COMPILER_CONCURRENCY == 4
    messages = build_messages(_candidate(language="chinese"))
    assert "English-only does not mean Western-only" in messages[0]["content"]
    envelope = json.loads(messages[1]["content"])
    assert envelope["output_language"] == "english"
    assert envelope["rubric_sha256"] == RUBRIC_SHA256
    assert "information_density" in envelope["output_schema"]["scores"]
    assert "human_expression_value" in envelope["output_schema"]["scores"]
    assert "cross_domain_bridge_value" in envelope["output_schema"]["scores"]
    assert 1 <= len(envelope["evidence_quote_candidates"]) <= 12
    assert all(
        quote in envelope["document"] for quote in envelope["evidence_quote_candidates"]
    )


def test_evidence_quote_candidates_span_multiline_sources_exactly() -> None:
    text = "\n".join(
        f"Section {index}: exact LaTeX \\alpha_{{{index}}} and archival evidence "
        f"remain byte-for-byte stable in this sufficiently long line."
        for index in range(40)
    )
    anchors = evidence_quote_candidates(text)
    assert len(anchors) == 12
    assert anchors == evidence_quote_candidates(text)
    assert all(anchor in text and 24 <= len(anchor) <= 240 for anchor in anchors)


def test_compiler_accepts_translation_and_preservation_plan() -> None:
    candidate = _candidate(language="chinese")
    result = normalize_model_judgment(
        _judgment(candidate, language="chinese"), candidate
    )
    assert result["source_language"] == "chinese"
    assert result["translation_disposition"] == "translate_preserve_meaning"
    assert result["translation_priority"] == 4
    assert "english_translation" in result["recommended_representations"]
    assert result["epistemic_functions"][-1] == "human_expression"


def test_compiler_recovers_one_normalized_quote_as_exact_source_bytes() -> None:
    candidate = _candidate()
    candidate["text"] = candidate["text"].replace(
        "A historian compares irrigation records",
        "A historian\ncompares irrigation records",
    )
    candidate["source_content_sha256"] = hashlib.sha256(
        candidate["text"].encode()
    ).hexdigest()
    candidate["candidate_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in candidate.items()
            if key != "candidate_identity_sha256"
        }
    )
    raw = _judgment(candidate)
    raw["evidence_quotes"] = ["a historian compares irrigation records,"]
    repaired, repairs = repair_evidence_quotes(raw, candidate)
    assert repaired["evidence_quotes"] == ["A historian\ncompares irrigation records,"]
    assert len(repairs) == 1
    assert repairs[0]["source_span_byte_start"] == 0
    assert repairs[0]["source_span_byte_end"] == len(
        repaired["evidence_quotes"][0].encode()
    )
    result = normalize_model_judgment(raw, candidate)
    assert result["evidence_quotes"] == repaired["evidence_quotes"]


def test_compiler_rejects_ambiguous_normalized_quote_recovery() -> None:
    candidate = _candidate()
    candidate["text"] = (
        "Alpha\nbeta. "
        + "This independently grounded archival explanation supplies enough context "
        * 4
        + "Alpha beta."
    )
    candidate["source_content_sha256"] = hashlib.sha256(
        candidate["text"].encode()
    ).hexdigest()
    candidate["candidate_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in candidate.items()
            if key != "candidate_identity_sha256"
        }
    )
    raw = _judgment(candidate)
    raw["evidence_quotes"] = ["alpha beta."]
    with pytest.raises(DataCompilerLabelingError, match="evidence"):
        normalize_model_judgment(raw, candidate)


def test_compiler_leaves_exact_quote_unchanged_without_repair() -> None:
    candidate = _candidate()
    raw = _judgment(candidate)
    repaired, repairs = repair_evidence_quotes(raw, candidate)
    assert repaired["evidence_quotes"] == raw["evidence_quotes"]
    assert repairs == []


def test_compiler_rejects_non_english_without_translation_and_fake_evidence() -> None:
    candidate = _candidate(language="chinese")
    raw = _judgment(candidate, language="chinese")
    raw["translation_disposition"] = "not_needed_english"
    raw["translation_priority"] = 0
    raw["recommended_representations"] = ["conceptual_summary"]
    with pytest.raises(DataCompilerLabelingError, match="translation plan"):
        normalize_model_judgment(raw, candidate)
    raw = _judgment(candidate, language="chinese")
    raw["evidence_quotes"] = ["invented evidence"]
    with pytest.raises(DataCompilerLabelingError, match="evidence"):
        normalize_model_judgment(raw, candidate)


def test_compiler_worker_writes_one_resumable_receipt(tmp_path: Path) -> None:
    candidate = _candidate()
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(candidate) + "\n")

    def execute_function(row, **kwargs):
        receipt = {
            "schema": "test-compiler-receipt",
            "candidate_identity_sha256": row["candidate_identity_sha256"],
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        return receipt

    output = tmp_path / "outputs"
    summary = run_shard(
        candidates,
        output,
        model="stealth/ox-alpha",
        base_url="https://inference-api.nousresearch.com/v1",
        api_key="local-test",
        logical_shards=1,
        shard_index=0,
        concurrency=1,
        timeout_seconds=1,
        maximum_attempts=1,
        execute_function=execute_function,
    )
    assert summary["created_judgments"] == 1
    assert len(list(output.glob("*.compiler.json"))) == 1


def test_compiler_rejects_schema_tamper() -> None:
    candidate = _candidate()
    raw = _judgment(candidate)
    raw["scores"]["information_density"] = 5
    with pytest.raises(Exception, match="information_density"):
        normalize_model_judgment(raw, candidate)
    raw = _judgment(candidate)
    raw["cross_domain_bridges"] = ["history-economics"]
    with pytest.raises(DataCompilerLabelingError, match="bridge format"):
        normalize_model_judgment(raw, candidate)


def test_general_compiler_retry_names_the_actual_document_envelope() -> None:
    candidate = _candidate()
    calls = []

    def request_function(**kwargs):
        calls.append(kwargs["body"])
        content = (
            json.dumps({"wrong": "shape"})
            if len(calls) == 1
            else json.dumps(_judgment(candidate))
        )
        return {
            "id": f"response-{len(calls)}",
            "model": "stealth/ox-alpha",
            "provider": "test",
            "created": 1,
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }, 200

    execute_one(
        candidate,
        model="stealth/ox-alpha",
        base_url="https://inference-api.nousresearch.com/v1",
        api_key="not-persisted",
        timeout_seconds=1.0,
        maximum_attempts=2,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )

    repair = calls[1]["messages"][-1]["content"]
    assert "byte-for-byte quote from document" in repair
    assert "evidence_quote_candidates" in repair
    assert "book_excerpt" not in repair


def test_compiler_receipt_records_hashed_deterministic_quote_repair() -> None:
    candidate = _candidate()
    raw = _judgment(candidate)
    raw["evidence_quotes"] = ["a historian compares irrigation records,"]

    def request_function(**_kwargs):
        return {
            "id": "response-1",
            "model": "stealth/ox-alpha",
            "provider": "test",
            "created": 1,
            "choices": [
                {"message": {"content": json.dumps(raw)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }, 200

    receipt = execute_one(
        candidate,
        model="stealth/ox-alpha",
        base_url="https://inference-api.nousresearch.com/v1",
        api_key="not-persisted",
        timeout_seconds=1.0,
        maximum_attempts=1,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )
    assert len(receipt["deterministic_evidence_quote_repairs"]) == 1
    repair = receipt["deterministic_evidence_quote_repairs"][0]
    assert (
        repair["model_quote_utf8_sha256"]
        == hashlib.sha256(raw["evidence_quotes"][0].encode()).hexdigest()
    )
    assert receipt["judgment"]["evidence_quotes"][0] in candidate["text"]
    assert receipt["raw_model_json_sha256"] == canonical_sha256(raw)


def test_stored_compiler_judgment_replays_exact_evidence() -> None:
    candidate = _candidate()
    judgment = normalize_model_judgment(_judgment(candidate), candidate)
    assert validate_normalized_judgment(judgment, candidate) == judgment
    judgment["evidence_quotes"][0] = "invented evidence"
    with pytest.raises(DataCompilerLabelingError, match="evidence"):
        validate_normalized_judgment(judgment, candidate)
