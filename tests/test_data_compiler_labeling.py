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
    RETRY_TIMING_POLICY,
    _retry_delay_seconds,
    execute_one,
    run_shard,
    run_shard_locked,
)
from sai.data.nous_label_worker import (
    NousLabelWorkerError,
    _parse_sse_chat_completion,
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


def test_compiler_recovers_pdf_default_ignorables_as_literal_source_bytes() -> None:
    candidate = _candidate()
    candidate["text"] = candidate["text"].replace(
        "irrigation records",
        "irri\u00adgation\u200b records",
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
    raw["evidence_quotes"] = ["A historian compares irrigation records,"]
    repaired, repairs = repair_evidence_quotes(raw, candidate)
    assert repaired["evidence_quotes"] == [
        "A historian compares irri\u00adgation\u200b records,"
    ]
    assert "pdf-controls" in repairs[0]["algorithm"]
    assert (
        normalize_model_judgment(raw, candidate)["evidence_quotes"]
        == repaired["evidence_quotes"]
    )


def test_compiler_recovers_html_character_reference_as_literal_source_bytes() -> None:
    candidate = _candidate()
    candidate["text"] = candidate["text"].replace(
        "A historian compares irrigation records",
        "A historian compares irrigation &amp; rainfall records",
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
    raw["evidence_quotes"] = ["A historian compares irrigation & rainfall records,"]
    repaired, repairs = repair_evidence_quotes(raw, candidate)
    assert repaired["evidence_quotes"] == [
        "A historian compares irrigation &amp; rainfall records,"
    ]
    assert len(repairs) == 1
    assert "html-character-references" in repairs[0]["algorithm"]
    assert (
        normalize_model_judgment(raw, candidate)["evidence_quotes"]
        == repaired["evidence_quotes"]
    )


def test_compiler_rejects_ambiguous_html_character_reference_recovery() -> None:
    candidate = _candidate()
    candidate["text"] = (
        "Alpha &amp; beta. "
        + "This independently grounded archival explanation supplies enough context "
        * 4
        + "Alpha & beta."
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
    raw["evidence_quotes"] = ["alpha & beta."]
    with pytest.raises(DataCompilerLabelingError, match="evidence"):
        normalize_model_judgment(raw, candidate)


def test_compiler_drops_unsupported_quote_when_exact_evidence_survives() -> None:
    candidate = _candidate()
    raw = _judgment(candidate)
    exact = raw["evidence_quotes"][0]
    raw["evidence_quotes"] = [
        "A model-cleaned quotation that is absent from the source.",
        exact,
    ]
    repaired, repairs = repair_evidence_quotes(raw, candidate)
    assert repaired["evidence_quotes"] == [exact]
    assert repairs == [
        {
            "algorithm": (
                "drop-unrecoverable-quote-when-exact-evidence-survives-v1"
            ),
            "action": "dropped_unrecoverable_model_quote",
            "evidence_index": 0,
            "model_quote_utf8_sha256": hashlib.sha256(
                raw["evidence_quotes"][0].encode()
            ).hexdigest(),
        }
    ]
    assert normalize_model_judgment(raw, candidate)["evidence_quotes"] == [exact]


def test_compiler_rejects_when_no_exact_evidence_quote_survives() -> None:
    candidate = _candidate()
    raw = _judgment(candidate)
    raw["evidence_quotes"] = ["An unsupported model-cleaned quotation."]
    with pytest.raises(DataCompilerLabelingError, match="evidence"):
        repair_evidence_quotes(raw, candidate)


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
    summary = run_shard_locked(
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
    assert (output / ".shard_00000.lock").is_file()

    def refuse_duplicate(row, **kwargs):
        raise AssertionError(f"duplicate compiler request for {row}")

    replay = run_shard_locked(
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
        execute_function=refuse_duplicate,
    )
    assert replay == summary


def test_compiler_resume_rejects_missing_receipt(tmp_path: Path) -> None:
    candidate = _candidate()
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(candidate) + "\n")
    output = tmp_path / "outputs"

    def execute_function(row, **kwargs):
        receipt = {
            "schema": "test-compiler-receipt",
            "candidate_identity_sha256": row["candidate_identity_sha256"],
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        return receipt

    run_shard(
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
    (output / f"{candidate['candidate_identity_sha256']}.compiler.json").unlink()
    with pytest.raises(NousLabelWorkerError, match="missing or unsafe"):
        run_shard(
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


def test_compiler_enum_list_repair_error_states_the_hidden_geometry() -> None:
    candidate = _candidate()
    raw = _judgment(candidate)
    raw["recommended_representations"] = [
        "original_english",
        "cleaned_source",
        "concise_reference",
        "prerequisite_map",
        "conceptual_summary",
        "beginner_explanation",
        "undergraduate_explanation",
        "graduate_explanation",
        "faq",
    ]
    with pytest.raises(DataCompilerLabelingError, match=r"use 1\.\.8 unique"):
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
    assert "MUST replace evidence_quotes only" in repair
    assert "Do not shorten, normalize, join, or rewrite" in repair
    assert "book_excerpt" not in repair


def test_compiler_retry_explains_concept_label_geometry() -> None:
    candidate = _candidate()
    calls = []

    def request_function(**kwargs):
        calls.append(kwargs["body"])
        raw = _judgment(candidate)
        if len(calls) == 1:
            raw["concepts_taught"] = ["Irrigation"]
        return {
            "id": f"response-{len(calls)}",
            "model": "stealth/ox-alpha",
            "provider": "test",
            "created": 1,
            "choices": [
                {
                    "message": {"content": json.dumps(raw)},
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
    assert "concepts_taught must be a JSON list" in repair
    assert "20 unique, nonempty, lowercase strings" in repair
    assert "nested objects" in repair


def test_compiler_retry_disambiguates_document_language_from_subject() -> None:
    candidate = _candidate()
    calls = []

    def request_function(**kwargs):
        calls.append(kwargs["body"])
        raw = _judgment(candidate)
        if len(calls) == 1:
            raw["source_language"] = "german"
        return {
            "id": f"response-{len(calls)}",
            "model": "stealth/ox-alpha",
            "provider": "test",
            "created": 1,
            "choices": [
                {
                    "message": {"content": json.dumps(raw)},
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
    assert "predominant language of the actual supplied document" in repair
    assert "not a language, title, author, or work merely discussed" in repair
    assert "include english_translation" in repair
    assert "disconnected catalog form, field list, or metadata record" in repair


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda raw: raw["risks"].pop("weak_source_grounding"),
            "risks must be a JSON object with exactly these keys",
        ),
        (
            lambda raw: raw["recommended_representations"].append("invented_lane"),
            "recommended_representations must be a JSON list of 1..8",
        ),
    ],
)
def test_compiler_retry_explains_exact_object_and_enum_geometry(
    mutation, expected
) -> None:
    candidate = _candidate()
    calls = []

    def request_function(**kwargs):
        calls.append(kwargs["body"])
        raw = _judgment(candidate)
        if len(calls) == 1:
            mutation(raw)
        return {
            "id": f"response-{len(calls)}",
            "model": "stealth/ox-alpha",
            "provider": "test",
            "created": 1,
            "choices": [
                {
                    "message": {"content": json.dumps(raw)},
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
    assert expected in repair
    if "risks" in expected:
        assert "weak_source_grounding" in repair
        assert "license_or_provenance_unclear" in repair
    else:
        assert "original_english" in repair
        assert "cross_domain_problems" in repair


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


def test_streaming_compiler_transport_is_explicit_and_hashed() -> None:
    candidate = _candidate()
    raw = _judgment(candidate)

    def request_function(**kwargs):
        assert kwargs["body"]["stream"] is True
        return {
            "id": "response-1",
            "model": "stealth/ox-alpha",
            "provider": "test",
            "created": 1,
            "choices": [
                {"message": {"content": json.dumps(raw)}, "finish_reason": "stop"}
            ],
            "usage": {},
        }, 200

    receipt = execute_one(
        candidate,
        model="stealth/ox-alpha",
        base_url="http://127.0.0.1:8645/v1",
        api_key="not-persisted",
        timeout_seconds=1.0,
        maximum_attempts=1,
        stream_transport=True,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )
    assert receipt["request_stream_transport"] is True
    assert receipt["credential_transport"] == "hermes_loopback_proxy"
    assert receipt["shared_provider_concurrency_limit"] == 16
    assert receipt["retry_timing_policy"] == RETRY_TIMING_POLICY


def test_transient_http_retry_delay_is_deterministically_staggered() -> None:
    identity = "a" * 64
    first = _retry_delay_seconds(identity, 1, "transient_http_error")
    second = _retry_delay_seconds(identity, 2, "transient_http_error")
    assert 1.0 <= first <= 2.0
    assert second == min(30.0, first * 2)
    assert _retry_delay_seconds(identity, 2, "invalid_model_output") == 2.0
    assert _retry_delay_seconds(identity, 10, "transient_http_error") == 30.0
    assert _retry_delay_seconds(identity, 10_000, "transient_http_error") == 30.0


@pytest.mark.parametrize("identity", ["", "g" * 64, "a" * 63])
def test_retry_delay_rejects_invalid_identity(identity: str) -> None:
    with pytest.raises(NousLabelWorkerError, match="retry timing identity differs"):
        _retry_delay_seconds(identity, 1, "transient_http_error")


def test_sse_chat_completion_reconstruction_is_bounded_and_exact() -> None:
    chunks = [
        {
            "id": "r",
            "model": "m",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": '{"verdict":'},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "r",
            "model": "m",
            "created": 1,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": '"keep"}'},
                    "finish_reason": "stop",
                }
            ],
        },
        {
            "id": "r",
            "model": "m",
            "created": 1,
            "choices": [],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        },
    ]
    lines = [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks]
    lines.append(b"data: [DONE]\n")
    response = _parse_sse_chat_completion(lines)
    assert response["choices"][0]["message"]["content"] == '{"verdict":"keep"}'
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"]["total_tokens"] == 3
    assert response["_sse_done_marker_observed"] is True


def test_sse_clean_eof_requires_terminal_finish_reason() -> None:
    terminal = {
        "id": "r",
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "complete"},
                "finish_reason": "stop",
            }
        ],
    }
    response = _parse_sse_chat_completion([f"data: {json.dumps(terminal)}\n".encode()])
    assert response["choices"][0]["message"]["content"] == "complete"
    assert response["_sse_done_marker_observed"] is False
    terminal["choices"][0]["finish_reason"] = None
    with pytest.raises(NousLabelWorkerError, match="incomplete"):
        _parse_sse_chat_completion([f"data: {json.dumps(terminal)}\n".encode()])


def test_stored_compiler_judgment_replays_exact_evidence() -> None:
    candidate = _candidate()
    judgment = normalize_model_judgment(_judgment(candidate), candidate)
    assert validate_normalized_judgment(judgment, candidate) == judgment
    judgment["evidence_quotes"][0] = "invented evidence"
    with pytest.raises(DataCompilerLabelingError, match="evidence"):
        validate_normalized_judgment(judgment, candidate)
