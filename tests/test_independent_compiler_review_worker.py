import pytest

import sai.data.independent_compiler_review_worker as worker
from sai.data.independent_compiler_review_worker import (
    COHERE_OPENAI_BASE_URL,
    GOOGLE_OPENAI_BASE_URL,
    GROQ_OPENAI_BASE_URL,
    RECEIPT_SCHEMA,
    execute_one,
)
from sai.data.nous_label_worker import NousLabelWorkerError


def test_independent_review_has_distinct_custody_and_omits_reasoning_parameter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = {}

    def fake_execute_contract(candidate, **kwargs):
        seen["candidate"] = candidate
        seen.update(kwargs)
        return {"schema": kwargs["receipt_schema"]}

    monkeypatch.setattr(worker, "execute_contract", fake_execute_contract)

    receipt = execute_one(
        {"candidate_identity_sha256": "1" * 64},
        model="gemini-3.5-flash-lite",
        base_url=GOOGLE_OPENAI_BASE_URL,
        api_key="secret",
        timeout_seconds=1,
        maximum_attempts=1,
    )

    assert receipt["schema"] == RECEIPT_SCHEMA
    assert seen["base_url"] == GOOGLE_OPENAI_BASE_URL
    assert seen["model"] == "gemini-3.5-flash-lite"
    assert seen["reasoning_effort"] is None
    assert seen["receipt_schema"] == RECEIPT_SCHEMA


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("stealth/ox-alpha", GOOGLE_OPENAI_BASE_URL),
        ("gemini-3.5-flash-lite", "https://openrouter.ai/api/v1"),
        ("qwen/qwen3.6-27b", COHERE_OPENAI_BASE_URL),
        ("command-a-plus-05-2026", GROQ_OPENAI_BASE_URL),
    ],
)
def test_google_review_rejects_crossed_identity(model: str, base_url: str) -> None:
    with pytest.raises(NousLabelWorkerError, match="identity differs"):
        execute_one(
            {"candidate_identity_sha256": "1" * 64},
            model=model,
            base_url=base_url,
            api_key="secret",
            timeout_seconds=1,
            maximum_attempts=1,
        )
