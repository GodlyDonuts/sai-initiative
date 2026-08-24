from __future__ import annotations

import json

from sai.data.nemotron_grounded_bridge_verifier import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    RECEIPT_SCHEMA,
    execute_one,
)
from sai.data.nemotron_grounded_bridge_verifier_labeling import (
    normalize_model_judgment,
)
from tests.test_grounded_bridge_verifier_labeling import _candidate, _retain


def test_execute_one_seals_independent_family_bridge_verification() -> None:
    candidate = _candidate()
    payload = _retain(candidate)
    seen = {}

    def request_function(**kwargs):
        seen.update(kwargs)
        return (
            {
                "id": "independent-bridge-verification-1",
                "model": "nvidia/nemotron-3-ultra-550b-a55b",
                "provider": "nvidia",
                "created": 1,
                "choices": [
                    {
                        "message": {"content": json.dumps(payload)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            },
            200,
        )

    result = execute_one(
        candidate,
        model=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        api_key="nvidia-only",
        timeout_seconds=10,
        maximum_attempts=1,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )
    assert seen["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert seen["body"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert "reasoning" not in seen["body"]
    assert result["schema"] == RECEIPT_SCHEMA
    assert result["requested_model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert result["endpoint_origin"] == "https://integrate.api.nvidia.com/v1"
    assert result["credential_transport"] == "direct_portal_bearer"
    assert result["judgment"] == normalize_model_judgment(payload, candidate)
    assert result["judgment"]["independent_request_verification_complete"] is True
    assert result["judgment"]["independent_model_family_verification_complete"] is True
    assert result["judgment"]["bridge_verified"] is False
    assert result["training_ready"] is False
