from __future__ import annotations

import json

from sai.data.grounded_bridge_verifier_labeling import normalize_model_judgment
from sai.data.nous_grounded_bridge_verifier import RECEIPT_SCHEMA, execute_one
from tests.test_grounded_bridge_verifier_labeling import _candidate, _retain


def test_execute_one_seals_same_family_bridge_verification() -> None:
    candidate = _candidate()
    payload = _retain(candidate)

    def request_function(**_kwargs):
        return (
            {
                "id": "bridge-verification-1",
                "model": "stealth/ox-alpha",
                "provider": "nous",
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
        model="stealth/ox-alpha",
        base_url="http://127.0.0.1:8645/v1",
        api_key="loopback-only",
        timeout_seconds=10,
        maximum_attempts=1,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )
    assert result["schema"] == RECEIPT_SCHEMA
    assert result["judgment"] == normalize_model_judgment(payload, candidate)
    assert result["judgment"]["independent_request_verification_complete"] is True
    assert result["judgment"]["bridge_verified"] is False
    assert result["training_ready"] is False
