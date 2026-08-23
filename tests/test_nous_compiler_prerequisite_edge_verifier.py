from __future__ import annotations

import json

from sai.data.nous_compiler_prerequisite_edge_verifier import (
    RECEIPT_SCHEMA,
    execute_one,
)
from tests.test_compiler_prerequisite_edge_labeling import _candidate, _strict


def test_execute_one_seals_same_family_prerequisite_verification() -> None:
    candidate = _candidate()
    payload = _strict(candidate)

    def request_function(**_kwargs):
        return (
            {
                "id": "prerequisite-edge-1",
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

    receipt = execute_one(
        candidate,
        model="stealth/ox-alpha",
        base_url="http://127.0.0.1:8645/v1",
        api_key="loopback-only",
        timeout_seconds=10,
        maximum_attempts=1,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["judgment"]["verdict"] == "strict_prerequisite"
    assert receipt["judgment"]["independent_request_verification_complete"] is True
    assert receipt["judgment"]["directional_prerequisite_verified"] is False
    assert receipt["training_ready"] is False
