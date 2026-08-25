from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.nous_grounded_representation_worker import (
    RECEIPT_SCHEMA,
    execute_one,
    load_candidates,
)
from sai.data.nous_label_worker import NousLabelWorkerError
from sai.data.token_stream import canonical_sha256


def _candidate() -> dict:
    quote = "A documented archive relates craft history to material knowledge."
    text = quote + (" It preserves the source's uncertainty and context." * 6)
    row = {
        "schema": "sai-public-domain-review-representation-candidate-v1",
        "text": text,
        "source_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_record_sha256": "1" * 64,
        "original_candidate_identity_sha256": "2" * 64,
        "source": {
            "dataset": "common-pile/public_domain_review_filtered",
            "row_id": "craft-history",
            "source_url": "https://publicdomainreview.org/collection/craft-history/",
            "source_type": "collection",
            "license": "CC-BY-SA-4.0",
            "attribution_required": True,
            "share_alike_required": True,
        },
        "compiler": {
            "candidate_identity_sha256": "3" * 64,
            "receipt_sha256": "4" * 64,
            "judgment_sha256": "5" * 64,
            "work_record_sha256": "6" * 64,
            "content_route": "representation_verification",
            "rights_route": "editorial_scope_review",
            "verdict": "retain",
            "preservation_policy": "preserve_plus_derivatives",
            "requested_representations": ["conceptual_summary"],
            "domains": ["history"],
            "subdomains": ["craft history"],
            "concepts_taught": ["historical craft"],
            "prerequisites_assumed": [],
            "cross_domain_bridges": [],
            "difficulty": 1,
            "curriculum_phase": "breadth",
        },
        "compiler_route_is_verified_admission": False,
        "representation_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def _response() -> dict:
    quote = "A documented archive relates craft history to material knowledge."
    content = {
        "representations": [
            {
                "type": "conceptual_summary",
                "title": "Craft history and material knowledge",
                "text": (
                    "The archive grounds a connection between historical craft "
                    "practice and knowledge about materials while retaining context."
                ),
                "evidence_quotes": [quote],
                "concepts": ["historical craft"],
                "difficulty": 1,
            }
        ],
        "prerequisite_edges": [],
        "cross_domain_bridge_candidates": [],
        "coverage_note": "The representation covers the central documented relation.",
    }
    return {
        "id": "response-1",
        "model": "stealth/ox-alpha",
        "provider": "nous",
        "created": 1,
        "choices": [
            {
                "message": {"content": json.dumps(content)},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


def test_execute_one_seals_generated_representation() -> None:
    def request_function(**_kwargs):
        return _response(), 200

    receipt = execute_one(
        _candidate(),
        model="stealth/ox-alpha",
        base_url="http://127.0.0.1:8645/v1",
        api_key="loopback-only",
        timeout_seconds=10,
        maximum_attempts=1,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["judgment"]["representation_verified"] is False
    assert receipt["judgment"]["share_alike_required"] is True
    assert receipt["raw_source_is_training_data"] is False
    assert receipt["training_ready"] is False


def test_load_candidates_rejects_duplicate_identity(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    line = json.dumps(_candidate(), sort_keys=True)
    path.write_text(line + "\n" + line + "\n")
    with pytest.raises(NousLabelWorkerError, match="duplicated"):
        load_candidates(path)


def test_execute_one_uses_representation_specific_concept_retry_hint() -> None:
    calls = []

    def request_function(**kwargs):
        calls.append(kwargs["body"])
        response = _response()
        if len(calls) == 1:
            payload = json.loads(response["choices"][0]["message"]["content"])
            payload["representations"][0]["concepts"] = ["Historical Craft"]
            response["choices"][0]["message"]["content"] = json.dumps(payload)
        return response, 200

    execute_one(
        _candidate(),
        model="stealth/ox-alpha",
        base_url="http://127.0.0.1:8645/v1",
        api_key="loopback-only",
        timeout_seconds=10,
        maximum_attempts=2,
        request_function=request_function,
        sleep_function=lambda _seconds: None,
    )
    repair = calls[1]["messages"][-1]["content"]
    assert "concepts to a JSON list" in repair
    assert "1..8 unique, nonempty, lowercase strings" in repair
    assert "concepts_taught" not in repair
