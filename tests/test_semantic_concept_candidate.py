from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from sai.data.prerequisite import TAXONOMY_SCHEMA, validate_taxonomy_payload
from sai.data.token_stream import ALLOWED_DOMAINS, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_concept_graph_is_balanced_acyclic_and_rehearsed() -> None:
    candidate = json.loads(
        (
            ROOT / "docs" / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json"
        ).read_text()
    )
    assert candidate["schema"] == "sai-semantic-prerequisite-concept-list-v1"
    assert candidate["status"] == "candidate"
    concepts = candidate["concepts"]
    assert len(concepts) == 50
    assert Counter(item["domain"] for item in concepts) == {
        domain: 10 for domain in ALLOWED_DOMAINS
    }
    payload = {
        "schema": TAXONOMY_SCHEMA,
        "status": "prospective",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "minimum_annotation_confidence_ppm": 800_000,
        "annotation_method": {
            "method": "hybrid",
            "annotator_identity_sha256": "1" * 64,
            "policy_sha256": "2" * 64,
            "audit_sample_receipt_sha256": "3" * 64,
        },
        "concepts": concepts,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    assert validate_taxonomy_payload(payload) == payload

    roots = [item for item in concepts if not item["prerequisites"]]
    assert {item["concept_id"] for item in roots} == {
        "english.symbols",
        "math.number",
    }
    assert all(
        all(
            item["minimum_phase_documents"][phase] > 0
            for phase in (
                "grounding",
                "integration",
                "reasoning",
                "specialization",
            )
        )
        for item in roots
    )
