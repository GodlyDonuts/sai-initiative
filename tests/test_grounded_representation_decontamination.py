from __future__ import annotations

from sai.data.decontamination import _WORD, POLICY, _normalize
from sai.data.grounded_representation_decontamination import (
    CLEAN_SCHEMA,
    promote_clean,
    screen_representation,
)
from sai.data.token_stream import canonical_sha256


def _candidate() -> dict:
    text = (
        "Historical design connects material constraints, craft practice, public "
        "institutions, engineering choices, and documented cultural change."
    )
    row = {
        "schema": "sai-generated-grounded-representation-candidate-v1",
        "source_candidate_identity_sha256": "1" * 64,
        "source_text_sha256": "2" * 64,
        "source_record_sha256": "3" * 64,
        "compiler_judgment_sha256": "4" * 64,
        "generator_receipt_sha256": "5" * 64,
        "generator_judgment_sha256": "6" * 64,
        "source": {
            "dataset": "common-pile/public_domain_review_filtered",
            "row_id": "design-history",
            "source_url": "https://publicdomainreview.org/collection/design/",
            "license": "CC-BY-SA-4.0",
        },
        "attribution_required": True,
        "share_alike_required": True,
        "representation_index": 0,
        "representation_type": "conceptual_summary",
        "title": "Historical design",
        "text": text,
        "text_sha256": "7" * 64,
        "concepts": ["historical design"],
        "difficulty": 1,
        "evidence_quote_sha256s": ["8" * 64],
        "source_claims_independently_verified": False,
        "benchmark_decontamination_complete": False,
        "global_deduplication_complete": False,
        "representation_verified": False,
        "training_ready": False,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def test_clean_representation_advances_only_contamination_state() -> None:
    candidate = _candidate()
    decision = screen_representation(candidate, set(), set())
    assert decision["contaminated"] is False
    clean = promote_clean(candidate)
    assert clean["schema"] == CLEAN_SCHEMA
    assert clean["benchmark_decontamination_complete"] is True
    assert clean["pre_decontamination_record_sha256"] == candidate["record_sha256"]
    assert clean["global_deduplication_complete"] is False
    assert clean["representation_verified"] is False
    assert clean["training_ready"] is False


def test_exact_word_shingle_marks_generated_text_contaminated() -> None:
    candidate = _candidate()
    tokens = _WORD.findall(_normalize(candidate["text"]))
    digest = bytes.fromhex(canonical_sha256(tokens[: POLICY["word_shingle_tokens"]]))
    decision = screen_representation(candidate, {digest}, set())
    assert decision["word_overlap_count"] == 1
    assert decision["contaminated"] is True
    assert decision["representation_text_persisted"] is False
