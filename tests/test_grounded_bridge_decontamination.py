from __future__ import annotations

from copy import deepcopy

from sai.data.decontamination import _WORD, POLICY, _normalize
from sai.data.grounded_bridge_decontamination import (
    CLEAN_SCHEMA,
    _bridge_text,
    promote_clean,
    screen_bridge,
)
from sai.data.grounded_bridge_verification_aggregate import _evidence_hashes
from sai.data.nemotron_grounded_bridge_verification_aggregate import (
    RETAINED_SCHEMA,
)
from sai.data.token_stream import canonical_sha256


def _candidate() -> dict:
    row = {
        "schema": RETAINED_SCHEMA,
        "verification_candidate_identity_sha256": "1" * 64,
        "pair_identity_sha256": "2" * 64,
        "generated_candidate_identity_sha256": "3" * 64,
        "anchor_a_candidate_identity_sha256": "4" * 64,
        "anchor_a_source_content_sha256": "5" * 64,
        "anchor_b_candidate_identity_sha256": "6" * 64,
        "anchor_b_source_content_sha256": "7" * 64,
        "generator_receipt_sha256": "8" * 64,
        "generator_judgment_sha256": "9" * 64,
        "verification_receipt_sha256": "a" * 64,
        "verification_judgment_sha256": "b" * 64,
        "bridge_label": "music_x_fourier_analysis",
        "verification_confidence_ppm": 950_000,
        "source_disjoint": True,
        "source_text_persisted": False,
        "same_model_family_verification_complete": True,
        "independent_model_family_verification_complete": True,
        "benchmark_decontamination_complete": False,
        "global_deduplication_complete": False,
        "transfer_ablation_complete": False,
        "bridge_verified": False,
        "training_ready": False,
        "bridge_thesis": "Harmonic structure links musical timbre to spectra.",
        "shared_structure": "Both decompose complex signals into components.",
        "claims": [{"anchor_side": "A", "claim": "A source-grounded claim"}],
        "representations": [
            {
                "type": "worked_transfer",
                "text": "Compare a chord waveform with its frequency peaks.",
            }
        ],
        "prerequisite_map": ["waves", "trigonometry"],
        "analogy_failure_modes": ["A spectrum is not a musical interpretation."],
        "verification_questions": ["Which peaks correspond to harmonics?"],
        "same_family_retention_passed": True,
        "independent_family_retention_passed": True,
        **_evidence_hashes(
            {
                "claim_checks": [],
                "anchor_a_evidence_quotes": ["anchor a"],
                "anchor_b_evidence_quotes": ["anchor b"],
                "generated_evidence_quotes": ["generated"],
            }
        ),
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def test_clean_bridge_advances_only_contamination_state() -> None:
    candidate = _candidate()
    decision = screen_bridge(candidate, set(), set())
    assert decision["contaminated"] is False
    assert decision["bridge_text_persisted"] is False
    clean = promote_clean(candidate)
    assert clean["schema"] == CLEAN_SCHEMA
    assert clean["benchmark_decontamination_complete"] is True
    assert clean["pre_decontamination_record_sha256"] == candidate["record_sha256"]
    assert clean["independent_model_family_verification_complete"] is True
    assert clean["global_deduplication_complete"] is False
    assert clean["transfer_ablation_complete"] is False
    assert clean["bridge_verified"] is False
    assert clean["training_ready"] is False


def test_exact_word_shingle_marks_bridge_contaminated() -> None:
    candidate = _candidate()
    tokens = _WORD.findall(_normalize(_bridge_text(candidate)))
    digest = bytes.fromhex(canonical_sha256(tokens[: POLICY["word_shingle_tokens"]]))
    decision = screen_bridge(candidate, {digest}, set())
    assert decision["word_overlap_count"] == 1
    assert decision["contaminated"] is True


def test_bridge_text_changes_when_any_generated_field_changes() -> None:
    candidate = _candidate()
    changed = deepcopy(candidate)
    changed["verification_questions"] = ["A different question"]
    assert _bridge_text(candidate) != _bridge_text(changed)
