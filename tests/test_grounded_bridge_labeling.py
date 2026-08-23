from copy import deepcopy

import pytest

from sai.data.grounded_bridge_labeling import (
    REPRESENTATION_TYPES,
    GroundedBridgeLabelingError,
    normalize_candidate,
    normalize_model_judgment,
)
from sai.data.grounded_bridge_population import build_pair_plan
from tests.test_grounded_bridge_population import anchor


def paired_candidate() -> dict:
    anchors = [
        anchor(1, "mathematics", "mathematics::computer_science"),
        anchor(2, "computer_science", "computer_science::mathematics"),
        anchor(3, "mathematics", "mathematics::computer_science"),
        anchor(4, "computer_science", "computer_science::mathematics"),
    ]
    for row in anchors:
        row["judgment"]["evidence_quotes"] = [row["candidate"]["text"]]
    return build_pair_plan(anchors, target_pairs=1, seed=9)[0]


def model_output(candidate: dict) -> dict:
    return {
        "bridge_label": candidate["bridge_label"],
        "bridge_thesis": "Both anchors expose a transferable structural relationship.",
        "shared_structure": (
            "The two sources describe structures that can be compared at the level "
            "of representation and transformation while keeping their limits explicit."
        ),
        "claims": [
            {
                "claim": "The first anchor supplies a mathematical concept.",
                "anchor_side": "A",
                "evidence_quote": candidate["anchor_a"]["compiler"]["evidence_quotes"][
                    0
                ],
            },
            {
                "claim": "The first anchor provides a second grounded observation.",
                "anchor_side": "A",
                "evidence_quote": candidate["anchor_a"]["compiler"]["evidence_quotes"][
                    0
                ],
            },
            {
                "claim": "The second anchor supplies a computing concept.",
                "anchor_side": "B",
                "evidence_quote": candidate["anchor_b"]["compiler"]["evidence_quotes"][
                    0
                ],
            },
            {
                "claim": "The second anchor provides a second grounded observation.",
                "anchor_side": "B",
                "evidence_quote": candidate["anchor_b"]["compiler"]["evidence_quotes"][
                    0
                ],
            },
        ],
        "representations": [
            {
                "type": kind,
                "title": f"Grounded {kind}",
                "text": (
                    "This representation connects both source anchors through a "
                    "specific shared structure, states the transfer carefully, and "
                    "makes clear that the analogy does not license unsupported claims."
                ),
            }
            for kind in REPRESENTATION_TYPES
        ],
        "prerequisite_map": ["basic structure", "domain-specific representation"],
        "analogy_failure_modes": [
            "The domains use different empirical assumptions.",
            "A structural analogy does not prove causal equivalence.",
        ],
        "verification_questions": [
            {
                "question": "What fact is supported by the first anchor?",
                "expected_answer": "The fact quoted from the first anchor.",
                "anchor_side": "A",
            },
            {
                "question": "What fact is supported by the second anchor?",
                "expected_answer": "The fact quoted from the second anchor.",
                "anchor_side": "B",
            },
        ],
        "confidence_ppm": 900_000,
    }


def test_normalizes_grounded_pair_and_keeps_it_unverified() -> None:
    candidate = paired_candidate()
    assert normalize_candidate(candidate) == candidate
    result = normalize_model_judgment(model_output(candidate), candidate)
    assert result["grounded_synthesis_verified"] is False
    assert result["benchmark_decontamination_complete"] is False
    assert result["training_ready"] is False


def test_rejects_quote_from_wrong_or_absent_anchor() -> None:
    candidate = paired_candidate()
    output = deepcopy(model_output(candidate))
    output["claims"][0]["evidence_quote"] = "invented evidence"
    with pytest.raises(GroundedBridgeLabelingError, match="quote"):
        normalize_model_judgment(output, candidate)
