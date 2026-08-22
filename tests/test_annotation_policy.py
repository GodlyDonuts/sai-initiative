from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sai.data.annotation_policy import (
    AnnotationPolicyError,
    validate_policy,
    validate_policy_payload,
)
from sai.data.token_stream import canonical_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "docs" / "SAI_SEMANTIC_ANNOTATION_POLICY.json"
CONCEPTS = ROOT / "docs" / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json"


def _resign(payload: dict) -> dict:
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )
    return payload


def test_candidate_policy_is_exact_conservative_and_non_authorizing() -> None:
    payload = validate_policy(
        POLICY, expected_concept_list_sha256=sha256_file(CONCEPTS)
    )
    assert payload["annotation_unit"] == "document_concept_presence"
    assert payload["confidence_contract"]["minimum_confidence_ppm"] == 800_000
    assert (
        payload["prerequisite_contract"]["same_document_exposure_counts_as_prior"]
        is False
    )
    assert payload["review_contract"]["maximum_disagreement_ppm"] == 50_000
    assert payload["training_authorized"] is False
    assert payload["four_b_training_authorized"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(positive_label_rule="term_mention_is_enough"),
        lambda value: value["confidence_contract"].update(
            minimum_confidence_ppm=500_000
        ),
        lambda value: value["prerequisite_contract"].update(
            same_document_exposure_counts_as_prior=True
        ),
        lambda value: value["review_contract"].update(maximum_disagreement_ppm=100_000),
        lambda value: value.update(extra="undeclared"),
    ],
)
def test_rejects_resigned_policy_weakening(mutate) -> None:
    payload = json.loads(POLICY.read_text())
    mutate(payload)
    _resign(payload)
    with pytest.raises(AnnotationPolicyError):
        validate_policy_payload(
            payload, expected_concept_list_sha256=sha256_file(CONCEPTS)
        )


def test_rejects_wrong_concept_identity_and_unsafe_file(tmp_path: Path) -> None:
    with pytest.raises(AnnotationPolicyError, match="boundary differs"):
        validate_policy(POLICY, expected_concept_list_sha256="f" * 64)
    copy = tmp_path / "policy.json"
    copy.write_bytes(POLICY.read_bytes())
    link = tmp_path / "policy-link.json"
    link.symlink_to(copy)
    with pytest.raises(AnnotationPolicyError, match="missing or unsafe"):
        validate_policy(link, expected_concept_list_sha256=sha256_file(CONCEPTS))


def test_policy_self_hash_cannot_be_reused_after_mutation() -> None:
    payload = deepcopy(json.loads(POLICY.read_text()))
    payload["negative_label_rule"] = "infer_when_missing"
    with pytest.raises(AnnotationPolicyError, match="boundary differs"):
        validate_policy_payload(
            payload, expected_concept_list_sha256=sha256_file(CONCEPTS)
        )
