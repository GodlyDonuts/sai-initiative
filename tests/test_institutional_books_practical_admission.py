from sai.data.institutional_books_practical_admission import (
    ALLOWED_RIGHTS,
    POLICY,
    POLICY_SHA256,
)
from sai.data.token_stream import canonical_sha256


def test_practical_policy_is_english_non_slop_and_private() -> None:
    assert POLICY["language_gen"] == "eng"
    assert POLICY["minimum_ocr_score_gen"] == 95
    assert POLICY["quality_requirement"] == "pass_mechanical_gate"
    assert POLICY["semantic_model_review_required"] is False
    assert POLICY["redistribution"] == "private_only"
    assert tuple(POLICY["allowed_rights_codes"]) == ALLOWED_RIGHTS
    assert POLICY_SHA256 == canonical_sha256(POLICY)


def test_practical_policy_keeps_evaluation_claims_behind_decontamination() -> None:
    assert POLICY["benchmark_decontamination_blocks_pretraining"] is False
    assert POLICY["benchmark_decontamination_blocks_evaluation_claims"] is True
