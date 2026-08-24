from sai.data.data_compiler_labeling import RISK_KEYS, SCORE_KEYS
from sai.data.pleias_semantic_stratum_decision import decide_strata


def _judgment(*, verdict="retain", risk=None, score=4):
    risks = {key: False for key in RISK_KEYS}
    if risk is not None:
        risks[risk] = True
    return {
        "verdict": verdict,
        "risks": risks,
        "source_language": "english",
        "scores": {key: score for key in SCORE_KEYS},
        "epistemic_functions": ["reality_anchor"],
        "domains": ["physics_astronomy"],
        "difficulty": 3,
        "prerequisite_burden": 2,
        "curriculum_phase": "integration",
        "concepts_taught": ["orbital mechanics"],
        "prerequisites_assumed": ["classical mechanics"],
    }


def _comparison(route="representation_verification", agree=True):
    return {
        "complete_review_coverage": True,
        "all_available_route_agree": agree,
        "reviews": {"nemotron": {"route": route}},
    }


def test_advances_only_strong_cross_family_supported_strata():
    primary = [("science::open::512to4095", _judgment()) for _ in range(8)]
    comparison = [
        ("science::open::512to4095", _comparison()),
        ("science::open::512to4095", _comparison()),
    ]
    result = decide_strata(primary, comparison)
    assert len(result) == 1
    assert result[0]["decision"] == "advance_to_full_candidate_decontamination"
    assert result[0]["reasons"] == []
    assert result[0]["primary"]["difficulty_mean_milli"] == 3_000
    assert result[0]["primary"]["prerequisite_burden_mean_milli"] == 2_000
    assert result[0]["primary"]["dominant_curriculum_phase"] == "integration"
    assert result[0]["primary"]["recurring_concepts"] == [
        {"concept": "orbital mechanics", "votes": 8}
    ]
    assert result[0]["automatic_training_admission"] is False


def test_holds_on_any_primary_blocking_route():
    stratum = "science::open::512to4095"
    primary = [(stratum, _judgment()) for _ in range(7)] + [
        (stratum, _judgment(verdict="reject"))
    ]
    comparison = [(stratum, _comparison()), (stratum, _comparison())]
    result = decide_strata(primary, comparison)[0]
    assert result["decision"] == "hold_semantic_stratum"
    assert "primary_blocking_route_present" in result["reasons"]


def test_holds_without_two_complete_independent_rows():
    stratum = "books::culture::ge32768"
    primary = [(stratum, _judgment()) for _ in range(8)]
    result = decide_strata(primary, [(stratum, _comparison())])[0]
    assert result["decision"] == "hold_semantic_stratum"
    assert "insufficient_independent_rows" in result["reasons"]


def test_holds_cross_family_disagreement_and_independent_block():
    stratum = "code::source::4096to32767"
    primary = [(stratum, _judgment()) for _ in range(8)]
    comparison = [
        (stratum, _comparison("quarantine", agree=False)),
        (stratum, _comparison("representation_verification", agree=True)),
    ]
    result = decide_strata(primary, comparison)[0]
    assert result["decision"] == "hold_semantic_stratum"
    assert "independent_blocking_route_present" in result["reasons"]
    assert "cross_family_route_agreement_below_threshold" in result["reasons"]
