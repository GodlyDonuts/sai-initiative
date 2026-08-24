from sai.data.decontamination import (
    _CODE,
    _WORD,
    POLICY,
    _code_shingles,
    _normalize,
    _shingles,
)
from sai.data.pleias_full_candidate_decontamination import (
    PleiasFullCandidateDecontaminationError,
    _advanced_strata,
    screen_text,
)
from sai.data.token_stream import canonical_sha256


def _decision(stratum, disposition):
    row = {
        "stratum": stratum,
        "decision": disposition,
        "automatic_training_admission": False,
    }
    row["row_sha256"] = canonical_sha256(row)
    return row


def test_replays_only_explicitly_advanced_strata():
    rows = [
        _decision(
            "science::Open Science::512to4095",
            "advance_to_full_candidate_decontamination",
        ),
        _decision("web::Open Web::lt512", "hold_semantic_stratum"),
    ]
    payload = {
        "status": "complete_nontraining_pleias_semantic_stratum_decision",
        "decisions": rows,
        "ordered_decisions_sha256": canonical_sha256(
            [row["row_sha256"] for row in rows]
        ),
        "advanced_strata": ["science::Open Science::512to4095"],
    }
    assert _advanced_strata(payload) == frozenset({"science::Open Science::512to4095"})
    payload["advanced_strata"] = ["web::Open Web::lt512"]
    try:
        _advanced_strata(payload)
    except PleiasFullCandidateDecontaminationError:
        pass
    else:
        raise AssertionError("changed advancement must fail closed")


def test_screens_full_text_on_any_exact_boundary_shingle():
    text = (
        "A benchmark sentence contains thirteen carefully selected words for "
        "an exact contamination boundary check today."
    )
    normalized = _normalize(text)
    word_boundary = _shingles(_WORD.findall(normalized), POLICY["word_shingle_tokens"])
    code_boundary = _code_shingles(_CODE.findall(normalized))
    word_overlap, code_overlap = screen_text(text, word_boundary, set())
    assert word_overlap > 0
    assert code_overlap == 0
    clean_word, matching_code = screen_text(text, set(), code_boundary)
    assert clean_word == 0
    assert matching_code > 0
    assert screen_text(
        "A completely unrelated short document.", word_boundary, code_boundary
    ) == (
        0,
        0,
    )
