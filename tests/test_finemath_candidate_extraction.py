import json

from sai.data.finemath_candidate_extraction import candidate_reasons


def candidate(**overrides):
    row = {
        "url": "https://math.example.org/proof",
        "text": "x" * 512,
        "token_count": 256,
        "metadata": json.dumps({"found_math": True}),
        "int_score": 5,
        "language": "en",
        "language_score": 0.95,
    }
    row.update(overrides)
    return row


def test_conservative_candidate_accepts_only_complete_high_signal_row() -> None:
    assert candidate_reasons(candidate()) == ()


def test_conservative_candidate_reports_all_failures() -> None:
    reasons = candidate_reasons(
        candidate(
            url="not-a-url",
            text="short",
            token_count=32,
            metadata="{}",
            int_score=4,
            language="fr",
            language_score=0.2,
        )
    )
    assert reasons == (
        "language_not_english",
        "language_confidence_below_0p90",
        "upstream_integer_score_below_5",
        "found_math_absent_or_metadata_invalid",
        "token_count_outside_128_to_32767",
        "text_below_512_utf8_bytes",
        "url_host_missing_or_invalid",
    )
