import json
from copy import deepcopy

import pytest

from sai.data.agent_labeling import CANDIDATE_SCHEMA
from sai.data.source_quality_gate import (
    SourceQualityGateError,
    build,
    mechanical_quality_evidence,
    validate,
)
from sai.data.token_stream import canonical_sha256


def candidate(text: str, ordinal: int) -> dict:
    import hashlib

    source = {
        "dataset": "unit/source",
        "revision": "immutable-revision",
        "row_id": str(ordinal),
        "license": "test-only",
        "source_type": "reference",
    }
    row = {
        "schema": CANDIDATE_SCHEMA,
        "text": text,
        "source": source,
        "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "provenance_sha256": hashlib.sha256(
            f"provenance-{ordinal}".encode()
        ).hexdigest(),
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def test_rejects_contextless_answer_key_but_not_worked_questions() -> None:
    answer_key = "Answer Key\n" + "\n".join(f"{index}. A" for index in range(1, 10))
    worked = "\n".join(
        f"Question {index}? Answer A. Explanation: the governing principle applies."
        for index in range(1, 10)
    )
    assert mechanical_quality_evidence(answer_key)["decision"] == "hard_reject"
    assert mechanical_quality_evidence(worked)["decision"] == "pass_mechanical_gate"


def test_rejects_scored_answer_sheet_without_problem_statements() -> None:
    sheet = "Cambridge Physics\n" + "\n".join(
        f"{index} a\nUse conservation of energy.\n[1]" for index in range(1, 12)
    )
    result = mechanical_quality_evidence(sheet)
    assert result["decision"] == "hard_reject"
    assert "contextless_scored_answer_sheet" in result["reasons"]

    worksheet = "\n".join(
        f"{index}. What principle applies?\nUse conservation of energy.\n[1]"
        for index in range(1, 12)
    )
    assert mechanical_quality_evidence(worksheet)["decision"] == "pass_mechanical_gate"


def test_does_not_mistake_citations_or_array_indexes_for_score_markers() -> None:
    paper = (
        "A technical paper derives the result and grounds every claim in context.\n"
        + "\n".join(
            f"{index} Section {index} analyzes x[{index}] using citation [{index}]."
            for index in range(1, 20)
        )
        + "\nReferences include [1] Smith and [2] Jones in ordinary prose."
    )
    result = mechanical_quality_evidence(paper)
    assert result["measurements"]["score_marker_count"] == 0
    assert result["decision"] == "pass_mechanical_gate"


@pytest.mark.parametrize(
    ("text", "reason", "decision"),
    [
        (
            "Useful heading and prose. " * 20 + "\ufffd" * 10,
            "unicode_replacement_corruption",
            "hard_reject",
        ),
        (
            "Useful heading and prose. " * 20 + "\x00" * 10,
            "control_character_corruption",
            "hard_reject",
        ),
        (
            "Context begins here. " * 15 + "Z" * 80,
            "repeated_alphanumeric_gibberish",
            "hard_reject",
        ),
        (
            "\n".join(f"https://example.org/{index}" for index in range(10)),
            "url_only_link_index",
            "context_review",
        ),
        (
            "".join("<div></div>" for _ in range(20)),
            "markup_only_fragment",
            "context_review",
        ),
        (
            ("NAVIGATION ITEM\n" * 10) + "brief footer",
            "duplicated_boilerplate",
            "cleanup_review",
        ),
        (
            ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 3),
            "placeholder_lorem_ipsum",
            "hard_reject",
        ),
        (
            "Home\nMenu\nSearch\nSign in\nSign up\nAbout us\nContact us\n"
            "Privacy policy\nTerms of service\nSubscribe",
            "web_navigation_shell",
            "context_review",
        ),
        (
            "Access denied\nRequest blocked\nPlease verify you are human.",
            "access_or_error_placeholder",
            "context_review",
        ),
    ],
)
def test_routes_high_confidence_junk(text: str, reason: str, decision: str) -> None:
    result = mechanical_quality_evidence(text)
    assert result["decision"] == decision
    assert reason in result["reasons"]
    assert result["training_ready"] is False


def test_preserves_prose_code_math_and_structured_context() -> None:
    prose = (
        "A field guide explains how pressure, temperature, and humidity interact "
        "in atmospheric systems. It derives the relationship and then gives a "
        "worked example so each variable has explicit context. "
    ) * 3
    code = "\n".join(f"value_{index} = solve(source_{index})" for index in range(20))
    table = "\n".join(
        ["Measured tensile strength in megapascals by alloy and temperature."]
        + [
            f"alloy {chr(65 + index)}, {20 + index} C, {410 + index} MPa"
            for index in range(20)
        ]
    )
    web_security = (
        "A web-security guide explains why an access denied response may be "
        "triggered, how a request blocked rule is audited, and why engineers "
        "must not ask users to disable safeguards. It compares 403 and 404 "
        "responses with full examples and diagnostic context. "
    ) * 4
    design_history = (
        "Designers sometimes use the phrase lorem ipsum while documenting the "
        "history of typesetting, but this essay develops a real argument about "
        "layout, legibility, and publishing practice. "
    ) * 3
    long_navigation_discussion = (
        "This long design essay analyzes menu, search, sign in, sign up, about us, "
        "contact us, privacy policy, terms of service, and subscribe interfaces as "
        "historical artifacts with evidence and sustained contextual reasoning. "
    ) * 100
    for text in (
        prose,
        code,
        table,
        web_security,
        design_history,
        long_navigation_discussion,
    ):
        assert mechanical_quality_evidence(text)["decision"] == "pass_mechanical_gate"


def test_routes_contextless_catalog_form_without_rejecting_real_questions() -> None:
    catalog_form = """Type of book:
Type of book
Full title of book:
Die Psalmen. Uebersetzt von Moses Mendelssohn
Text is presented as a translation?
Name of original text:
Textual and cultural sources for the book
Are there sources mentioned in the book itself?
List of sources:"""
    result = mechanical_quality_evidence(catalog_form)
    assert result["decision"] == "context_review"
    assert "contextless_metadata_form" in result["reasons"]
    assert result["measurements"]["metadata_field_line_count"] == 7

    questions = "\n".join(
        f"What physical principle explains observation {index}?" for index in range(12)
    )
    assert mechanical_quality_evidence(questions)["decision"] == (
        "pass_mechanical_gate"
    )

    bibliographic_essay = (
        "Title: Geometry across cultures\n"
        + (
            "The essay explains how historians compare primary evidence, preserve "
            "cultural context, and distinguish documented influence from analogy. "
        )
        * 8
    )
    assert mechanical_quality_evidence(bibliographic_essay)["decision"] == (
        "pass_mechanical_gate"
    )


def test_build_replay_and_tamper_detection(tmp_path) -> None:
    good = (
        "This source provides a coherent explanation with definitions, evidence, "
        "and enough surrounding context to support later semantic review. "
    ) * 3
    bad = "Answer Key\n" + "\n".join(f"{index}. B" for index in range(1, 50))
    source = tmp_path / "candidates.jsonl"
    source.write_text(
        "".join(
            json.dumps(candidate(text, index), sort_keys=True) + "\n"
            for index, text in enumerate((good, bad))
        )
    )
    decisions = tmp_path / "decisions.jsonl"
    receipt = tmp_path / "receipt.json"
    result = build(source, decisions, receipt)
    assert result["decision_counts"] == {"hard_reject": 1, "pass_mechanical_gate": 1}
    assert result["training_ready"] is False
    assert validate(receipt) == result

    rows = decisions.read_text().splitlines()
    tampered = json.loads(rows[0])
    tampered["decision"] = "hard_reject"
    rows[0] = json.dumps(tampered, sort_keys=True, separators=(",", ":"))
    decisions.write_text("\n".join(rows) + "\n")
    with pytest.raises(SourceQualityGateError, match="decisions"):
        validate(receipt)


def test_rejects_duplicate_candidate_identity(tmp_path) -> None:
    row = candidate("Coherent source material. " * 20, 1)
    source = tmp_path / "candidates.jsonl"
    source.write_text(json.dumps(row) + "\n" + json.dumps(deepcopy(row)) + "\n")
    with pytest.raises(SourceQualityGateError, match="population"):
        build(source, tmp_path / "decisions.jsonl", tmp_path / "receipt.json")


def test_accepts_provenance_bound_custom_excerpt_schema(tmp_path) -> None:
    import hashlib

    text = "A carefully extracted historical source with coherent context. " * 8
    row = {
        "schema": "sai-custom-source-candidate-v1",
        "candidate_identity_sha256": "a" * 64,
        "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source": {"dataset": "unit/custom"},
        "text_excerpt": text,
        "custom_metadata": {"edition": "first"},
    }
    source = tmp_path / "custom.jsonl"
    source.write_text(json.dumps(row) + "\n")
    result = build(source, tmp_path / "decisions.jsonl", tmp_path / "receipt.json")
    assert result["decision_counts"] == {"pass_mechanical_gate": 1}
