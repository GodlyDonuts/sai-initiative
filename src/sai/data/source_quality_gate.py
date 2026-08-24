"""Fail closed on high-confidence junk before semantic source admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import normalize_candidate
from sai.data.contextless_answer_key_filter import (
    POLICY_SHA256 as ANSWER_KEY_POLICY_SHA256,
)
from sai.data.contextless_answer_key_filter import answer_key_evidence
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-source-mechanical-quality-gate-v1"
DECISION_SCHEMA = "sai-source-mechanical-quality-decision-v1"
POLICY = {
    "contextless_answer_key_policy_sha256": ANSWER_KEY_POLICY_SHA256,
    "hard_reject": {
        "contextless_mcq_answer_key": True,
        "contextless_scored_answer_sheet": {
            "minimum_score_markers": 8,
            "minimum_numbered_or_lettered_parts": 8,
            "maximum_question_marks": 1,
        },
        "control_character_corruption": {
            "minimum_count": 4,
            "minimum_ppm": 1_000,
        },
        "unicode_replacement_corruption": {
            "minimum_count": 4,
            "minimum_ppm": 1_000,
        },
        "repeated_alphanumeric_gibberish": {
            "minimum_run": 64,
        },
        "placeholder_lorem_ipsum": {
            "minimum_full_phrase_occurrences": 2,
            "minimum_short_phrase_occurrences": 4,
        },
    },
    "context_review": {
        "url_only_link_index": {
            "minimum_urls": 8,
            "minimum_url_line_ppm": 750_000,
            "maximum_non_url_words_per_url": 5,
        },
        "markup_only_fragment": {
            "minimum_tags": 16,
            "maximum_visible_words": 12,
        },
        "contextless_structured_fragment": {
            "minimum_utf8_bytes": 256,
            "maximum_alpha_words": 8,
            "minimum_digit_symbol_ppm": 750_000,
        },
        "contextless_metadata_form": {
            "minimum_nonempty_lines": 6,
            "maximum_utf8_bytes": 4_096,
            "minimum_metadata_field_lines": 4,
            "minimum_metadata_field_line_ppm": 400_000,
            "maximum_median_line_characters": 80,
            "maximum_line_alpha_words": 16,
        },
        "web_navigation_shell": {
            "minimum_nonempty_lines": 8,
            "minimum_distinct_shell_markers": 4,
            "minimum_short_line_ppm": 600_000,
            "maximum_alpha_words": 160,
            "maximum_utf8_bytes": 16_384,
        },
        "access_or_error_placeholder": {
            "maximum_utf8_bytes": 4_096,
            "minimum_distinct_markers": 2,
            "maximum_alpha_words": 120,
        },
    },
    "cleanup_review": {
        "duplicated_boilerplate": {
            "minimum_nonempty_lines": 10,
            "minimum_repeated_line_count": 5,
            "minimum_duplicate_character_ppm": 600_000,
        },
    },
    "decision_precedence": [
        "hard_reject",
        "context_review",
        "cleanup_review",
        "pass_mechanical_gate",
    ],
    "semantic_admission_implied": False,
}
POLICY_SHA256 = canonical_sha256(POLICY)

_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_HTML_TAG = re.compile(r"<\s*/?\s*[A-Za-z][^>]{0,240}>")
_ALPHA_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
_REPEATED_ALPHANUMERIC = re.compile(r"([A-Za-z0-9])\1{63,}")
_LOREM_FULL = re.compile(r"\blorem\s+ipsum\s+dolor\s+sit\s+amet\b", re.IGNORECASE)
_LOREM_SHORT = re.compile(r"\blorem\s+ipsum\b", re.IGNORECASE)
_SCORE_MARKER = re.compile(r"(?m)^\s*\[\s*(?:\d{1,2}|[MAB]\d)\s*\]\s*$", re.IGNORECASE)
_NUMBERED_OR_LETTERED_PART = re.compile(
    r"(?m)^\s*(?:\d{1,3}\s*[a-h]?|[a-h])(?:[.)\]:-]|\s|$)", re.IGNORECASE
)
_METADATA_FIELD_LINE = re.compile(
    r"(?i)^\s*(?:(?:full\s+)?title|type(?:\s+of)?|name(?:\s+of)?|author|"
    r"translator|translation\s+date|publication\s+date|original\s+language|"
    r"language|isbn|edition|source(?:s)?|list\s+of|are\s+there|is\s+there|"
    r"text\s+is\s+presented|rights(?:\s+status)?|license|genre|subject|"
    r"keywords?)\b[^\n]{0,160}$"
)
_WEB_SHELL_MARKERS = (
    "about us",
    "accept cookies",
    "contact us",
    "cookie settings",
    "log in",
    "menu",
    "privacy policy",
    "search",
    "sign in",
    "sign up",
    "skip to content",
    "subscribe",
    "terms of service",
)
_ERROR_PLACEHOLDER_MARKERS = (
    "403 forbidden",
    "404 not found",
    "access denied",
    "checking your browser",
    "enable cookies",
    "enable javascript",
    "page not found",
    "request blocked",
    "service unavailable",
    "verify you are human",
)


class SourceQualityGateError(RuntimeError):
    """The source population, decision stream, or replay differs."""


def _ppm(numerator: int, denominator: int) -> int:
    return numerator * 1_000_000 // denominator if denominator else 0


def _longest_alphanumeric_run(text: str) -> int:
    longest = current = 0
    previous = ""
    for character in text:
        if character.isalnum() and character == previous:
            current += 1
        elif character.isalnum():
            current = 1
        else:
            current = 0
        previous = character
        longest = max(longest, current)
    return longest


def mechanical_quality_evidence(text: str) -> dict[str, Any]:
    """Return deterministic evidence and a conservative non-admission route."""

    if not isinstance(text, str) or not text:
        raise SourceQualityGateError("quality-gate text differs")
    codepoints = len(text)
    utf8_bytes = len(text.encode())
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    control_count = sum(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in text
    )
    replacement_count = text.count("\ufffd")
    url_matches = list(_URL.finditer(text))
    url_lines = sum(bool(_URL.search(line)) for line in nonempty_lines)
    without_urls = _URL.sub(" ", text)
    non_url_words = sum(1 for _ in _ALPHA_WORD.finditer(without_urls))
    tag_count = sum(1 for _ in _HTML_TAG.finditer(text))
    visible_words = sum(1 for _ in _ALPHA_WORD.finditer(_HTML_TAG.sub(" ", text)))
    alphanumeric_count = sum(character.isalnum() for character in text)
    whitespace_count = sum(character.isspace() for character in text)
    digit_symbol_count = max(0, codepoints - whitespace_count - alphanumeric_count)
    digit_symbol_count += sum(character.isdigit() for character in text)

    line_counts = Counter(line for line in nonempty_lines if len(line) >= 8)
    repeated_line_characters = sum(
        len(line) * count for line, count in line_counts.items() if count > 1
    )
    total_line_characters = sum(len(line) for line in nonempty_lines)
    maximum_repeated_line_count = max(line_counts.values(), default=0)

    answer_key = answer_key_evidence(text)
    score_marker_count = len(_SCORE_MARKER.findall(text))
    numbered_or_lettered_part_count = len(_NUMBERED_OR_LETTERED_PART.findall(text))
    question_mark_count = text.count("?")
    longest_run = _longest_alphanumeric_run(text)
    control_ppm = _ppm(control_count, codepoints)
    replacement_ppm = _ppm(replacement_count, codepoints)
    url_line_ppm = _ppm(url_lines, len(nonempty_lines))
    duplicate_character_ppm = _ppm(repeated_line_characters, total_line_characters)
    digit_symbol_ppm = _ppm(digit_symbol_count, codepoints)
    metadata_field_line_count = sum(
        bool(_METADATA_FIELD_LINE.fullmatch(line)) for line in nonempty_lines
    )
    metadata_field_line_ppm = _ppm(metadata_field_line_count, len(nonempty_lines))
    line_character_counts = sorted(len(line) for line in nonempty_lines)
    median_line_characters = (
        line_character_counts[(len(line_character_counts) - 1) // 2]
        if line_character_counts
        else 0
    )
    maximum_line_alpha_words = max(
        (len(_ALPHA_WORD.findall(line)) for line in nonempty_lines), default=0
    )
    alpha_word_count = sum(1 for _ in _ALPHA_WORD.finditer(text))
    short_line_ppm = _ppm(
        sum(len(line) <= 40 for line in nonempty_lines), len(nonempty_lines)
    )
    marker_source = text.casefold() if utf8_bytes <= 16_384 else ""
    shell_markers = sorted(
        marker for marker in _WEB_SHELL_MARKERS if marker in marker_source
    )
    error_markers = sorted(
        marker for marker in _ERROR_PLACEHOLDER_MARKERS if marker in marker_source
    )
    lorem_full_count = len(_LOREM_FULL.findall(text))
    lorem_short_count = len(_LOREM_SHORT.findall(text))

    flags = {
        "contextless_mcq_answer_key": answer_key["contextless_answer_key"],
        "contextless_scored_answer_sheet": bool(
            score_marker_count
            >= POLICY["hard_reject"]["contextless_scored_answer_sheet"][
                "minimum_score_markers"
            ]
            and numbered_or_lettered_part_count
            >= POLICY["hard_reject"]["contextless_scored_answer_sheet"][
                "minimum_numbered_or_lettered_parts"
            ]
            and question_mark_count
            <= POLICY["hard_reject"]["contextless_scored_answer_sheet"][
                "maximum_question_marks"
            ]
        ),
        "control_character_corruption": bool(
            control_count
            >= POLICY["hard_reject"]["control_character_corruption"]["minimum_count"]
            and control_ppm
            >= POLICY["hard_reject"]["control_character_corruption"]["minimum_ppm"]
        ),
        "unicode_replacement_corruption": bool(
            replacement_count
            >= POLICY["hard_reject"]["unicode_replacement_corruption"]["minimum_count"]
            and replacement_ppm
            >= POLICY["hard_reject"]["unicode_replacement_corruption"]["minimum_ppm"]
        ),
        "repeated_alphanumeric_gibberish": bool(_REPEATED_ALPHANUMERIC.search(text)),
        "placeholder_lorem_ipsum": bool(
            lorem_full_count
            >= POLICY["hard_reject"]["placeholder_lorem_ipsum"][
                "minimum_full_phrase_occurrences"
            ]
            or lorem_short_count
            >= POLICY["hard_reject"]["placeholder_lorem_ipsum"][
                "minimum_short_phrase_occurrences"
            ]
        ),
        "url_only_link_index": bool(
            len(url_matches)
            >= POLICY["context_review"]["url_only_link_index"]["minimum_urls"]
            and url_line_ppm
            >= POLICY["context_review"]["url_only_link_index"]["minimum_url_line_ppm"]
            and non_url_words
            <= len(url_matches)
            * POLICY["context_review"]["url_only_link_index"][
                "maximum_non_url_words_per_url"
            ]
        ),
        "markup_only_fragment": bool(
            tag_count
            >= POLICY["context_review"]["markup_only_fragment"]["minimum_tags"]
            and visible_words
            <= POLICY["context_review"]["markup_only_fragment"]["maximum_visible_words"]
        ),
        "contextless_structured_fragment": bool(
            utf8_bytes
            >= POLICY["context_review"]["contextless_structured_fragment"][
                "minimum_utf8_bytes"
            ]
            and len(_ALPHA_WORD.findall(text))
            <= POLICY["context_review"]["contextless_structured_fragment"][
                "maximum_alpha_words"
            ]
            and digit_symbol_ppm
            >= POLICY["context_review"]["contextless_structured_fragment"][
                "minimum_digit_symbol_ppm"
            ]
        ),
        "contextless_metadata_form": bool(
            len(nonempty_lines)
            >= POLICY["context_review"]["contextless_metadata_form"][
                "minimum_nonempty_lines"
            ]
            and utf8_bytes
            <= POLICY["context_review"]["contextless_metadata_form"][
                "maximum_utf8_bytes"
            ]
            and metadata_field_line_count
            >= POLICY["context_review"]["contextless_metadata_form"][
                "minimum_metadata_field_lines"
            ]
            and metadata_field_line_ppm
            >= POLICY["context_review"]["contextless_metadata_form"][
                "minimum_metadata_field_line_ppm"
            ]
            and median_line_characters
            <= POLICY["context_review"]["contextless_metadata_form"][
                "maximum_median_line_characters"
            ]
            and maximum_line_alpha_words
            <= POLICY["context_review"]["contextless_metadata_form"][
                "maximum_line_alpha_words"
            ]
        ),
        "web_navigation_shell": bool(
            utf8_bytes
            <= POLICY["context_review"]["web_navigation_shell"][
                "maximum_utf8_bytes"
            ]
            and len(nonempty_lines)
            >= POLICY["context_review"]["web_navigation_shell"][
                "minimum_nonempty_lines"
            ]
            and len(shell_markers)
            >= POLICY["context_review"]["web_navigation_shell"][
                "minimum_distinct_shell_markers"
            ]
            and short_line_ppm
            >= POLICY["context_review"]["web_navigation_shell"][
                "minimum_short_line_ppm"
            ]
            and alpha_word_count
            <= POLICY["context_review"]["web_navigation_shell"][
                "maximum_alpha_words"
            ]
        ),
        "access_or_error_placeholder": bool(
            utf8_bytes
            <= POLICY["context_review"]["access_or_error_placeholder"][
                "maximum_utf8_bytes"
            ]
            and len(error_markers)
            >= POLICY["context_review"]["access_or_error_placeholder"][
                "minimum_distinct_markers"
            ]
            and alpha_word_count
            <= POLICY["context_review"]["access_or_error_placeholder"][
                "maximum_alpha_words"
            ]
        ),
        "duplicated_boilerplate": bool(
            len(nonempty_lines)
            >= POLICY["cleanup_review"]["duplicated_boilerplate"][
                "minimum_nonempty_lines"
            ]
            and maximum_repeated_line_count
            >= POLICY["cleanup_review"]["duplicated_boilerplate"][
                "minimum_repeated_line_count"
            ]
            and duplicate_character_ppm
            >= POLICY["cleanup_review"]["duplicated_boilerplate"][
                "minimum_duplicate_character_ppm"
            ]
        ),
    }
    hard_reasons = [
        key
        for key in (
            "contextless_mcq_answer_key",
            "contextless_scored_answer_sheet",
            "control_character_corruption",
            "unicode_replacement_corruption",
            "repeated_alphanumeric_gibberish",
            "placeholder_lorem_ipsum",
        )
        if flags[key]
    ]
    context_reasons = [
        key
        for key in (
            "url_only_link_index",
            "markup_only_fragment",
            "contextless_structured_fragment",
            "contextless_metadata_form",
            "web_navigation_shell",
            "access_or_error_placeholder",
        )
        if flags[key]
    ]
    cleanup_reasons = (
        ["duplicated_boilerplate"] if flags["duplicated_boilerplate"] else []
    )
    if hard_reasons:
        decision, reasons = "hard_reject", hard_reasons
    elif context_reasons:
        decision, reasons = "context_review", context_reasons
    elif cleanup_reasons:
        decision, reasons = "cleanup_review", cleanup_reasons
    else:
        decision, reasons = "pass_mechanical_gate", []
    return {
        "measurements": {
            "utf8_bytes": utf8_bytes,
            "codepoints": codepoints,
            "nonempty_lines": len(nonempty_lines),
            "control_character_count": control_count,
            "control_character_ppm": control_ppm,
            "unicode_replacement_count": replacement_count,
            "unicode_replacement_ppm": replacement_ppm,
            "longest_repeated_alphanumeric_run": longest_run,
            "score_marker_count": score_marker_count,
            "numbered_or_lettered_part_count": numbered_or_lettered_part_count,
            "question_mark_count": question_mark_count,
            "url_count": len(url_matches),
            "url_line_ppm": url_line_ppm,
            "non_url_alpha_words": non_url_words,
            "html_tag_count": tag_count,
            "visible_alpha_words": visible_words,
            "digit_symbol_ppm": digit_symbol_ppm,
            "metadata_field_line_count": metadata_field_line_count,
            "metadata_field_line_ppm": metadata_field_line_ppm,
            "median_line_characters": median_line_characters,
            "maximum_line_alpha_words": maximum_line_alpha_words,
            "maximum_repeated_line_count": maximum_repeated_line_count,
            "duplicate_line_character_ppm": duplicate_character_ppm,
            "alpha_word_count": alpha_word_count,
            "short_line_ppm": short_line_ppm,
            "web_shell_markers": shell_markers,
            "error_placeholder_markers": error_markers,
            "lorem_full_phrase_count": lorem_full_count,
            "lorem_short_phrase_count": lorem_short_count,
        },
        "answer_key_evidence": answer_key,
        "flags": flags,
        "decision": decision,
        "reasons": reasons,
        "semantic_admission_complete": False,
        "training_ready": False,
    }


def _read_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise SourceQualityGateError("quality-gate source is missing or unsafe")
    rows = []
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise SourceQualityGateError(
                            "quality-gate source record differs"
                        )
                    if row.get("schema") == "sai-agent-data-candidate-v1":
                        row = normalize_candidate(row)
                    else:
                        text_key = (
                            "text"
                            if isinstance(row.get("text"), str)
                            else "text_excerpt"
                        )
                        text = row.get(text_key)
                        identity = row.get("candidate_identity_sha256")
                        content_sha256 = row.get("source_content_sha256")
                        if (
                            not isinstance(text, str)
                            or not text
                            or not isinstance(row.get("source"), dict)
                            or not isinstance(identity, str)
                            or len(identity) != 64
                            or not isinstance(content_sha256, str)
                            or len(content_sha256) != 64
                            or hashlib.sha256(text.encode()).hexdigest()
                            != content_sha256
                        ):
                            raise SourceQualityGateError(
                                "quality-gate source record differs"
                            )
                        try:
                            bytes.fromhex(identity)
                            bytes.fromhex(content_sha256)
                        except ValueError as error:
                            raise SourceQualityGateError(
                                "quality-gate source digest differs"
                            ) from error
                    rows.append(row)
                except (json.JSONDecodeError, RuntimeError) as error:
                    raise SourceQualityGateError(
                        f"quality-gate source row {line_number} differs"
                    ) from error
    except (OSError, UnicodeError) as error:
        raise SourceQualityGateError("quality-gate source cannot be read") from error
    identities = [row["candidate_identity_sha256"] for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise SourceQualityGateError("quality-gate source population differs")
    return rows


def _compute(source: Path, emit) -> dict[str, Any]:
    rows = _read_candidates(source)
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    ordered_decisions = []
    for candidate in rows:
        text = candidate.get("text")
        if not isinstance(text, str):
            text = candidate["text_excerpt"]
        evidence = mechanical_quality_evidence(text)
        decision = {
            "schema": DECISION_SCHEMA,
            "candidate_identity_sha256": candidate["candidate_identity_sha256"],
            "source_content_sha256": candidate["source_content_sha256"],
            "policy_sha256": POLICY_SHA256,
            **evidence,
        }
        decision["decision_sha256"] = canonical_sha256(decision)
        decision_counts[decision["decision"]] += 1
        reason_counts.update(decision["reasons"])
        ordered_decisions.append(decision["decision_sha256"])
        emit(decision)
    return {
        "source": {
            "path": str(source.resolve()),
            "rows": len(rows),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "ordered_identities_sha256": canonical_sha256(
                [row["candidate_identity_sha256"] for row in rows]
            ),
        },
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "ordered_decisions_sha256": canonical_sha256(ordered_decisions),
    }


def build(source: Path, decisions: Path, receipt: Path) -> dict[str, Any]:
    """Emit text-free decisions for one exact source population."""

    if (
        decisions.exists()
        or decisions.is_symlink()
        or receipt.exists()
        or receipt.is_symlink()
    ):
        raise SourceQualityGateError("quality-gate output exists")
    decisions.parent.mkdir(parents=True, exist_ok=True)
    temporary = decisions.parent / f".{decisions.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
        )
        with os.fdopen(descriptor, "w") as handle:

            def emit(row: dict[str, Any]) -> None:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )

            metadata = _compute(source, emit)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, decisions)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    payload = {
        "schema": SCHEMA,
        "status": "complete_mechanical_quality_gate",
        "policy": POLICY,
        "policy_sha256": POLICY_SHA256,
        **metadata,
        "decisions": {
            "path": str(decisions.resolve()),
            "rows": metadata["source"]["rows"],
            "bytes": decisions.stat().st_size,
            "sha256": sha256_file(decisions),
        },
        "all_nonpass_rows_excluded_from_direct_admission": True,
        "semantic_admission_complete": False,
        "rights_admission_complete": False,
        "benchmark_decontamination_complete": False,
        "global_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    stage = receipt.parent / f".{receipt.name}.{uuid.uuid4().hex}.tmp"
    stage.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(stage, receipt)
    return payload


def validate(receipt: Path) -> dict[str, Any]:
    """Replay the complete decision stream and compare every byte."""

    if not receipt.is_file() or receipt.is_symlink() or receipt.stat().st_nlink != 1:
        raise SourceQualityGateError("quality-gate receipt is missing or unsafe")
    try:
        payload = json.loads(receipt.read_text())
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise SourceQualityGateError("quality-gate receipt cannot be read") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "complete_mechanical_quality_gate"
        or payload.get("policy") != POLICY
        or payload.get("policy_sha256") != POLICY_SHA256
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise SourceQualityGateError("quality-gate receipt differs")
    source = Path(payload.get("source", {}).get("path", ""))
    decisions = Path(payload.get("decisions", {}).get("path", ""))
    if (
        not decisions.is_file()
        or decisions.is_symlink()
        or decisions.stat().st_nlink != 1
        or decisions.stat().st_size != payload.get("decisions", {}).get("bytes")
        or sha256_file(decisions) != payload.get("decisions", {}).get("sha256")
    ):
        raise SourceQualityGateError("quality-gate decisions differ")
    with decisions.open() as handle:

        def compare(row: dict[str, Any]) -> None:
            expected = (
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )
            if handle.readline() != expected:
                raise SourceQualityGateError("quality-gate decision replay differs")

        metadata = _compute(source, compare)
        if handle.read(1):
            raise SourceQualityGateError("quality-gate decision replay differs")
    for key, value in metadata.items():
        if payload.get(key) != value:
            raise SourceQualityGateError("quality-gate receipt replay differs")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source", type=Path, required=True)
    build_parser.add_argument("--decisions", type=Path, required=True)
    build_parser.add_argument("--receipt", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        build(args.source, args.decisions, args.receipt)
        if args.command == "build"
        else validate(args.receipt)
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["source"]["rows"],
                "decision_counts": payload["decision_counts"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
