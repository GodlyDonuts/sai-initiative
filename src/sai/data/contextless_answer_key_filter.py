"""Reject high-confidence contextless multiple-choice answer keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-contextless-answer-key-filter-v1"
POLICY = {
    "answer_entry": "number_plus_single_letter_A_through_E",
    "minimum_entries_with_answer_key_marker": 5,
    "minimum_entries_without_marker": 8,
    "minimum_unique_question_numbers": 5,
    "maximum_context_words_per_entry": 1,
    "question_mark_veto": True,
    "decision": "reject_only_high_confidence_contextless_mcq_answer_keys",
}
POLICY_SHA256 = canonical_sha256(POLICY)
_ENTRY = re.compile(
    r"(?<![\w.])(?:q(?:uestion)?\s*)?(\d{1,4})\s*(?:[.)\]:-]\s*)?"
    r"(?:answer\s*[:=-]?\s*)?[([]?([A-Ea-e])[]) ]?(?=\s|[,;]|$)",
    re.IGNORECASE,
)
_ANSWER_KEY = re.compile(r"\b(?:answer|solutions?)\s*key\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
_STRUCTURAL_WORDS = {
    "a",
    "b",
    "c",
    "d",
    "e",
    "answer",
    "answers",
    "key",
    "keys",
    "question",
    "questions",
    "solution",
    "solutions",
    "mcq",
}


class ContextlessAnswerKeyError(RuntimeError):
    """The source, answer-key decision, output, or replay differs."""


def answer_key_evidence(text: str) -> dict[str, Any]:
    """Return bounded mechanical evidence without retaining source excerpts."""

    if not isinstance(text, str):
        raise ContextlessAnswerKeyError("answer-key text differs")
    matches = list(_ENTRY.finditer(text))
    numbers = {int(match.group(1)) for match in matches}
    marker = _ANSWER_KEY.search(text) is not None
    context_words = sum(
        1
        for match in _WORD.finditer(_ENTRY.sub(" ", _ANSWER_KEY.sub(" ", text)))
        if match.group().casefold() not in _STRUCTURAL_WORDS
    )
    minimum_entries = (
        POLICY["minimum_entries_with_answer_key_marker"]
        if marker
        else POLICY["minimum_entries_without_marker"]
    )
    rejected = bool(
        len(matches) >= minimum_entries
        and len(numbers) >= POLICY["minimum_unique_question_numbers"]
        and context_words
        <= len(matches) * POLICY["maximum_context_words_per_entry"]
        and (not POLICY["question_mark_veto"] or "?" not in text)
    )
    return {
        "answer_entry_count": len(matches),
        "unique_question_number_count": len(numbers),
        "answer_key_marker": marker,
        "context_word_count": context_words,
        "question_mark_present": "?" in text,
        "contextless_answer_key": rejected,
    }


def _compute(source: Path, on_accepted) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink() or source.stat().st_size <= 0:
        raise ContextlessAnswerKeyError("answer-key source is missing")
    source_sha256 = sha256_file(source)
    scanned = accepted = dropped = 0
    accepted_digest = hashlib.sha256()
    dropped_digest = hashlib.sha256()
    with source.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                document = normalize_document(json.loads(line))
            except (json.JSONDecodeError, RuntimeError) as error:
                raise ContextlessAnswerKeyError(
                    f"answer-key source row {line_number} differs"
                ) from error
            scanned += 1
            evidence = answer_key_evidence(document["text"])
            decision = {
                "document_identity_sha256": document["identity_sha256"],
                "policy_sha256": POLICY_SHA256,
                "evidence": evidence,
            }
            decision_sha256 = canonical_sha256(decision)
            if evidence["contextless_answer_key"]:
                dropped += 1
                dropped_digest.update(bytes.fromhex(decision_sha256))
                continue
            accepted += 1
            accepted_digest.update(bytes.fromhex(document["identity_sha256"]))
            on_accepted(document)
    if not scanned or not accepted or scanned != accepted + dropped:
        raise ContextlessAnswerKeyError("answer-key filter row coverage differs")
    return {
        "source": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": source_sha256,
        },
        "policy": POLICY,
        "policy_sha256": POLICY_SHA256,
        "scanned": scanned,
        "accepted": accepted,
        "dropped_contextless_answer_keys": dropped,
        "accepted_identity_sha256": accepted_digest.hexdigest(),
        "dropped_evidence_sha256": dropped_digest.hexdigest(),
    }


def build(source: Path, output: Path, receipt: Path) -> dict[str, Any]:
    """Filter one benchmark-disjoint population while preserving accepted rows."""

    if (
        output.exists()
        or output.is_symlink()
        or receipt.exists()
        or receipt.is_symlink()
    ):
        raise ContextlessAnswerKeyError("answer-key filter output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        with stage.open("x") as handle:

            def write_row(row: dict[str, Any]) -> None:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )

            metadata = _compute(source, write_row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, output)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    payload = {
        "schema": SCHEMA,
        "status": "passed_contextless_answer_key_filter",
        **metadata,
        "output": {
            "path": str(output.resolve()),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "benchmark_disjointness_preserved": True,
        "semantic_quality_review_complete": False,
        "rights_admission_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    stage = receipt.parent / f".{receipt.name}.{uuid.uuid4().hex}.tmp"
    stage.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(stage, receipt)
    return payload


def validate(receipt: Path) -> dict[str, Any]:
    """Replay source decisions and byte-compare every accepted row."""

    if not receipt.is_file() or receipt.is_symlink():
        raise ContextlessAnswerKeyError("answer-key receipt is missing")
    payload = json.loads(receipt.read_text())
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("schema") != SCHEMA or payload.get(
        "receipt_sha256"
    ) != canonical_sha256(unsigned):
        raise ContextlessAnswerKeyError("answer-key receipt differs")
    source = Path(payload.get("source", {}).get("path", ""))
    output = Path(payload.get("output", {}).get("path", ""))
    if (
        not output.is_file()
        or output.is_symlink()
        or output.stat().st_size != payload.get("output", {}).get("bytes")
        or sha256_file(output) != payload.get("output", {}).get("sha256")
    ):
        raise ContextlessAnswerKeyError("answer-key output differs")
    with output.open() as handle:

        def compare(row: dict[str, Any]) -> None:
            expected = (
                json.dumps(
                    row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                + "\n"
            )
            if handle.readline() != expected:
                raise ContextlessAnswerKeyError("answer-key output replay differs")

        metadata = _compute(source, compare)
        if handle.read(1):
            raise ContextlessAnswerKeyError("answer-key output replay differs")
    for key, value in metadata.items():
        if payload.get(key) != value:
            raise ContextlessAnswerKeyError("answer-key receipt replay differs")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--receipt", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        build(args.source, args.output, args.receipt)
        if args.command == "build"
        else validate(args.receipt)
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "dropped": payload["dropped_contextless_answer_keys"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
