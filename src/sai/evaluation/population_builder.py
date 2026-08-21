"""Convert frozen Shohin benchmark pairs into Sai development MC populations.

The converter accepts only the exact question/assessor schemas written by the
Shohin dense-public-benchmark freezer.  It also requires the Sai training
decontamination receipt to name both inputs as benchmark boundaries.  The
small evaluator-facing receipt is accompanied by a complete conversion audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from sai.data.decontamination import POLICY as DECONTAMINATION_POLICY
from sai.data.decontamination import RECEIPT_SCHEMA as DECONTAMINATION_SCHEMA
from sai.evaluation.development_mc import DISJOINT_RECEIPT_SCHEMA

QUESTION_SCHEMA = "shohin-dense-public-benchmark-question-v1"
ASSESSOR_SCHEMA = "shohin-dense-public-benchmark-assessor-v1"
CONVERSION_SCHEMA = "sai-development-mc-population-conversion-v1"
SUPPORTED_BENCHMARKS = ("mmlu_pro", "musr")
MMLU_LETTERS = "ABCDEFGHIJKLMNOP"

MUSR_HINTS = {
    "murder_mystery": (
        "Before selecting a choice, explain your reasoning step by step. The "
        "murderer needs to have a means (access to weapon), motive (reason to "
        "kill the victim), and opportunity (access to crime scene) in order to "
        "have killed the victim. Innocent suspects may have two of these "
        "proven, but not all three. An innocent suspect may be suspicious for "
        "some other reason, but they will not have all of motive, means, and "
        "opportunity established.\n\nIf you believe that both suspects have "
        "motive, means, and opportunity, you should make an educated guess pick "
        "the one for whom these are best established. If you believe that "
        "neither suspect has all three established, then choose the suspect "
        "where these are most clearly established."
    ),
    "object_placements": (
        "Based on this story, we want to identify where someone believes that a "
        "certain object is at the end of the story. In order to do that, you "
        "need to read the story and keep track of where they think the object is "
        "at each point. When an object is moved, the person may observe its new "
        "location if they saw it move.\n\nTo see where an object ends up, they "
        "must be able to see the location that it moves to and not be too "
        "distracted by what they are doing. If they do not observe the object "
        "moving, then they will still believe it to be in the last location "
        "where they observed it."
    ),
    "team_allocation": (
        "The story should allow you to determine how good each person is at a "
        "skill. Roughly, each person is either great, acceptable, or bad at a "
        "task. We want to find an optimal assignment of people to tasks that "
        "uses their skills as well as possible. In addition, one task will have "
        "to have two people assigned to it. The effectiveness of their teamwork "
        "(great team, acceptable team, or bad team) also impacts the overall "
        "quality of the assignment.\n\nWhen two people need to work on a task "
        "and one is bad at it, they don't necessarily benefit from the other "
        "person being good, unless they work well together.\n\nWith different "
        "strengths, weaknesses, and interpersonal dynamics at play, you should "
        "allocate your team to find the single assignment to ensure that the "
        "tasks overall are completed as effectively as possible.\n\n"
    ),
}
MUSR_FINAL_INSTRUCTION = (
    "Explain your reasoning step by step before you answer. Finally, the last "
    'thing you generate should be "ANSWER: (your answer here, including the '
    'choice number)"'
)
PARSER_CONTRACT = {
    "schema": "sai-development-mc-shohin-parser-v1",
    "question_schema": QUESTION_SCHEMA,
    "assessor_schema": ASSESSOR_SCHEMA,
    "identity": "sha256(benchmark\\0upstream_id\\0normalized_prompt_sha256)",
    "normalized_prompt": "single_ascii_space_join_of_unicode_whitespace_split",
    "mmlu_pro": (
        "official_five_shot_cot_tail_question_and_lettered_options_with_"
        "surrounding_whitespace_normalized"
    ),
    "musr": "official_cot_zero_shot_context_question_and_numbered_choices",
    "answer_index": "zero_based",
    "pairing": "same_line_same_id_same_upstream_id_same_normalized_prompt_hash",
}


class PopulationConversionError(RuntimeError):
    """An input, parser, custody, or output invariant differs."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


PARSER_CONTRACT_SHA256 = canonical_sha256(PARSER_CONTRACT)


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise PopulationConversionError(f"artifact is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PopulationConversionError(f"{field} must be a lowercase SHA256")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PopulationConversionError(f"{field} must be a positive integer")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise PopulationConversionError(f"{field} must be a nonempty stripped string")
    return value


def _normalized_text_sha256(value: str) -> str:
    return hashlib.sha256(" ".join(value.split()).encode("utf-8")).hexdigest()


def _source_identity(benchmark: str, upstream_id: str, prompt: str) -> str:
    return hashlib.sha256(
        f"{benchmark}\0{upstream_id}\0{_normalized_text_sha256(prompt)}".encode()
    ).hexdigest()


def _load_jsonl(
    path: Path, *, expected_sha256: str, expected_rows: int, label: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(path)
    expected_sha256 = _sha256(expected_sha256, f"expected {label} SHA256")
    observed_sha256 = _sha256_file(path)
    if observed_sha256 != expected_sha256:
        raise PopulationConversionError(f"{label} SHA256 differs")
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    raise PopulationConversionError(
                        f"blank {label} row at line {line_number}"
                    )
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise PopulationConversionError(
                        f"{label} row {line_number} must be an object"
                    )
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PopulationConversionError(f"{label} JSONL is unreadable") from error
    if len(rows) != expected_rows:
        raise PopulationConversionError(f"{label} row count differs")
    return rows, {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": observed_sha256,
        "rows": len(rows),
    }


def _question_row(row: dict[str, Any], benchmark: str) -> dict[str, Any]:
    expected = {
        "schema",
        "id",
        "benchmark",
        "upstream_id",
        "question",
        "response_mode",
    }
    if set(row) != expected or row.get("schema") != QUESTION_SCHEMA:
        raise PopulationConversionError("question row schema differs")
    if row.get("benchmark") != benchmark or row.get("response_mode") != "general":
        raise PopulationConversionError("question row contract differs")
    identity = _sha256(row.get("id"), "question ID")
    upstream_id = _text(row.get("upstream_id"), "question upstream ID")
    prompt = _text(row.get("question"), "question prompt")
    if identity != _source_identity(benchmark, upstream_id, prompt):
        raise PopulationConversionError("question identity differs")
    return row


def _assessor_row(row: dict[str, Any], benchmark: str) -> dict[str, Any]:
    expected = {
        "schema",
        "id",
        "benchmark",
        "upstream_id",
        "stratum",
        "question_sha256",
        "assessor",
    }
    if set(row) != expected or row.get("schema") != ASSESSOR_SCHEMA:
        raise PopulationConversionError("assessor row schema differs")
    if row.get("benchmark") != benchmark or not isinstance(row.get("assessor"), dict):
        raise PopulationConversionError("assessor row contract differs")
    _sha256(row.get("id"), "assessor ID")
    _sha256(row.get("question_sha256"), "assessor question SHA256")
    _text(row.get("upstream_id"), "assessor upstream ID")
    _text(row.get("stratum"), "assessor stratum")
    return row


def _parse_labeled_choices(text: str, *, alphabet: str) -> list[str]:
    marker = re.compile(rf"^([{re.escape(alphabet)}])\. (.*)$")
    choices: list[str] = []
    expected_index = 0
    for line in text.splitlines():
        match = marker.fullmatch(line)
        if match:
            if (
                expected_index >= len(alphabet)
                or match.group(1) != alphabet[expected_index]
            ):
                raise PopulationConversionError("MMLU-Pro option order differs")
            choices.append(match.group(2))
            expected_index += 1
        elif not choices:
            raise PopulationConversionError("MMLU-Pro option prefix differs")
        else:
            choices[-1] += "\n" + line
    choices = [choice.strip() for choice in choices]
    if not 2 <= len(choices) <= len(alphabet) or any(not choice for choice in choices):
        raise PopulationConversionError("MMLU-Pro options differ")
    return choices


def _parse_mmlu(prompt: str, domain: str) -> tuple[str, list[str]]:
    prefix = (
        "The following are multiple choice questions (with answers) about "
        f'{domain}. Think step by step and then finish your answer with "the '
        'answer is (X)" where X is the correct letter choice.\n\n'
    )
    ending = "\nAnswer: Let's think step by step."
    if not prompt.startswith(prefix) or not prompt.endswith(ending):
        raise PopulationConversionError("MMLU-Pro prompt envelope differs")
    positions = [match.start() for match in re.finditer(r"(?m)^Question:\n", prompt)]
    if len(positions) < 6:
        raise PopulationConversionError("MMLU-Pro five-shot prompt differs")
    tail = prompt[positions[-1] + len("Question:\n") : -len(ending)]
    if "\nQuestion:\n" in tail or "\n\nQuestion:\n" in tail:
        raise PopulationConversionError("MMLU-Pro final question boundary is ambiguous")
    if tail.count("\nOptions:\n") != 1:
        raise PopulationConversionError("MMLU-Pro final options boundary differs")
    question, options_text = tail.split("\nOptions:\n")
    question = _text(question.strip(), "MMLU-Pro final question")
    return question, _parse_labeled_choices(options_text, alphabet=MMLU_LETTERS)


def _parse_numbered_choices(text: str) -> list[str]:
    marker = re.compile(r"^([1-9][0-9]*) - (.*)$")
    choices: list[str] = []
    expected = 1
    for line in text.splitlines():
        match = marker.fullmatch(line)
        if match:
            if int(match.group(1)) != expected:
                raise PopulationConversionError("MuSR choice order differs")
            choices.append(match.group(2))
            expected += 1
        elif not choices:
            raise PopulationConversionError("MuSR choice prefix differs")
        else:
            choices[-1] += "\n" + line
    if not 2 <= len(choices) <= 16 or any(
        not choice or choice != choice.strip() for choice in choices
    ):
        raise PopulationConversionError("MuSR choices differ")
    return choices


def _parse_musr(prompt: str, domain: str) -> tuple[str, str, list[str]]:
    if domain not in MUSR_HINTS:
        raise PopulationConversionError("MuSR domain differs")
    choice_marker = "\n\nPick one of the following choices:\n"
    suffix_marker = "\n\nYou must pick one option. "
    if prompt.count(choice_marker) != 1 or prompt.count(suffix_marker) != 1:
        raise PopulationConversionError("MuSR prompt boundary differs")
    body, remainder = prompt.split(choice_marker)
    choices_text, suffix_tail = remainder.split(suffix_marker)
    hint = MUSR_HINTS[domain]
    expected_tail = (
        "" if domain == "object_placements" else hint + " "
    ) + MUSR_FINAL_INSTRUCTION
    if suffix_tail != expected_tail:
        raise PopulationConversionError("MuSR final instruction differs")
    if domain == "object_placements":
        middle = f"\n\n{hint}\n\n"
        if body.count(middle) != 1:
            raise PopulationConversionError("MuSR hint placement differs")
        context, question = body.split(middle)
    else:
        if "\n\n" not in body:
            raise PopulationConversionError("MuSR context/question boundary differs")
        context, question = body.rsplit("\n\n", 1)
    return (
        _text(context, "MuSR context"),
        _text(question, "MuSR final question"),
        _parse_numbered_choices(choices_text),
    )


def _convert_pair(
    question_row: dict[str, Any], assessor_row: dict[str, Any], benchmark: str
) -> dict[str, Any]:
    question_row = _question_row(question_row, benchmark)
    assessor_row = _assessor_row(assessor_row, benchmark)
    for field in ("id", "upstream_id"):
        if question_row[field] != assessor_row[field]:
            raise PopulationConversionError(f"paired {field} order differs")
    prompt = question_row["question"]
    if assessor_row["question_sha256"] != _normalized_text_sha256(prompt):
        raise PopulationConversionError("paired question hash differs")
    domain = assessor_row["stratum"]
    assessor = assessor_row["assessor"]
    if benchmark == "mmlu_pro":
        if set(assessor) != {"answer", "category", "question_id"}:
            raise PopulationConversionError("MMLU-Pro assessor schema differs")
        if (
            assessor.get("category") != domain
            or str(assessor.get("question_id")) != question_row["upstream_id"]
        ):
            raise PopulationConversionError("MMLU-Pro assessor identity differs")
        question, choices = _parse_mmlu(prompt, domain)
        answer = assessor.get("answer")
        if not isinstance(answer, str) or answer not in MMLU_LETTERS[: len(choices)]:
            raise PopulationConversionError("MMLU-Pro answer differs")
        return {
            "benchmark": benchmark,
            "row_id": question_row["id"],
            "domain": domain,
            "question": question,
            "choices": choices,
            "answer_index": MMLU_LETTERS.index(answer),
        }
    if set(assessor) != {"answer", "choice_count", "domain"}:
        raise PopulationConversionError("MuSR assessor schema differs")
    context, question, choices = _parse_musr(prompt, domain)
    answer = assessor.get("answer")
    if (
        assessor.get("domain") != domain
        or isinstance(assessor.get("choice_count"), bool)
        or assessor.get("choice_count") != len(choices)
        or isinstance(answer, bool)
        or not isinstance(answer, int)
        or not 1 <= answer <= len(choices)
    ):
        raise PopulationConversionError("MuSR assessor answer differs")
    return {
        "benchmark": benchmark,
        "row_id": question_row["id"],
        "domain": domain,
        "context": context,
        "question": question,
        "choices": choices,
        "answer_index": answer - 1,
    }


def _validate_training_receipt(
    receipt_path: Path,
    *,
    expected_receipt_file_sha256: str,
    questions: dict[str, Any],
    assessors: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = Path(receipt_path)
    observed_file_sha256 = _sha256_file(receipt_path)
    if observed_file_sha256 != _sha256(
        expected_receipt_file_sha256,
        "expected training decontamination receipt file SHA256",
    ):
        raise PopulationConversionError(
            "training decontamination receipt SHA256 differs"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PopulationConversionError(
            "training decontamination receipt is unreadable"
        ) from error
    expected_keys = {
        "schema",
        "status",
        "source",
        "boundaries",
        "boundary_manifest_sha256",
        "policy",
        "policy_sha256",
        "scanned",
        "accepted",
        "dropped",
        "identity_accumulation",
        "accepted_identity_sha256",
        "dropped_evidence_sha256",
        "output",
        "receipt_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != expected_keys:
        raise PopulationConversionError(
            "training decontamination receipt schema differs"
        )
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("schema") != DECONTAMINATION_SCHEMA
        or receipt.get("status") != "passed"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise PopulationConversionError(
            "training decontamination receipt identity differs"
        )
    if (
        receipt.get("policy") != DECONTAMINATION_POLICY
        or receipt.get("policy_sha256") != canonical_sha256(DECONTAMINATION_POLICY)
        or receipt.get("identity_accumulation") != "ordered_raw_sha256_bytes"
    ):
        raise PopulationConversionError("training decontamination policy differs")
    for field in ("scanned", "accepted"):
        _positive_integer(receipt.get(field), f"decontamination {field}")
    dropped = receipt.get("dropped")
    if isinstance(dropped, bool) or not isinstance(dropped, int) or dropped < 0:
        raise PopulationConversionError("decontamination dropped count differs")
    _sha256(receipt.get("accepted_identity_sha256"), "accepted identity SHA256")
    _sha256(receipt.get("dropped_evidence_sha256"), "dropped evidence SHA256")
    boundaries = receipt.get("boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise PopulationConversionError("decontamination boundaries differ")
    expected_boundary_keys = {"order", "path", "bytes", "sha256", "rows", "strings"}
    for order, boundary in enumerate(boundaries):
        if not isinstance(boundary, dict) or set(boundary) != expected_boundary_keys:
            raise PopulationConversionError("decontamination boundary schema differs")
        if boundary.get("order") != order:
            raise PopulationConversionError("decontamination boundary order differs")
        _text(boundary.get("path"), "decontamination boundary path")
        _sha256(boundary.get("sha256"), "decontamination boundary SHA256")
        for field in ("bytes", "rows", "strings"):
            _positive_integer(boundary.get(field), f"decontamination boundary {field}")
    if (
        receipt.get("boundary_manifest_sha256") != canonical_sha256(boundaries)
        or not isinstance(receipt.get("source"), dict)
        or set(receipt["source"]) != {"path", "bytes", "sha256"}
        or not isinstance(receipt.get("output"), dict)
        or set(receipt["output"]) != {"path", "bytes", "sha256"}
    ):
        raise PopulationConversionError("decontamination artifact manifest differs")
    output = receipt["output"]
    training_path = Path(_text(output.get("path"), "decontaminated training path"))
    training_sha256 = _sha256(output.get("sha256"), "decontaminated training SHA256")
    _positive_integer(output.get("bytes"), "decontaminated training bytes")
    if (
        not training_path.is_file()
        or training_path.is_symlink()
        or str(training_path.resolve()) != output["path"]
        or training_path.stat().st_size != output["bytes"]
        or _sha256_file(training_path) != training_sha256
    ):
        raise PopulationConversionError("decontaminated training artifact differs")
    matched = []
    for artifact in (questions, assessors):
        candidates = [
            boundary
            for boundary in boundaries
            if boundary["path"] == artifact["path"]
            and boundary["sha256"] == artifact["sha256"]
            and boundary["bytes"] == artifact["bytes"]
            and boundary["rows"] == artifact["rows"]
        ]
        if len(candidates) != 1:
            raise PopulationConversionError(
                "benchmark input is not an exact decontamination boundary"
            )
        matched.append(candidates[0])
    return {
        "receipt": receipt,
        "receipt_path": str(receipt_path.resolve()),
        "receipt_file_sha256": observed_file_sha256,
        "training_source": {
            "path": output["path"],
            "bytes": output["bytes"],
            "sha256": training_sha256,
        },
        "matched_boundaries": matched,
    }


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True).encode() + b"\n"
    )


def _write_outputs(outputs: list[tuple[Path, bytes]]) -> None:
    if len({str(path.resolve()) for path, _ in outputs}) != len(outputs):
        raise PopulationConversionError("output paths are duplicated")
    for path, _ in outputs:
        if path.exists() or path.is_symlink():
            raise PopulationConversionError(f"refusing to replace output: {path}")
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise PopulationConversionError(
                f"output parent is missing or unsafe: {path}"
            )
    stages: list[tuple[Path, Path]] = []
    linked: list[Path] = []
    try:
        for path, content in outputs:
            stage = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
            descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            stages.append((stage, path))
        for stage, path in stages:
            os.link(stage, path)
            linked.append(path)
        for stage, _ in stages:
            stage.unlink()
        directories = {path.parent for path, _ in outputs}
        for parent in directories:
            descriptor = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for stage, _ in stages:
            stage.unlink(missing_ok=True)
        for path in linked:
            path.unlink(missing_ok=True)
        raise


def convert(
    *,
    benchmark: str,
    questions_path: Path,
    assessors_path: Path,
    expected_questions_sha256: str,
    expected_assessors_sha256: str,
    expected_rows: int,
    expected_identity_order_sha256: str,
    training_decontamination_receipt_path: Path,
    expected_training_decontamination_receipt_sha256: str,
    output_source_path: Path,
    output_disjoint_receipt_path: Path,
    output_conversion_receipt_path: Path,
) -> dict[str, Any]:
    """Convert one exact paired population and atomically write its custody set."""

    if benchmark not in SUPPORTED_BENCHMARKS:
        raise PopulationConversionError("benchmark is unsupported")
    expected_rows = _positive_integer(expected_rows, "expected rows")
    expected_identity_order_sha256 = _sha256(
        expected_identity_order_sha256, "expected identity-order SHA256"
    )
    questions, question_artifact = _load_jsonl(
        questions_path,
        expected_sha256=expected_questions_sha256,
        expected_rows=expected_rows,
        label="questions",
    )
    assessors, assessor_artifact = _load_jsonl(
        assessors_path,
        expected_sha256=expected_assessors_sha256,
        expected_rows=expected_rows,
        label="assessors",
    )
    converted = [
        _convert_pair(question, assessor, benchmark)
        for question, assessor in zip(questions, assessors, strict=True)
    ]
    identities = [row["row_id"] for row in converted]
    if len(set(identities)) != len(identities):
        raise PopulationConversionError("converted row identities are duplicated")
    identity_order_sha256 = canonical_sha256(identities)
    if identity_order_sha256 != expected_identity_order_sha256:
        raise PopulationConversionError("identity order differs")
    training = _validate_training_receipt(
        training_decontamination_receipt_path,
        expected_receipt_file_sha256=(expected_training_decontamination_receipt_sha256),
        questions=question_artifact,
        assessors=assessor_artifact,
    )
    source_bytes = _jsonl_bytes(converted)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    evidence = {
        "schema": "sai-development-mc-source-disjoint-evidence-v1",
        "benchmark": benchmark,
        "parser_contract_sha256": PARSER_CONTRACT_SHA256,
        "questions": question_artifact,
        "assessors": assessor_artifact,
        "paired_rows": len(converted),
        "identity_order_sha256": identity_order_sha256,
        "benchmark_source_sha256": source_sha256,
        "training_decontamination_receipt": {
            "path": training["receipt_path"],
            "file_sha256": training["receipt_file_sha256"],
            "receipt_sha256": training["receipt"]["receipt_sha256"],
            "policy_sha256": training["receipt"]["policy_sha256"],
            "boundary_manifest_sha256": training["receipt"]["boundary_manifest_sha256"],
        },
        "training_source": training["training_source"],
        "matched_decontamination_boundaries": training["matched_boundaries"],
    }
    evidence_sha256 = canonical_sha256(evidence)
    disjoint_receipt = {
        "schema": DISJOINT_RECEIPT_SCHEMA,
        "benchmark": benchmark,
        "benchmark_source_sha256": source_sha256,
        "training_source_sha256": training["training_source"]["sha256"],
        "source_disjoint": True,
        "method": "identity-and-contamination-audit",
        "evidence_sha256": evidence_sha256,
    }
    disjoint_bytes = _json_bytes(disjoint_receipt)
    conversion_receipt = {
        "schema": CONVERSION_SCHEMA,
        "status": "complete",
        "development_only": True,
        "official_benchmark_result": False,
        "public_terminal_result": False,
        "architecture_promotion_allowed": False,
        "benchmark": benchmark,
        "parser_contract": PARSER_CONTRACT,
        "parser_contract_sha256": PARSER_CONTRACT_SHA256,
        "evidence": evidence,
        "evidence_sha256": evidence_sha256,
        "output": {
            "source": {
                "path": str(Path(output_source_path).resolve()),
                "bytes": len(source_bytes),
                "sha256": source_sha256,
                "rows": len(converted),
                "identity_order_sha256": identity_order_sha256,
            },
            "source_disjoint_receipt": {
                "path": str(Path(output_disjoint_receipt_path).resolve()),
                "bytes": len(disjoint_bytes),
                "sha256": hashlib.sha256(disjoint_bytes).hexdigest(),
            },
        },
    }
    conversion_receipt["receipt_sha256"] = canonical_sha256(conversion_receipt)
    conversion_bytes = _json_bytes(conversion_receipt)
    _write_outputs(
        [
            (Path(output_source_path), source_bytes),
            (Path(output_disjoint_receipt_path), disjoint_bytes),
            (Path(output_conversion_receipt_path), conversion_bytes),
        ]
    )
    return conversion_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=SUPPORTED_BENCHMARKS, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--assessors", type=Path, required=True)
    parser.add_argument("--expected-questions-sha256", required=True)
    parser.add_argument("--expected-assessors-sha256", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--expected-identity-order-sha256", required=True)
    parser.add_argument("--training-decontamination-receipt", type=Path, required=True)
    parser.add_argument(
        "--expected-training-decontamination-receipt-sha256", required=True
    )
    parser.add_argument("--output-source", type=Path, required=True)
    parser.add_argument("--output-disjoint-receipt", type=Path, required=True)
    parser.add_argument("--output-conversion-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = convert(
        benchmark=args.benchmark,
        questions_path=args.questions,
        assessors_path=args.assessors,
        expected_questions_sha256=args.expected_questions_sha256,
        expected_assessors_sha256=args.expected_assessors_sha256,
        expected_rows=args.expected_rows,
        expected_identity_order_sha256=args.expected_identity_order_sha256,
        training_decontamination_receipt_path=(args.training_decontamination_receipt),
        expected_training_decontamination_receipt_sha256=(
            args.expected_training_decontamination_receipt_sha256
        ),
        output_source_path=args.output_source,
        output_disjoint_receipt_path=args.output_disjoint_receipt,
        output_conversion_receipt_path=args.output_conversion_receipt,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "benchmark": receipt["benchmark"],
                "source_sha256": receipt["output"]["source"]["sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
