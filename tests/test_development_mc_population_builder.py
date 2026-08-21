from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.decontamination import POLICY, RECEIPT_SCHEMA
from sai.evaluation.development_mc import DISJOINT_RECEIPT_SCHEMA
from sai.evaluation.population_builder import (
    ASSESSOR_SCHEMA,
    CONVERSION_SCHEMA,
    MUSR_FINAL_INSTRUCTION,
    MUSR_HINTS,
    PARSER_CONTRACT_SHA256,
    QUESTION_SCHEMA,
    PopulationConversionError,
    canonical_sha256,
    convert,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def identity(benchmark: str, upstream_id: str, prompt: str) -> str:
    return hashlib.sha256(
        f"{benchmark}\0{upstream_id}\0{normalized_sha256(prompt)}".encode()
    ).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def mmlu_prompt(domain: str, question: str, choices: list[str]) -> str:
    prefix = (
        "The following are multiple choice questions (with answers) about "
        f'{domain}. Think step by step and then finish your answer with "the '
        'answer is (X)" where X is the correct letter choice.\n\n'
    )
    examples = "".join(
        f"Question:\nExample {index}?\nOptions:\nA. yes\nB. no\n"
        "Answer: Let's think step by step. the answer is (A)\n\n"
        for index in range(5)
    )
    options = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip("ABCDEFGHIJKLMNOP", choices, strict=False)
    )
    return (
        f"{prefix}{examples}Question:\n{question}\nOptions:\n{options}\n"
        "Answer: Let's think step by step."
    )


def musr_prompt(domain: str, context: str, question: str, choices: list[str]) -> str:
    hint = MUSR_HINTS[domain]
    middle = f"{hint}\n\n{question}" if domain == "object_placements" else question
    choice_text = "\n".join(
        f"{index} - {choice}" for index, choice in enumerate(choices, 1)
    )
    suffix = (
        "You must pick one option. "
        + ("" if domain == "object_placements" else hint + " ")
        + MUSR_FINAL_INSTRUCTION
    )
    return (
        f"{context}\n\n{middle}\n\nPick one of the following choices:\n"
        f"{choice_text}\n\n{suffix}"
    )


def source_rows(
    benchmark: str, prompts: list[str], domain: str
) -> tuple[list[dict], list[dict], list[str]]:
    questions = []
    assessors = []
    identities = []
    for index, prompt in enumerate(prompts):
        upstream_id = str(index + 40)
        row_id = identity(benchmark, upstream_id, prompt)
        identities.append(row_id)
        questions.append(
            {
                "schema": QUESTION_SCHEMA,
                "id": row_id,
                "benchmark": benchmark,
                "upstream_id": upstream_id,
                "question": prompt,
                "response_mode": "general",
            }
        )
        assessor = (
            {"answer": "B", "category": domain, "question_id": int(upstream_id)}
            if benchmark == "mmlu_pro"
            else {"answer": 2, "choice_count": 2, "domain": domain}
        )
        assessors.append(
            {
                "schema": ASSESSOR_SCHEMA,
                "id": row_id,
                "benchmark": benchmark,
                "upstream_id": upstream_id,
                "stratum": domain,
                "question_sha256": normalized_sha256(prompt),
                "assessor": assessor,
            }
        )
    return questions, assessors, identities


def decontamination_receipt(
    tmp_path: Path,
    questions: Path,
    assessors: Path,
    *,
    include_questions: bool = True,
) -> tuple[Path, Path]:
    training = tmp_path / "admitted-training.jsonl"
    training.write_text('{"schema":"sai-pretraining-document-v1","text":"clean"}\n')
    boundary_paths = ([questions] if include_questions else []) + [assessors]
    boundaries = [
        {
            "order": order,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "rows": len(path.read_text().splitlines()),
            "strings": len(path.read_text().splitlines()),
        }
        for order, path in enumerate(boundary_paths)
    ]
    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "source": {"path": "/frozen/raw.jsonl", "bytes": 1, "sha256": "1" * 64},
        "boundaries": boundaries,
        "boundary_manifest_sha256": canonical_sha256(boundaries),
        "policy": POLICY,
        "policy_sha256": canonical_sha256(POLICY),
        "scanned": 2,
        "accepted": 1,
        "dropped": 1,
        "identity_accumulation": "ordered_raw_sha256_bytes",
        "accepted_identity_sha256": "2" * 64,
        "dropped_evidence_sha256": "3" * 64,
        "output": {
            "path": str(training.resolve()),
            "bytes": training.stat().st_size,
            "sha256": file_sha256(training),
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt = tmp_path / "decontamination.json"
    receipt.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return receipt, training


def run_conversion(
    tmp_path: Path,
    benchmark: str,
    prompts: list[str],
    domain: str,
    *,
    include_questions_boundary: bool = True,
) -> tuple[dict, Path, Path, Path, Path, Path]:
    questions_rows, assessor_rows, identities = source_rows(benchmark, prompts, domain)
    questions = write_jsonl(tmp_path / "questions.jsonl", questions_rows)
    assessors = write_jsonl(tmp_path / "assessors.jsonl", assessor_rows)
    decontamination, training = decontamination_receipt(
        tmp_path,
        questions,
        assessors,
        include_questions=include_questions_boundary,
    )
    output = tmp_path / "development.jsonl"
    disjoint = tmp_path / "disjoint.json"
    audit = tmp_path / "conversion.json"
    receipt = convert(
        benchmark=benchmark,
        questions_path=questions,
        assessors_path=assessors,
        expected_questions_sha256=file_sha256(questions),
        expected_assessors_sha256=file_sha256(assessors),
        expected_rows=len(prompts),
        expected_identity_order_sha256=canonical_sha256(identities),
        training_decontamination_receipt_path=decontamination,
        expected_training_decontamination_receipt_sha256=file_sha256(decontamination),
        output_source_path=output,
        output_disjoint_receipt_path=disjoint,
        output_conversion_receipt_path=audit,
    )
    return receipt, output, disjoint, audit, questions, assessors


def test_mmlu_pair_converts_tail_and_emits_evaluator_compatible_receipt(
    tmp_path: Path,
) -> None:
    prompt = mmlu_prompt(
        "physics", "Which result follows?", ["first", "second\ncontinued"]
    )
    receipt, output, disjoint, audit, _, _ = run_conversion(
        tmp_path, "mmlu_pro", [prompt], "physics"
    )
    row = json.loads(output.read_text())
    assert row == {
        "benchmark": "mmlu_pro",
        "row_id": row["row_id"],
        "domain": "physics",
        "question": "Which result follows?",
        "choices": ["first", "second\ncontinued"],
        "answer_index": 1,
    }
    evaluator_receipt = json.loads(disjoint.read_text())
    assert set(evaluator_receipt) == {
        "schema",
        "benchmark",
        "benchmark_source_sha256",
        "training_source_sha256",
        "source_disjoint",
        "method",
        "evidence_sha256",
    }
    assert evaluator_receipt["schema"] == DISJOINT_RECEIPT_SCHEMA
    assert evaluator_receipt["benchmark_source_sha256"] == file_sha256(output)
    assert evaluator_receipt["evidence_sha256"] == canonical_sha256(receipt["evidence"])
    assert receipt["schema"] == CONVERSION_SCHEMA
    assert receipt["parser_contract_sha256"] == PARSER_CONTRACT_SHA256
    assert json.loads(audit.read_text()) == receipt


def test_mmlu_normalizes_frozen_upstream_surrounding_whitespace(
    tmp_path: Path,
) -> None:
    prompt = mmlu_prompt("business", "Question with spacing?", ["alpha", "beta"])
    prompt = prompt.replace(
        "Question:\nQuestion with spacing?\nOptions:\nA. alpha\nB. beta\nAnswer:",
        "Question:\n Question with spacing? \nOptions:\nA. alpha \n\nB. beta\nAnswer:",
    )
    _, output, _, _, _, _ = run_conversion(tmp_path, "mmlu_pro", [prompt], "business")
    row = json.loads(output.read_text())
    assert row["question"] == "Question with spacing?"
    assert row["choices"] == ["alpha", "beta"]


@pytest.mark.parametrize(
    "domain",
    ["murder_mystery", "object_placements", "team_allocation"],
)
def test_musr_extracts_context_question_choices_and_one_based_answer(
    tmp_path: Path, domain: str
) -> None:
    prompt = musr_prompt(
        domain,
        "First paragraph.\n\nThe final context paragraph.",
        "Which choice is supported?",
        ["alpha", "beta"],
    )
    _, output, _, _, _, _ = run_conversion(tmp_path, "musr", [prompt], domain)
    row = json.loads(output.read_text())
    assert row["context"] == "First paragraph.\n\nThe final context paragraph."
    assert row["question"] == "Which choice is supported?"
    assert row["choices"] == ["alpha", "beta"]
    assert row["answer_index"] == 1


def test_hash_order_and_boundary_tampering_fail_closed(tmp_path: Path) -> None:
    prompts = [
        mmlu_prompt("logic", "First final question?", ["yes", "no"]),
        mmlu_prompt("logic", "Second final question?", ["yes", "no"]),
    ]
    questions_rows, assessor_rows, identities = source_rows(
        "mmlu_pro", prompts, "logic"
    )
    questions = write_jsonl(tmp_path / "questions.jsonl", questions_rows)
    assessors = write_jsonl(tmp_path / "assessors.jsonl", list(reversed(assessor_rows)))
    decontamination, _ = decontamination_receipt(tmp_path, questions, assessors)
    arguments = {
        "benchmark": "mmlu_pro",
        "questions_path": questions,
        "assessors_path": assessors,
        "expected_questions_sha256": file_sha256(questions),
        "expected_assessors_sha256": file_sha256(assessors),
        "expected_rows": 2,
        "expected_identity_order_sha256": canonical_sha256(identities),
        "training_decontamination_receipt_path": decontamination,
        "expected_training_decontamination_receipt_sha256": file_sha256(
            decontamination
        ),
        "output_source_path": tmp_path / "out.jsonl",
        "output_disjoint_receipt_path": tmp_path / "disjoint.json",
        "output_conversion_receipt_path": tmp_path / "audit.json",
    }
    with pytest.raises(PopulationConversionError, match="paired id order"):
        convert(**arguments)
    assessors.write_text(assessors.read_text() + " ")
    with pytest.raises(PopulationConversionError, match="assessors SHA256"):
        convert(**arguments)


def test_missing_exact_training_boundary_and_parse_deviation_fail_closed(
    tmp_path: Path,
) -> None:
    prompt = mmlu_prompt("logic", "Final question?", ["yes", "no"])
    with pytest.raises(
        PopulationConversionError, match="exact decontamination boundary"
    ):
        run_conversion(
            tmp_path,
            "mmlu_pro",
            [prompt],
            "logic",
            include_questions_boundary=False,
        )

    second = tmp_path / "parse"
    second.mkdir()
    malformed = prompt.replace("B. no", "C. no")
    with pytest.raises(PopulationConversionError, match="option order"):
        run_conversion(second, "mmlu_pro", [malformed], "logic")
