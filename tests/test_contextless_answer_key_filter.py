import json
from pathlib import Path

from sai.data.contextless_answer_key_filter import (
    answer_key_evidence,
    build,
    validate,
)
from sai.data.token_stream import ROW_SCHEMA, canonical_sha256


def document(index: int, text: str) -> dict:
    row = {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": "unit",
            "row_id": f"row-{index}",
            "license": "test",
            "domain": "math",
        },
        "verification": {
            "benchmark_disjoint": True,
            "evidence_sha256": f"{index + 1:064x}",
        },
    }
    row["identity_sha256"] = canonical_sha256(row)
    return row


def test_rejects_bare_answer_key_without_context() -> None:
    text = "Answer Key\n1. B\n2. C\n3. A\n4. D\n5. E\n6. B\n7. A\n8. C"
    evidence = answer_key_evidence(text)
    assert evidence["answer_entry_count"] == 8
    assert evidence["context_word_count"] == 0
    assert evidence["contextless_answer_key"] is True


def test_retains_questions_or_explanations() -> None:
    questions = "\n".join(
        f"{index}. Which theorem applies here? Answer: A" for index in range(1, 9)
    )
    explained = "\n".join(
        f"{index}. A because the derivative is positive on the interval"
        for index in range(1, 9)
    )
    assert answer_key_evidence(questions)["contextless_answer_key"] is False
    assert answer_key_evidence(explained)["contextless_answer_key"] is False


def test_build_filters_and_replays_exactly(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    kept = document(0, "A worked explanation of the Pythagorean theorem.")
    dropped = document(1, "Answer Key\n1 A\n2 B\n3 C\n4 D\n5 E\n6 A\n7 B\n8 C")
    source.write_text(json.dumps(kept) + "\n" + json.dumps(dropped) + "\n")
    output = tmp_path / "output.jsonl"
    receipt = tmp_path / "receipt.json"
    report = build(source, output, receipt)
    assert report["scanned"] == 2
    assert report["accepted"] == 1
    assert report["dropped_contextless_answer_keys"] == 1
    assert json.loads(output.read_text())["text"] == kept["text"]
    assert validate(receipt) == report
