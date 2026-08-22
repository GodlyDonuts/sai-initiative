import hashlib
import json
from pathlib import Path

import pytest

from sai.tokenizer.numeric_pretokenization import (
    NumericPretokenizationError,
    build_numeric_ablation,
)
from sai.tokenizer.qualification import PROTECTED_CATEGORIES, PROTECTED_SCHEMA


def _corpus(path: Path) -> None:
    rows = [
        ("english", "The measured temperature is 273.15 kelvin."),
        ("math", "First add 12 and 30; then divide 42 by 7."),
        ("science", "Avogadro's constant is 6.02214076e23 mol^-1."),
        ("code", "for i in range(1024): total += values[i]"),
    ]
    with path.open("w") as handle:
        for index, (domain, text) in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "schema": "sai-pretraining-document-v1",
                        "text": text * 30,
                        "source": {
                            "dataset": "numeric-tokenizer-test",
                            "row_id": f"row-{index}",
                            "domain": domain,
                            "license": "CC0-1.0",
                        },
                        "verification": {
                            "benchmark_disjoint": True,
                            "evidence_sha256": hashlib.sha256(
                                f"evidence-{index}".encode()
                            ).hexdigest(),
                        },
                    }
                )
                + "\n"
            )


def _protected(path: Path) -> None:
    with path.open("w") as handle:
        for index, category in enumerate(sorted(PROTECTED_CATEGORIES)):
            text = (
                "1234567890 3.14159265 6.022e23 42kg 2048x4096"
                if category in {"numbers_and_units", "math", "science_notation"}
                else f"protected {category} row {index}"
            )
            handle.write(
                json.dumps(
                    {
                        "schema": PROTECTED_SCHEMA,
                        "id": f"protected-{index}",
                        "category": category,
                        "text": text,
                    }
                )
                + "\n"
            )


def test_builds_matched_lossless_numeric_ablation(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    protected = tmp_path / "protected.jsonl"
    output = tmp_path / "output"
    _corpus(corpus)
    _protected(protected)
    result = build_numeric_ablation([corpus], protected, output, vocab_size=300)
    assert result["status"] == "mechanically_qualified_capability_selection_pending"
    assert set(result["candidates"]) == {"individual_digits", "digit_runs"}
    assert all(row["qualified"] for row in result["candidates"].values())
    assert result["production_tokenizer_selected"] is False
    assert result["production_geometry"] is False
    assert (
        result["digit_runs_minus_individual"][
            "numbers_and_units_tokens_per_1k_utf8_bytes"
        ]["digit_runs"]
        < result["digit_runs_minus_individual"][
            "numbers_and_units_tokens_per_1k_utf8_bytes"
        ]["individual_digits"]
    )
    assert (output / "report.json").is_file()


def test_rejects_existing_output(tmp_path: Path):
    corpus = tmp_path / "corpus.jsonl"
    protected = tmp_path / "protected.jsonl"
    output = tmp_path / "output"
    _corpus(corpus)
    _protected(protected)
    output.mkdir()
    with pytest.raises(NumericPretokenizationError):
        build_numeric_ablation([corpus], protected, output, vocab_size=300)
