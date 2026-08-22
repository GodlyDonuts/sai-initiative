from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.token_stream import ROW_SCHEMA, sha256_tree
from sai.tokenizer.build import TokenizerBuildError, build_candidates


def test_tokenizer_tournament_job_is_cpu_only_and_replays_decontamination() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (root / "jobs" / "sai-tokenizer-tournament-cpu.sbatch").read_text()
    assert "--no-requeue" in job
    assert "--gres=" not in job
    assert "sai.data.decontamination validate" in job
    assert "sai.data.split" in job
    assert "sai.tokenizer.build" in job
    assert "sai.tokenizer.qualification" in job


def _row(index: int) -> dict:
    text = (
        f"Document {index}: def f_{index}(x): return x * {index + 1}. "
        f"The measured energy is {index}.125e-3 joules. "
        + "technical English mathematics science code " * 12
    )
    return {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": "tokenizer-test",
            "row_id": str(index),
            "license": "CC0-1.0",
            "domain": "technical",
        },
        "verification": {
            "benchmark_disjoint": True,
            "evidence_sha256": hashlib.sha256(f"evidence-{index}".encode()).hexdigest(),
        },
    }


def _corpus(path: Path) -> None:
    path.write_text("".join(json.dumps(_row(index)) + "\n" for index in range(80)))


def test_builds_exact_lossless_candidate_trees(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    corpus = tmp_path / "corpus.jsonl"
    _corpus(corpus)
    output = tmp_path / "tokenizers"

    manifest = build_candidates([corpus], output, sizes={"small": 280, "large": 300})

    assert manifest["status"] == "complete"
    assert manifest["candidate_build_authorized"] is True
    assert set(manifest["candidates"]) == {"small", "large"}
    for name, size in {"small": 280, "large": 300}.items():
        root = output / name
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=False, use_fast=True
        )
        assert len(tokenizer.get_vocab()) == size
        assert tokenizer.eos_token == "<|eos|>"
        assert manifest["candidates"][name]["tree_sha256"] == sha256_tree(root)
        text = "def solve(x):\n    return x**2 + 3.14e-8  λ"
        assert (
            tokenizer.decode(
                tokenizer.encode(text, add_special_tokens=False),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            == text
        )


def test_rejects_unsafe_inputs_and_existing_output(tmp_path: Path) -> None:
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    corpus = tmp_path / "corpus.jsonl"
    _corpus(corpus)

    with pytest.raises(TokenizerBuildError, match="sizes"):
        build_candidates([corpus], tmp_path / "bad", sizes={"tiny": 10})
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(TokenizerBuildError, match="already exists"):
        build_candidates([corpus], output, sizes={"small": 300})

    corpus.write_text("{}\n")
    with pytest.raises(TokenizerBuildError, match="corpus row"):
        build_candidates([corpus], tmp_path / "malformed", sizes={"small": 300})
