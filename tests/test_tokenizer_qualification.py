from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.token_stream import ROW_SCHEMA
from sai.tokenizer.qualification import (
    CANDIDATE_SIZES,
    TokenizerQualificationError,
    declares_byte_fallback,
    qualify,
    selected_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
PROTECTED_SUITE = ROOT / "docs" / "SAI_TOKENIZER_PROTECTED_SUITE.jsonl"


class ChunkTokenizer:
    eos_token_id = 0
    unk_token_id = 1
    all_special_ids = [0, 1]
    byte_fallback = True

    def __init__(self, vocab_size: int, chunk_size: int) -> None:
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.piece_to_id: dict[str, int] = {}
        self.id_to_piece: dict[int, str] = {}

    def get_vocab(self):
        return {f"piece-{index}": index for index in range(self.vocab_size)}

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        result = []
        for index in range(0, len(text), self.chunk_size):
            piece = text[index : index + self.chunk_size]
            if piece not in self.piece_to_id:
                token_id = len(self.piece_to_id) + 2
                assert token_id < self.vocab_size
                self.piece_to_id[piece] = token_id
                self.id_to_piece[token_id] = piece
            result.append(self.piece_to_id[piece])
        return result

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return "".join(self.id_to_piece[token_id] for token_id in token_ids)

    def convert_ids_to_tokens(self, token_id):
        return {0: "<eos>", 1: "<unk>"}.get(token_id, f"piece-{token_id}")


class NormalizingTokenizer(ChunkTokenizer):
    def decode(self, token_ids, **kwargs):
        return super().decode(token_ids, **kwargs).casefold()


class UnknownTokenizer(ChunkTokenizer):
    def encode(self, text, *, add_special_tokens=False):
        result = super().encode(text, add_special_tokens=add_special_tokens)
        result[0] = self.unk_token_id
        return result

    def decode(self, token_ids, **kwargs):
        return (
            "<unknown>"
            if self.unk_token_id in token_ids
            else super().decode(token_ids, **kwargs)
        )


def document(index: int, domain: str, text: str) -> dict:
    return {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": "tokenizer-fixture",
            "row_id": str(index),
            "license": "CC0-1.0",
            "domain": domain,
        },
        "verification": {
            "benchmark_disjoint": True,
            "evidence_sha256": hashlib.sha256(f"row-{index}".encode()).hexdigest(),
        },
    }


def corpus(tmp_path: Path) -> Path:
    rows = [
        document(0, "english", "A clear compact explanation."),
        document(1, "code", "def add(a, b):\n    return a + b"),
        document(2, "math", "For x in R, x^2 is nonnegative."),
        document(3, "science", "The speed of light is 2.99792458e8 m/s."),
        document(4, "technical", "CUDA kernels execute thread blocks."),
    ]
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def candidates() -> dict[str, ChunkTokenizer]:
    return {
        "64k": ChunkTokenizer(64_000, 4),
        "48k": ChunkTokenizer(48_000, 3),
        "32k": ChunkTokenizer(32_000, 2),
    }


def identities() -> dict[str, str]:
    return {name: f"{index + 1:064x}" for index, name in enumerate(CANDIDATE_SIZES)}


def test_exact_candidates_qualify_and_report_byte_normalized_fertility(
    tmp_path: Path,
) -> None:
    report = qualify(candidates(), identities(), [corpus(tmp_path)], PROTECTED_SUITE)
    assert report["status"] == "all_candidates_qualified"
    assert not report["training_authorized"]
    assert not report["candidate_build_authorized"]
    metrics = {
        name: row["corpus"]["tokens_per_1k_utf8_bytes"]
        for name, row in report["candidates"].items()
    }
    assert metrics["64k"] < metrics["48k"] < metrics["32k"]
    assert all(
        row["protected_suite"]["roundtrip_failures"] == 0
        for row in report["candidates"].values()
    )


def test_selected_48k_receipt_is_consumable_and_still_authorizes_nothing(
    tmp_path: Path,
) -> None:
    report = qualify(candidates(), identities(), [corpus(tmp_path)], PROTECTED_SUITE)
    receipt = selected_receipt(report)
    assert receipt["schema"] == "sai-tokenizer-qualification-receipt-v1"
    assert receipt["vocab_size"] == 48_000
    assert receipt["byte_fallback"]
    assert receipt["roundtrip_failures"] == 0
    assert not receipt["training_authorized"]


@pytest.mark.parametrize("candidate_name", ["64k", "48k", "32k"])
def test_lossy_or_unknown_candidate_rejects_entire_tournament(
    tmp_path: Path, candidate_name: str
) -> None:
    values = candidates()
    size = CANDIDATE_SIZES[candidate_name]
    values[candidate_name] = NormalizingTokenizer(size, 3)
    report = qualify(values, identities(), [corpus(tmp_path)], PROTECTED_SUITE)
    assert report["status"] == "candidate_rejected"
    assert not report["candidates"][candidate_name]["qualified"]
    with pytest.raises(TokenizerQualificationError, match="qualified 48K"):
        selected_receipt(report)


def test_declared_byte_fallback_is_required(tmp_path: Path) -> None:
    values = candidates()
    values["48k"].byte_fallback = False
    report = qualify(values, identities(), [corpus(tmp_path)], PROTECTED_SUITE)
    assert report["status"] == "candidate_rejected"
    assert not report["candidates"]["48k"]["byte_fallback"]


def test_unknown_token_emission_rejects_candidate(tmp_path: Path) -> None:
    values = candidates()
    values["48k"] = UnknownTokenizer(48_000, 3)
    report = qualify(values, identities(), [corpus(tmp_path)], PROTECTED_SUITE)
    assert report["status"] == "candidate_rejected"
    assert report["candidates"]["48k"]["corpus"]["unknown_tokens"] > 0


def test_special_token_contract_must_match_across_candidates(tmp_path: Path) -> None:
    values = candidates()
    values["32k"].all_special_ids = [0]
    with pytest.raises(TokenizerQualificationError, match="special-token contracts"):
        qualify(values, identities(), [corpus(tmp_path)], PROTECTED_SUITE)


def test_wrong_candidate_size_or_missing_candidate_fails_closed(tmp_path: Path) -> None:
    values = candidates()
    values["48k"] = ChunkTokenizer(47_999, 3)
    with pytest.raises(TokenizerQualificationError, match="vocabulary size"):
        qualify(values, identities(), [corpus(tmp_path)], PROTECTED_SUITE)
    del values["32k"]
    with pytest.raises(TokenizerQualificationError, match="exact 64K"):
        qualify(values, identities(), [corpus(tmp_path)], PROTECTED_SUITE)


def test_incomplete_protected_suite_fails_closed(tmp_path: Path) -> None:
    lines = PROTECTED_SUITE.read_text().splitlines()
    suite = tmp_path / "suite.jsonl"
    suite.write_text("\n".join(lines[:-3]) + "\n")
    with pytest.raises(TokenizerQualificationError, match="categories are incomplete"):
        qualify(candidates(), identities(), [corpus(tmp_path)], suite)


def test_tampered_report_cannot_produce_selected_receipt(tmp_path: Path) -> None:
    report = qualify(candidates(), identities(), [corpus(tmp_path)], PROTECTED_SUITE)
    report["candidates"]["48k"]["corpus"]["tokens"] -= 1
    with pytest.raises(TokenizerQualificationError, match="qualified 48K"):
        selected_receipt(report)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"model": {"type": "BPE", "byte_fallback": True}}, True),
        (
            {
                "model": {"type": "BPE"},
                "pre_tokenizer": {"type": "ByteLevel"},
                "decoder": {"type": "ByteLevel"},
            },
            True,
        ),
        (
            {
                "model": {"type": "BPE"},
                "pre_tokenizer": {"type": "ByteLevel"},
                "decoder": {"type": "BPEDecoder"},
            },
            False,
        ),
    ],
)
def test_byte_fallback_must_be_declared_by_candidate_tree(
    tmp_path: Path, payload: dict, expected: bool
) -> None:
    root = tmp_path / "tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text(json.dumps(payload))
    assert declares_byte_fallback(root) is expected


@pytest.mark.parametrize("mutation", ["missing_provenance", "bad_evidence"])
def test_corpus_provenance_and_evidence_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    path = corpus(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing_provenance":
        rows[0]["source"]["license"] = ""
    else:
        rows[0]["verification"]["evidence_sha256"] = "not-a-digest"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(TokenizerQualificationError, match="corpus row contract"):
        qualify(candidates(), identities(), [path], PROTECTED_SUITE)


def test_duplicate_corpus_document_identity_fails_closed(tmp_path: Path) -> None:
    path = corpus(tmp_path)
    first = path.read_text().splitlines()[0]
    path.write_text(path.read_text() + first + "\n")
    with pytest.raises(TokenizerQualificationError, match="identities are not unique"):
        qualify(candidates(), identities(), [path], PROTECTED_SUITE)
