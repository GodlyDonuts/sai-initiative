from __future__ import annotations

import hashlib
import json
import struct
from collections import UserDict
from pathlib import Path

import pytest

from sai.data.token_stream import (
    ROW_SCHEMA,
    TokenStreamError,
    causal_loss_mask_from_start_bits,
    decode_segment_starts,
    freeze,
    segment_ids_from_start_bits,
    sha256_tree,
    validate_frozen_stream,
)


class CharacterTokenizer:
    eos_token_id = 0
    vocab_size = 512

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": [ord(character) + 1 for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return "".join(chr(token_id - 1) for token_id in token_ids)


class GapTokenizer(CharacterTokenizer):
    def __call__(self, text, **kwargs):
        payload = super().__call__(text, **kwargs)
        payload["offset_mapping"][1] = (2, 2)
        return payload


class NormalizingTokenizer(CharacterTokenizer):
    def decode(self, token_ids, **kwargs):
        return super().decode(token_ids, **kwargs).upper()


class MappingTokenizer(CharacterTokenizer):
    def __call__(self, text, **kwargs):
        return UserDict(super().__call__(text, **kwargs))


def document(index: int, text: str, *, benchmark_disjoint: bool = True) -> dict:
    return {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": "synthetic-pretraining",
            "row_id": str(index),
            "license": "CC0-1.0",
            "domain": ("english", "code", "math", "science", "technical")[index % 5],
        },
        "verification": {
            "benchmark_disjoint": benchmark_disjoint,
            "evidence_sha256": hashlib.sha256(f"evidence-{index}".encode()).hexdigest(),
        },
    }


def write_documents(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def source(tmp_path: Path) -> Path:
    return write_documents(
        tmp_path / "source.jsonl",
        [
            document(0, "abc"),
            document(1, "de"),
            document(2, "fghij"),
            document(3, "kl"),
        ],
    )


def test_freeze_packs_exact_tokens_boundaries_and_utf8_prefixes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "stream"
    report = freeze(
        CharacterTokenizer(),
        [source(tmp_path)],
        output,
        tokenizer_identity_sha256="1" * 64,
        sequence_length=8,
        prefix_sequences={1, 2},
        sequences_per_shard=1,
    )
    assert validate_frozen_stream(output) == report
    assert report["prefix_utf8_bytes"] == {"1": 6, "2": 12}
    assert report["valid_tokens"] == 16
    assert report["documents"]["accepted"] == 4

    first_tokens = struct.unpack(
        "<8I", (output / "shard_00000.tokens.u32le").read_bytes()
    )
    assert first_tokens == (
        ord("a") + 1,
        ord("b") + 1,
        ord("c") + 1,
        0,
        ord("d") + 1,
        ord("e") + 1,
        0,
        ord("f") + 1,
    )
    first_payload = (output / "shard_00000.starts.bitset").read_bytes()
    second_payload = (output / "shard_00001.starts.bitset").read_bytes()
    first_starts = decode_segment_starts(first_payload, 8)
    second_starts = decode_segment_starts(second_payload, 8)
    assert [index for index, value in enumerate(first_starts) if value] == [0, 4, 7]
    assert [index for index, value in enumerate(second_starts) if value] == [0, 5]
    assert segment_ids_from_start_bits(first_payload, 8) == [0, 0, 0, 0, 1, 1, 1, 2]
    assert causal_loss_mask_from_start_bits(first_payload, 8) == [
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        False,
    ]


def test_freeze_accepts_hugging_face_style_mapping_output(tmp_path: Path) -> None:
    report = freeze(
        MappingTokenizer(),
        [source(tmp_path)],
        tmp_path / "stream",
        tokenizer_identity_sha256="a" * 64,
        sequence_length=8,
        prefix_sequences={1},
    )
    assert report["status"] == "complete"


def test_freeze_is_byte_deterministic_across_output_roots(tmp_path: Path) -> None:
    input_path = source(tmp_path)
    reports = []
    for name in ("first", "second"):
        reports.append(
            freeze(
                CharacterTokenizer(),
                [input_path],
                tmp_path / name,
                tokenizer_identity_sha256="2" * 64,
                sequence_length=8,
                prefix_sequences={1, 2},
                sequences_per_shard=2,
            )
        )
    assert reports[0] == reports[1]
    assert (tmp_path / "first" / "shard_00000.tokens.u32le").read_bytes() == (
        tmp_path / "second" / "shard_00000.tokens.u32le"
    ).read_bytes()


def test_duplicate_and_unverified_documents_never_enter_stream(tmp_path: Path) -> None:
    input_path = write_documents(
        tmp_path / "source.jsonl",
        [
            document(0, "abc"),
            document(1, "abc"),
            document(2, "blocked", benchmark_disjoint=False),
            document(3, "defghijklmnop"),
        ],
    )
    report = freeze(
        CharacterTokenizer(),
        [input_path],
        tmp_path / "stream",
        tokenizer_identity_sha256="3" * 64,
        sequence_length=8,
        prefix_sequences={2},
    )
    assert report["documents"]["duplicates_dropped"] == 1
    assert report["documents"]["malformed_or_unverified_dropped"] == 1
    assert report["documents"]["accepted"] == 2


@pytest.mark.parametrize("tokenizer", [GapTokenizer(), NormalizingTokenizer()])
def test_offset_gap_or_nonlossless_tokenizer_fails_closed(
    tmp_path: Path, tokenizer
) -> None:
    with pytest.raises(TokenStreamError, match="offsets|round trip"):
        freeze(
            tokenizer,
            [source(tmp_path)],
            tmp_path / "stream",
            tokenizer_identity_sha256="4" * 64,
            sequence_length=8,
            prefix_sequences={1},
        )


def test_insufficient_source_cleans_partial_stage(tmp_path: Path) -> None:
    input_path = write_documents(tmp_path / "small.jsonl", [document(0, "a")])
    output = tmp_path / "stream"
    with pytest.raises(TokenStreamError, match="cannot fill"):
        freeze(
            CharacterTokenizer(),
            [input_path],
            output,
            tokenizer_identity_sha256="5" * 64,
            sequence_length=8,
            prefix_sequences={2},
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".stream.partial.*"))


def test_shard_or_source_tampering_fails_replay(tmp_path: Path) -> None:
    input_path = source(tmp_path)
    output = tmp_path / "stream"
    freeze(
        CharacterTokenizer(),
        [input_path],
        output,
        tokenizer_identity_sha256="6" * 64,
        sequence_length=8,
        prefix_sequences={1, 2},
    )
    token_path = output / "shard_00000.tokens.u32le"
    token_path.write_bytes(token_path.read_bytes()[:-1] + b"x")
    with pytest.raises(TokenStreamError, match="shard"):
        validate_frozen_stream(output)

    token_path.write_bytes(token_path.read_bytes()[:-1] + b"\x00")
    input_path.write_text(input_path.read_text() + "\n")
    with pytest.raises(TokenStreamError, match="source content"):
        validate_frozen_stream(output)


def test_extra_output_member_fails_exact_membership(tmp_path: Path) -> None:
    output = tmp_path / "stream"
    freeze(
        CharacterTokenizer(),
        [source(tmp_path)],
        output,
        tokenizer_identity_sha256="7" * 64,
        sequence_length=8,
        prefix_sequences={1, 2},
    )
    (output / "unbound.cache").write_text("extra")
    with pytest.raises(TokenStreamError, match="membership"):
        validate_frozen_stream(output)


def test_invalid_segment_bitsets_fail_closed() -> None:
    with pytest.raises(TokenStreamError, match="begin a segment"):
        decode_segment_starts(b"\x00", 8)
    with pytest.raises(TokenStreamError, match="nonzero padding"):
        decode_segment_starts(b"\x81", 7)


def test_tokenizer_tree_hash_rejects_links_and_changes_with_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tokenizer"
    root.mkdir()
    (root / "tokenizer.json").write_text("first")
    first = sha256_tree(root)
    (root / "tokenizer.json").write_text("second")
    assert sha256_tree(root) != first
    (root / "link").symlink_to(root / "tokenizer.json")
    with pytest.raises(TokenStreamError, match="contains a link"):
        sha256_tree(root)


def test_mechanics_stream_job_is_cpu_only_and_freezes_exact_short_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    job = (root / "jobs" / "sai-freeze-mechanics-streams-cpu.sbatch").read_text()
    assert "--no-requeue" in job
    assert "--gres=" not in job
    assert "--prefix-sequences 256" in job
    assert "--prefix-sequences 48828" in job
    assert "--prefix-sequences 1024" in job
    assert "--prefix-sequences 4096" not in job
    assert "sai-tokenizer-qualification-receipt-v1" in job
    assert "sai-selected-tokenizer-receipt-v1" not in job
