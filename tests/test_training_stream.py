from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sai.data.token_stream import ROW_SCHEMA, freeze
from sai.training.stream import (
    IGNORE_TARGET,
    ReceiptBoundTokenStream,
    StreamCursor,
    TrainingStreamError,
)


class CharacterTokenizer:
    eos_token_id = 0
    vocab_size = 512

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": [ord(character) + 1 for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id - 1) for token_id in token_ids)


def _document(index: int, text: str) -> dict:
    return {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": "mechanics-test",
            "row_id": str(index),
            "license": "CC0-1.0",
            "domain": "technical",
        },
        "verification": {
            "benchmark_disjoint": True,
            "evidence_sha256": hashlib.sha256(f"evidence-{index}".encode()).hexdigest(),
        },
    }


def _stream(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "source.jsonl"
    rows = [
        _document(0, "abc"),
        _document(1, "de"),
        _document(2, "fghij"),
        _document(3, "klmnop"),
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    root = tmp_path / "stream"
    report = freeze(
        CharacterTokenizer(),
        [source],
        root,
        tokenizer_identity_sha256="a" * 64,
        sequence_length=8,
        prefix_sequences={1, 2},
        sequences_per_shard=1,
    )
    return root, report


def _open(root: Path, report: dict, **kwargs) -> ReceiptBoundTokenStream:
    return ReceiptBoundTokenStream(
        root,
        expected_ordered_stream_identity_sha256=report[
            "ordered_stream_identity_sha256"
        ],
        **kwargs,
    )


def test_loader_returns_shifted_rows_and_masks_cross_document_targets(
    tmp_path: Path,
) -> None:
    root, report = _stream(tmp_path)
    batch = _open(root, report).next_batch(1)

    assert batch.x == (
        (
            ord("a") + 1,
            ord("b") + 1,
            ord("c") + 1,
            0,
            ord("d") + 1,
            ord("e") + 1,
            0,
        ),
    )
    assert batch.y == (
        (
            ord("b") + 1,
            ord("c") + 1,
            0,
            IGNORE_TARGET,
            ord("e") + 1,
            0,
            IGNORE_TARGET,
        ),
    )
    assert batch.segment_ids == ((0, 0, 0, 0, 1, 1, 1),)
    assert batch.loss_mask == ((True, True, True, False, True, True, False),)
    assert batch.first_sequence == 0
    assert batch.resume_cursor.next_sequence == 1


def test_resume_cursor_reopens_at_the_exact_next_sequence(tmp_path: Path) -> None:
    root, report = _stream(tmp_path)
    uninterrupted = _open(root, report)
    uninterrupted.next_batch(1)
    expected_second = uninterrupted.next_batch(1)

    first = _open(root, report).next_batch(1)
    cursor_receipt = json.loads(json.dumps(first.resume_cursor.as_dict()))
    resumed = _open(root, report, resume_cursor=cursor_receipt)
    assert resumed.cursor == StreamCursor(report["ordered_stream_identity_sha256"], 1)
    assert resumed.next_batch(1) == expected_second
    assert resumed.remaining_sequences == 0


def test_cursor_is_bound_to_receipt_and_exact_geometry(tmp_path: Path) -> None:
    root, report = _stream(tmp_path)
    cursor = StreamCursor(report["ordered_stream_identity_sha256"], 1).as_dict()

    for key, replacement in (
        ("schema", "wrong-schema"),
        ("ordered_stream_identity_sha256", "b" * 64),
        ("next_sequence", -1),
        ("next_sequence", True),
        ("next_sequence", 3),
    ):
        tampered = {**cursor, key: replacement}
        with pytest.raises(TrainingStreamError, match="cursor"):
            _open(root, report, resume_cursor=tampered)

    with pytest.raises(TrainingStreamError, match="cursor"):
        _open(root, report, resume_cursor={**cursor, "extra": 1})


def test_receipt_or_shard_tampering_fails_before_iteration(tmp_path: Path) -> None:
    root, report = _stream(tmp_path)
    with pytest.raises(TrainingStreamError, match="receipt identity"):
        ReceiptBoundTokenStream(
            root,
            expected_ordered_stream_identity_sha256="b" * 64,
        )

    token_path = root / report["shards"][0]["tokens"]["path"]
    payload = bytearray(token_path.read_bytes())
    payload[0] ^= 1
    token_path.write_bytes(payload)
    with pytest.raises(TrainingStreamError, match="validation failed"):
        _open(root, report)


def test_source_tampering_and_partial_batch_fail_closed(tmp_path: Path) -> None:
    root, report = _stream(tmp_path)
    source = Path(report["source_receipts"][0]["path"])
    source.write_text(source.read_text() + "\n")
    with pytest.raises(TrainingStreamError, match="validation failed"):
        _open(root, report)

    loader = _open(root, report, verify_sources=False)
    before = loader.cursor
    with pytest.raises(TrainingStreamError, match="full batch"):
        loader.next_batch(3)
    assert loader.cursor == before
