"""Deterministically reopen frozen token streams for mechanics training."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.token_stream import (
    TokenStreamError,
    causal_loss_mask_from_start_bits,
    segment_ids_from_start_bits,
    validate_frozen_stream,
)

CURSOR_SCHEMA = "sai-training-stream-cursor-v1"
IGNORE_TARGET = -100


class TrainingStreamError(RuntimeError):
    """A frozen stream, resume cursor, or requested batch differs."""


@dataclass(frozen=True)
class StreamCursor:
    """Exact next-sequence position bound to one ordered stream receipt."""

    ordered_stream_identity_sha256: str
    next_sequence: int
    schema: str = CURSOR_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "ordered_stream_identity_sha256": self.ordered_stream_identity_sha256,
            "next_sequence": self.next_sequence,
        }

    @classmethod
    def from_dict(cls, value: Any) -> StreamCursor:
        if not isinstance(value, dict) or set(value) != {
            "schema",
            "ordered_stream_identity_sha256",
            "next_sequence",
        }:
            raise TrainingStreamError("training stream cursor receipt differs")
        return cls(
            schema=value.get("schema"),
            ordered_stream_identity_sha256=value.get("ordered_stream_identity_sha256"),
            next_sequence=value.get("next_sequence"),
        )


@dataclass(frozen=True)
class TrainingBatch:
    """A batch of shifted causal examples and its exact resume position."""

    x: tuple[tuple[int, ...], ...]
    y: tuple[tuple[int, ...], ...]
    segment_ids: tuple[tuple[int, ...], ...]
    loss_mask: tuple[tuple[bool, ...], ...]
    first_sequence: int
    resume_cursor: StreamCursor


class ReceiptBoundTokenStream:
    """Sequential reader whose position cannot be replayed on another stream."""

    def __init__(
        self,
        root: Path,
        *,
        expected_ordered_stream_identity_sha256: str,
        resume_cursor: StreamCursor | dict[str, Any] | None = None,
        verify_sources: bool = True,
    ) -> None:
        try:
            report = validate_frozen_stream(root, verify_sources=verify_sources)
        except TokenStreamError as error:
            raise TrainingStreamError(
                "frozen token stream validation failed"
            ) from error
        identity = report["ordered_stream_identity_sha256"]
        if expected_ordered_stream_identity_sha256 != identity:
            raise TrainingStreamError("ordered stream receipt identity differs")

        cursor = self._normalize_cursor(resume_cursor, identity)
        sequences = report["sequences"]
        if cursor.next_sequence > sequences:
            raise TrainingStreamError("training stream cursor exceeds the receipt")

        self._root = root
        self._report = report
        self._identity = identity
        self._sequence_length = report["sequence_length"]
        self._start_bytes = (self._sequence_length + 7) // 8
        self._next_sequence = cursor.next_sequence

        first_sequence = 0
        shard_ranges = []
        for shard in report["shards"]:
            stop_sequence = first_sequence + shard["sequences"]
            shard_ranges.append((first_sequence, stop_sequence, shard))
            first_sequence = stop_sequence
        self._shard_ranges = tuple(shard_ranges)

    @staticmethod
    def _normalize_cursor(
        cursor: StreamCursor | dict[str, Any] | None, identity: str
    ) -> StreamCursor:
        if cursor is None:
            result = StreamCursor(identity, 0)
        elif isinstance(cursor, StreamCursor):
            result = cursor
        else:
            result = StreamCursor.from_dict(cursor)
        if (
            result.schema != CURSOR_SCHEMA
            or result.ordered_stream_identity_sha256 != identity
            or isinstance(result.next_sequence, bool)
            or not isinstance(result.next_sequence, int)
            or result.next_sequence < 0
        ):
            raise TrainingStreamError("training stream cursor receipt differs")
        return result

    @property
    def cursor(self) -> StreamCursor:
        return StreamCursor(self._identity, self._next_sequence)

    @property
    def remaining_sequences(self) -> int:
        return self._report["sequences"] - self._next_sequence

    def _read_sequence(self, sequence_index: int) -> tuple[tuple[int, ...], bytes]:
        for first, stop, shard in self._shard_ranges:
            if first <= sequence_index < stop:
                local_index = sequence_index - first
                token_path = self._root / shard["tokens"]["path"]
                start_path = self._root / shard["segment_starts"]["path"]
                try:
                    with token_path.open("rb") as handle:
                        handle.seek(local_index * self._sequence_length * 4)
                        token_payload = handle.read(self._sequence_length * 4)
                    with start_path.open("rb") as handle:
                        handle.seek(local_index * self._start_bytes)
                        start_payload = handle.read(self._start_bytes)
                except OSError as error:
                    raise TrainingStreamError(
                        "frozen token stream shard became unreadable"
                    ) from error
                if (
                    len(token_payload) != self._sequence_length * 4
                    or len(start_payload) != self._start_bytes
                ):
                    raise TrainingStreamError(
                        "frozen token stream shard became partial"
                    )
                tokens = struct.unpack(f"<{self._sequence_length}I", token_payload)
                return tokens, start_payload
        raise TrainingStreamError("training stream sequence is outside the receipt")

    def next_batch(self, batch_size: int) -> TrainingBatch:
        """Read the next complete sequences and advance only after full assembly."""

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise TrainingStreamError("training stream batch size differs")
        if batch_size > self.remaining_sequences:
            raise TrainingStreamError("training stream does not contain a full batch")

        first_sequence = self._next_sequence
        x_rows = []
        y_rows = []
        segment_rows = []
        mask_rows = []
        for sequence_index in range(first_sequence, first_sequence + batch_size):
            tokens, start_payload = self._read_sequence(sequence_index)
            segment_ids = segment_ids_from_start_bits(
                start_payload, self._sequence_length
            )
            causal_mask = causal_loss_mask_from_start_bits(
                start_payload, self._sequence_length
            )
            x = tokens[:-1]
            mask = tuple(causal_mask[:-1])
            y = tuple(
                target if admitted else IGNORE_TARGET
                for target, admitted in zip(tokens[1:], mask, strict=True)
            )
            if any(
                admitted != (left == right)
                for admitted, left, right in zip(
                    mask, segment_ids[:-1], segment_ids[1:], strict=True
                )
            ):
                raise TrainingStreamError("cross-document target mask differs")
            x_rows.append(tuple(x))
            y_rows.append(y)
            segment_rows.append(tuple(segment_ids[:-1]))
            mask_rows.append(mask)

        resume = StreamCursor(self._identity, first_sequence + batch_size)
        batch = TrainingBatch(
            x=tuple(x_rows),
            y=tuple(y_rows),
            segment_ids=tuple(segment_rows),
            loss_mask=tuple(mask_rows),
            first_sequence=first_sequence,
            resume_cursor=resume,
        )
        self._next_sequence = resume.next_sequence
        return batch
