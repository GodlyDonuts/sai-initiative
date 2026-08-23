"""Canonical raw-source and structural-segment identities."""

from __future__ import annotations

import hashlib
from typing import Any

from sai.data.token_stream import canonical_sha256

SEGMENT_SCHEMA = "sai-structural-segment-lineage-v1"


class SourceLineageError(RuntimeError):
    """A raw source locator or optional structural segment differs."""


def source_locator(source: Any) -> dict[str, Any]:
    """Return the immutable upstream row locator shared by all representations."""

    if (
        not isinstance(source, dict)
        or not isinstance(source.get("dataset"), str)
        or not source["dataset"]
        or not isinstance(source.get("revision"), str)
        or not source["revision"]
        or not isinstance(source.get("source_file"), str)
        or not source["source_file"]
        or not isinstance(source.get("row_index"), int)
        or isinstance(source.get("row_index"), bool)
        or source["row_index"] < 0
    ):
        raise SourceLineageError("raw source locator differs")
    return {
        "dataset": source["dataset"],
        "revision": source["revision"],
        "source_file": source["source_file"],
        "row_index": source["row_index"],
    }


def parent_row_id(source: Any) -> str:
    """Identify the exact upstream row independently of its representations."""

    return canonical_sha256(source_locator(source))


def structural_segment(source: Any, text: str) -> dict[str, Any] | None:
    """Validate and canonicalize optional lossless structural-segment lineage."""

    locator = source_locator(source)
    segment = source.get("segment")
    if segment is None:
        return None
    expected_fields = {
        "schema",
        "parent_row_id",
        "parent_text_sha256",
        "index",
        "count",
        "utf8_start",
        "utf8_end",
        "segment_text_sha256",
    }
    if (
        not isinstance(text, str)
        or not text
        or not isinstance(segment, dict)
        or set(segment) != expected_fields
    ):
        raise SourceLineageError("structural segment differs")
    integers = ("index", "count", "utf8_start", "utf8_end")
    if any(
        not isinstance(segment.get(field), int) or isinstance(segment.get(field), bool)
        for field in integers
    ):
        raise SourceLineageError("structural segment geometry differs")
    if (
        segment.get("schema") != SEGMENT_SCHEMA
        or segment.get("parent_row_id") != canonical_sha256(locator)
        or not isinstance(segment.get("parent_text_sha256"), str)
        or len(segment["parent_text_sha256"]) != 64
        or not isinstance(segment.get("segment_text_sha256"), str)
        or len(segment["segment_text_sha256"]) != 64
        or not 0 <= segment["index"] < segment["count"]
        or segment["count"] < 2
        or not 0 <= segment["utf8_start"] < segment["utf8_end"]
        or segment["utf8_end"] - segment["utf8_start"] != len(text.encode())
        or segment["segment_text_sha256"] != hashlib.sha256(text.encode()).hexdigest()
    ):
        raise SourceLineageError("structural segment differs")
    try:
        bytes.fromhex(segment["parent_text_sha256"])
        bytes.fromhex(segment["segment_text_sha256"])
    except ValueError as error:
        raise SourceLineageError("structural segment digest differs") from error
    return {
        "schema": SEGMENT_SCHEMA,
        "parent_row_id": segment["parent_row_id"],
        "parent_text_sha256": segment["parent_text_sha256"],
        "index": segment["index"],
        "count": segment["count"],
        "utf8_start": segment["utf8_start"],
        "utf8_end": segment["utf8_end"],
        "segment_text_sha256": segment["segment_text_sha256"],
    }


def source_row_id(source: Any, text: str) -> str:
    """Identify an unsplit upstream row or one exact child representation."""

    locator = source_locator(source)
    segment = structural_segment(source, text)
    return canonical_sha256(
        locator if segment is None else {"parent": locator, "segment": segment}
    )
