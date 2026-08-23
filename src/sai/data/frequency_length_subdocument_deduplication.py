"""Apply global frequency/length-aware exact subdocument deduplication."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import stat
import struct
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack
from decimal import ROUND_CEILING, Decimal, localcontext
from pathlib import Path
from typing import Any, BinaryIO

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import (
    canonical_sha256,
    normalize_document,
    sha256_file,
)

SCHEMA = "sai-frequency-length-subdocument-deduplication-receipt-v1"
TRANSFORM_SCHEMA = "sai-frequency-length-subdocument-transform-v1"
DEFAULT_SEGMENT_CHARACTERS = 32
DEFAULT_DELETE_CHARACTERS = 100
DEFAULT_REFERENCE_CHARACTERS = 512
DEFAULT_EFFECTIVE_SHARDS_NUMERATOR = 100
DEFAULT_EFFECTIVE_SHARDS_DENOMINATOR = 3
DEFAULT_CHUNK_RECORDS = 100_000
DEFAULT_MAXIMUM_LINE_BYTES = 16 << 20
DEFAULT_MAXIMUM_OPEN_CHUNKS = 128
RETENTION_POLICIES = {"adaptive_frequency_length", "keep_one_control"}

# hash, document identity, input, line offset/bytes, chunk, character span,
# normalized character length, code flag
_INDEX = struct.Struct(">32s32sIQQIQQQB")
# hash, global frequency, normalized character length, retention budget
_GROUP = struct.Struct(">32sQQQ")
# input, line offset, document identity/bytes, chunk, character span, group,
# frequency, budget, normalized character length
_CANDIDATE = struct.Struct(">IQ32sQIQQ32sQQQ")

_NATURAL_BOUNDARY = re.compile(r"\n[ \t]*\n+|\n|(?<=[.!?])[ \t]+")
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
_NUMBER = re.compile(r"(?<![\w])[-+]?\d+(?:[.,:/-]\d+)*(?![\w])")


class FrequencyLengthSubdocumentDeduplicationError(RuntimeError):
    """Input, retention geometry, external index, or reconstruction differs."""


def retention_budget(
    frequency: int,
    normalized_characters: int,
    *,
    reference_characters: int = DEFAULT_REFERENCE_CHARACTERS,
    effective_shards_numerator: int = DEFAULT_EFFECTIVE_SHARDS_NUMERATOR,
    effective_shards_denominator: int = DEFAULT_EFFECTIVE_SHARDS_DENOMINATOR,
) -> int:
    """Return the deterministic frequency/length-aware copy budget ``T(C,L)``."""

    integers = (
        frequency,
        normalized_characters,
        reference_characters,
        effective_shards_numerator,
        effective_shards_denominator,
    )
    if (
        any(not isinstance(value, int) or isinstance(value, bool) for value in integers)
        or frequency <= 0
        or normalized_characters < 0
        or reference_characters <= 0
        or effective_shards_denominator <= 0
        or effective_shards_numerator <= effective_shards_denominator
    ):
        raise FrequencyLengthSubdocumentDeduplicationError("retention geometry differs")
    with localcontext() as context:
        context.prec = 64
        base = Decimal(
            effective_shards_numerator - effective_shards_denominator
        ) / Decimal(effective_shards_numerator)
        expected = Decimal(frequency) * (base ** (frequency - 1))
        if expected <= 1:
            return 1
        alpha = max(
            Decimal(0),
            Decimal(1) - Decimal(normalized_characters) / Decimal(reference_characters),
        )
        raw_budget = Decimal(1) + (expected - Decimal(1)) * alpha
        budget = int(raw_budget.to_integral_value(rounding=ROUND_CEILING))
    return min(frequency, max(1, budget))


def _natural_chunks(text: str, base: int) -> list[dict[str, Any]]:
    chunks = []
    start = 0
    for match in _NATURAL_BOUNDARY.finditer(text):
        end = match.end()
        if end > start:
            chunks.append(
                {
                    "text": text[start:end],
                    "character_start": base + start,
                    "character_end": base + end,
                    "code": False,
                }
            )
        start = end
    if start < len(text):
        chunks.append(
            {
                "text": text[start:],
                "character_start": base + start,
                "character_end": base + len(text),
                "code": False,
            }
        )
    return chunks


def _fence_aware_chunks(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines(keepends=True)
    chunks = []
    natural_start = 0
    position = 0
    index = 0
    while index < len(lines):
        match = _FENCE.match(lines[index])
        if match is None:
            position += len(lines[index])
            index += 1
            continue
        if position > natural_start:
            chunks.extend(_natural_chunks(text[natural_start:position], natural_start))
        fence_start = position
        marker = match.group(1)
        marker_character = marker[0]
        marker_length = len(marker)
        position += len(lines[index])
        index += 1
        while index < len(lines):
            closing = _FENCE.match(lines[index])
            position += len(lines[index])
            index += 1
            if (
                closing is not None
                and closing.group(1)[0] == marker_character
                and len(closing.group(1)) >= marker_length
            ):
                break
        chunks.append(
            {
                "text": text[fence_start:position],
                "character_start": fence_start,
                "character_end": position,
                "code": True,
            }
        )
        natural_start = position
    if natural_start < len(text):
        chunks.extend(_natural_chunks(text[natural_start:], natural_start))
    return chunks


def segment_subdocuments(
    text: str,
    *,
    minimum_characters: int = DEFAULT_SEGMENT_CHARACTERS,
    code_document: bool = False,
) -> list[dict[str, Any]]:
    """Segment losslessly at natural boundaries and merge undersized units."""

    if (
        not isinstance(text, str)
        or not text
        or not isinstance(minimum_characters, int)
        or isinstance(minimum_characters, bool)
        or minimum_characters <= 0
        or not isinstance(code_document, bool)
    ):
        raise FrequencyLengthSubdocumentDeduplicationError(
            "segmentation geometry differs"
        )
    if code_document:
        initial = [
            {
                "text": text,
                "character_start": 0,
                "character_end": len(text),
                "code": True,
            }
        ]
    else:
        initial = _fence_aware_chunks(text)
    merged = []
    index = 0
    while index < len(initial):
        current = initial[index]
        if current["code"] or len(current["text"]) >= minimum_characters:
            merged.append(current)
            index += 1
            continue
        start = current["character_start"]
        end = current["character_end"]
        parts = [current["text"]]
        index += 1
        while (
            sum(len(part) for part in parts) < minimum_characters
            and index < len(initial)
            and initial[index]["code"] is False
        ):
            parts.append(initial[index]["text"])
            end = initial[index]["character_end"]
            index += 1
        merged.append(
            {
                "text": "".join(parts),
                "character_start": start,
                "character_end": end,
                "code": False,
            }
        )
    if not merged or "".join(chunk["text"] for chunk in merged) != text:
        raise FrequencyLengthSubdocumentDeduplicationError(
            "subdocument segmentation is not lossless"
        )
    return merged


def _normalized_chunk(text: str, *, code: bool) -> str:
    if code:
        return text
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _NUMBER.sub("<num>", normalized)
    return " ".join(normalized.split())


def _input_descriptors(paths: list[Path]) -> list[dict[str, Any]]:
    if not paths or len(paths) != len(set(paths)):
        raise FrequencyLengthSubdocumentDeduplicationError(
            "subdocument input set differs"
        )
    file_identities = set()
    descriptors = []
    for order, path in enumerate(paths):
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError as error:
            raise FrequencyLengthSubdocumentDeduplicationError(
                "subdocument input is missing or unsafe"
            ) from error
        try:
            metadata = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        file_identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size <= 0
            or file_identity in file_identities
        ):
            raise FrequencyLengthSubdocumentDeduplicationError(
                "subdocument input is missing, aliased, or unsafe"
            )
        file_identities.add(file_identity)
        descriptors.append(
            {
                "order": order,
                "path": str(path.resolve()),
                "bytes": metadata.st_size,
                "sha256": sha256_file(path),
            }
        )
    return descriptors


def _write_records(
    records: list[tuple[Any, ...]],
    record_struct: struct.Struct,
    path: Path,
) -> Path:
    records.sort()
    with path.open("xb") as handle:
        for record in records:
            handle.write(record_struct.pack(*record))
    return path


def _records(
    handle: BinaryIO, record_struct: struct.Struct
) -> Iterator[tuple[Any, ...]]:
    while encoded := handle.read(record_struct.size):
        if len(encoded) != record_struct.size:
            raise FrequencyLengthSubdocumentDeduplicationError(
                "subdocument temporary index is truncated"
            )
        yield record_struct.unpack(encoded)


def _reduce_runs(
    runs: list[Path],
    root: Path,
    record_struct: struct.Struct,
    *,
    prefix: str,
    maximum_open_chunks: int,
) -> tuple[list[Path], int]:
    passes = 0
    current = runs
    while len(current) > maximum_open_chunks:
        reduced = []
        for batch_index, start in enumerate(
            range(0, len(current), maximum_open_chunks)
        ):
            batch = current[start : start + maximum_open_chunks]
            output = root / f"{prefix}-merge-{passes:04d}-{batch_index:08d}.bin"
            with ExitStack() as stack, output.open("xb") as target:
                handles = [stack.enter_context(path.open("rb")) for path in batch]
                for record in heapq.merge(
                    *(_records(handle, record_struct) for handle in handles)
                ):
                    target.write(record_struct.pack(*record))
            reduced.append(output)
        for path in current:
            path.unlink()
        current = reduced
        passes += 1
    return current, passes


def _merged_records(
    paths: list[Path], record_struct: struct.Struct
) -> Iterator[tuple[Any, ...]]:
    with ExitStack() as stack:
        handles = [stack.enter_context(path.open("rb")) for path in paths]
        yield from heapq.merge(*(_records(handle, record_struct) for handle in handles))


def _build_index(
    source_paths: list[Path],
    root: Path,
    *,
    minimum_characters: int,
    chunk_records: int,
    maximum_line_bytes: int,
) -> tuple[list[Path], Counter[str]]:
    records = []
    runs = []
    counts: Counter[str] = Counter()
    for input_index, path in enumerate(source_paths):
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline(maximum_line_bytes + 1)
                if not line:
                    break
                if len(line) > maximum_line_bytes:
                    raise FrequencyLengthSubdocumentDeduplicationError(
                        "subdocument input line exceeds the frozen cap"
                    )
                if not line.strip():
                    counts["blank_lines"] += 1
                    continue
                try:
                    document = normalize_document(json.loads(line))
                except (json.JSONDecodeError, RuntimeError) as error:
                    raise FrequencyLengthSubdocumentDeduplicationError(
                        "subdocument input row differs"
                    ) from error
                chunks = segment_subdocuments(
                    document["text"],
                    minimum_characters=minimum_characters,
                    code_document=document["source"]["domain"] == "code",
                )
                counts["documents"] += 1
                counts["input_characters"] += len(document["text"])
                counts["input_utf8_bytes"] += len(document["text"].encode())
                counts["chunks"] += len(chunks)
                for chunk_index, chunk in enumerate(chunks):
                    normalized = _normalized_chunk(chunk["text"], code=chunk["code"])
                    if not normalized:
                        counts["unindexed_empty_normalized_chunks"] += 1
                        continue
                    records.append(
                        (
                            hashlib.sha256(normalized.encode()).digest(),
                            bytes.fromhex(document["identity_sha256"]),
                            input_index,
                            offset,
                            len(line),
                            chunk_index,
                            chunk["character_start"],
                            chunk["character_end"],
                            len(normalized),
                            chunk["code"],
                        )
                    )
                    counts["indexed_chunks"] += 1
                    counts["code_chunks"] += chunk["code"]
                    if len(records) == chunk_records:
                        path = root / f"index-{len(runs):08d}.bin"
                        runs.append(_write_records(records, _INDEX, path))
                        records = []
    if records:
        path = root / f"index-{len(runs):08d}.bin"
        runs.append(_write_records(records, _INDEX, path))
    if not counts["documents"] or not runs:
        raise FrequencyLengthSubdocumentDeduplicationError("subdocument input is empty")
    return runs, counts


def _load_indexed_chunk(
    handles: list[BinaryIO], record: tuple[Any, ...], minimum_characters: int
) -> str:
    (
        expected_hash,
        expected_identity,
        input_index,
        offset,
        line_bytes,
        chunk_index,
        character_start,
        character_end,
        normalized_characters,
        code,
    ) = record
    if input_index >= len(handles) or line_bytes <= 0:
        raise FrequencyLengthSubdocumentDeduplicationError(
            "subdocument index locator differs"
        )
    handle = handles[input_index]
    handle.seek(offset)
    encoded = handle.read(line_bytes)
    if len(encoded) != line_bytes:
        raise FrequencyLengthSubdocumentDeduplicationError(
            "subdocument source replay differs"
        )
    try:
        document = normalize_document(json.loads(encoded))
    except (json.JSONDecodeError, RuntimeError) as error:
        raise FrequencyLengthSubdocumentDeduplicationError(
            "subdocument source replay differs"
        ) from error
    chunks = segment_subdocuments(
        document["text"],
        minimum_characters=minimum_characters,
        code_document=document["source"]["domain"] == "code",
    )
    if bytes.fromhex(
        document["identity_sha256"]
    ) != expected_identity or chunk_index >= len(chunks):
        raise FrequencyLengthSubdocumentDeduplicationError(
            "subdocument source identity differs"
        )
    chunk = chunks[chunk_index]
    normalized = _normalized_chunk(chunk["text"], code=chunk["code"])
    if (
        chunk["character_start"] != character_start
        or chunk["character_end"] != character_end
        or chunk["code"] != bool(code)
        or len(normalized) != normalized_characters
        or hashlib.sha256(normalized.encode()).digest() != expected_hash
    ):
        raise FrequencyLengthSubdocumentDeduplicationError(
            "subdocument indexed chunk differs"
        )
    return normalized


def _build_groups(
    index_runs: list[Path],
    source_paths: list[Path],
    group_path: Path,
    *,
    minimum_characters: int,
    reference_characters: int,
    effective_shards_numerator: int,
    effective_shards_denominator: int,
    retention_policy: str,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    with ExitStack() as stack, group_path.open("xb") as groups:
        handles = [stack.enter_context(path.open("rb")) for path in source_paths]
        group_hash = None
        group_normalized = None
        group_frequency = 0
        group_length = 0

        def flush() -> None:
            nonlocal group_hash, group_frequency, group_length
            if group_hash is None:
                return
            budget = (
                1
                if retention_policy == "keep_one_control"
                else retention_budget(
                    group_frequency,
                    group_length,
                    reference_characters=reference_characters,
                    effective_shards_numerator=effective_shards_numerator,
                    effective_shards_denominator=effective_shards_denominator,
                )
            )
            groups.write(_GROUP.pack(group_hash, group_frequency, group_length, budget))
            counts["groups"] += 1
            counts["duplicate_groups"] += group_frequency > 1
            counts["duplicate_occurrences"] += max(0, group_frequency - 1)

        for record in _merged_records(index_runs, _INDEX):
            normalized = _load_indexed_chunk(handles, record, minimum_characters)
            if record[0] != group_hash:
                flush()
                group_hash = record[0]
                group_normalized = normalized
                group_frequency = 1
                group_length = len(normalized)
            else:
                if normalized != group_normalized or len(normalized) != group_length:
                    raise FrequencyLengthSubdocumentDeduplicationError(
                        "normalized subdocument SHA-256 collision differs"
                    )
                group_frequency += 1
        flush()
    return counts


def _build_candidate_runs(
    index_runs: list[Path],
    group_path: Path,
    root: Path,
    *,
    chunk_records: int,
) -> tuple[list[Path], int]:
    records = []
    runs = []
    candidates = 0
    with group_path.open("rb") as group_handle:
        group_iterator = _records(group_handle, _GROUP)
        group = next(group_iterator, None)
        active_hash = None
        rank = 0
        boundary_document = None
        for record in _merged_records(index_runs, _INDEX):
            while group is not None and group[0] < record[0]:
                group = next(group_iterator, None)
            if group is None or group[0] != record[0]:
                raise FrequencyLengthSubdocumentDeduplicationError(
                    "subdocument group join differs"
                )
            if active_hash != record[0]:
                active_hash = record[0]
                rank = 0
                boundary_document = None
            rank += 1
            budget = group[3]
            if rank <= budget:
                boundary_document = record[1]
                continue
            if record[1] == boundary_document:
                continue
            records.append(
                (
                    record[2],
                    record[3],
                    record[1],
                    record[4],
                    record[5],
                    record[6],
                    record[7],
                    record[0],
                    group[1],
                    budget,
                    group[2],
                )
            )
            candidates += 1
            if len(records) == chunk_records:
                path = root / f"candidate-{len(runs):08d}.bin"
                runs.append(_write_records(records, _CANDIDATE, path))
                records = []
        if next(group_iterator, None) is not None:
            raise FrequencyLengthSubdocumentDeduplicationError(
                "subdocument group population differs"
            )
    if records:
        path = root / f"candidate-{len(runs):08d}.bin"
        runs.append(_write_records(records, _CANDIDATE, path))
    return runs, candidates


def _transformed_document(document: dict[str, Any], text: str) -> dict[str, Any]:
    payload = {
        key: value for key, value in document.items() if key != "identity_sha256"
    }
    payload["text"] = text
    return normalize_document(payload)


def _reconstruct(
    source_paths: list[Path],
    candidate_runs: list[Path],
    output_path: Path,
    manifest_path: Path,
    *,
    minimum_characters: int,
    delete_characters: int,
    maximum_line_bytes: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    decisions = iter(_merged_records(candidate_runs, _CANDIDATE))
    decision = next(decisions, None)
    with output_path.open("x") as output, manifest_path.open("x") as manifest:
        for input_index, path in enumerate(source_paths):
            with path.open("rb") as source:
                while True:
                    offset = source.tell()
                    line = source.readline(maximum_line_bytes + 1)
                    if not line:
                        break
                    if len(line) > maximum_line_bytes:
                        raise FrequencyLengthSubdocumentDeduplicationError(
                            "subdocument input line exceeds the frozen cap"
                        )
                    if not line.strip():
                        continue
                    try:
                        document = normalize_document(json.loads(line))
                    except (json.JSONDecodeError, RuntimeError) as error:
                        raise FrequencyLengthSubdocumentDeduplicationError(
                            "subdocument reconstruction row differs"
                        ) from error
                    document_decisions = []
                    while (
                        decision is not None
                        and decision[0] == input_index
                        and decision[1] == offset
                    ):
                        document_decisions.append(decision)
                        decision = next(decisions, None)
                    if not document_decisions:
                        output.write(
                            json.dumps(document, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                        counts["output_documents"] += 1
                        counts["output_characters"] += len(document["text"])
                        counts["output_utf8_bytes"] += len(document["text"].encode())
                        continue
                    chunks = segment_subdocuments(
                        document["text"],
                        minimum_characters=minimum_characters,
                        code_document=document["source"]["domain"] == "code",
                    )
                    candidate_by_index = {}
                    for row in document_decisions:
                        chunk_index = row[4]
                        if (
                            row[2] != bytes.fromhex(document["identity_sha256"])
                            or row[3] != len(line)
                            or chunk_index >= len(chunks)
                            or chunk_index in candidate_by_index
                        ):
                            raise FrequencyLengthSubdocumentDeduplicationError(
                                "subdocument reconstruction decision differs"
                            )
                        chunk = chunks[chunk_index]
                        normalized = _normalized_chunk(
                            chunk["text"], code=chunk["code"]
                        )
                        if (
                            chunk["character_start"] != row[5]
                            or chunk["character_end"] != row[6]
                            or hashlib.sha256(normalized.encode()).digest() != row[7]
                            or len(normalized) != row[10]
                        ):
                            raise FrequencyLengthSubdocumentDeduplicationError(
                                "subdocument reconstruction chunk differs"
                            )
                        candidate_by_index[chunk_index] = row
                    deleted = set()
                    restored = set()
                    ordered_candidates = sorted(candidate_by_index)
                    start = 0
                    while start < len(ordered_candidates):
                        end = start + 1
                        while (
                            end < len(ordered_candidates)
                            and ordered_candidates[end]
                            == ordered_candidates[end - 1] + 1
                        ):
                            end += 1
                        run = ordered_candidates[start:end]
                        run_characters = sum(
                            len(chunks[index]["text"]) for index in run
                        )
                        (
                            deleted if run_characters >= delete_characters else restored
                        ).update(run)
                        start = end
                    output_text = "".join(
                        chunk["text"]
                        for index, chunk in enumerate(chunks)
                        if index not in deleted
                    )
                    output_document = (
                        _transformed_document(document, output_text)
                        if output_text
                        else None
                    )
                    if output_document is not None:
                        output.write(
                            json.dumps(
                                output_document, ensure_ascii=False, sort_keys=True
                            )
                            + "\n"
                        )
                        counts["output_documents"] += 1
                        counts["output_characters"] += len(output_text)
                        counts["output_utf8_bytes"] += len(output_text.encode())
                    else:
                        counts["documents_fully_deleted"] += 1
                    counts["documents_with_candidate_chunks"] += 1
                    counts["documents_modified"] += bool(deleted)
                    counts["candidate_chunks"] += len(candidate_by_index)
                    counts["deleted_chunks"] += len(deleted)
                    counts["coherence_restored_chunks"] += len(restored)
                    deleted_characters = sum(
                        len(chunks[index]["text"]) for index in deleted
                    )
                    counts["deleted_characters"] += deleted_characters
                    counts["deleted_utf8_bytes"] += sum(
                        len(chunks[index]["text"].encode()) for index in deleted
                    )
                    transform = {
                        "schema": TRANSFORM_SCHEMA,
                        "parent_identity_sha256": document["identity_sha256"],
                        "parent_text_sha256": hashlib.sha256(
                            document["text"].encode()
                        ).hexdigest(),
                        "output_identity_sha256": (
                            output_document["identity_sha256"]
                            if output_document is not None
                            else None
                        ),
                        "output_text_sha256": (
                            hashlib.sha256(output_text.encode()).hexdigest()
                            if output_text
                            else None
                        ),
                        "candidate_chunks": len(candidate_by_index),
                        "deleted_chunks": len(deleted),
                        "coherence_restored_chunks": len(restored),
                        "deleted_characters": deleted_characters,
                        "decisions": [
                            {
                                "chunk_index": index,
                                "character_start": chunks[index]["character_start"],
                                "character_end": chunks[index]["character_end"],
                                "normalized_text_sha256": candidate_by_index[index][
                                    7
                                ].hex(),
                                "global_frequency": candidate_by_index[index][8],
                                "initial_retention_budget": candidate_by_index[index][
                                    9
                                ],
                                "normalized_characters": candidate_by_index[index][10],
                                "outcome": (
                                    "deleted"
                                    if index in deleted
                                    else "retained_for_coherence"
                                ),
                            }
                            for index in ordered_candidates
                        ],
                        "contains_source_text": False,
                    }
                    transform["record_sha256"] = canonical_sha256(transform)
                    manifest.write(json.dumps(transform, sort_keys=True) + "\n")
    if decision is not None:
        raise FrequencyLengthSubdocumentDeduplicationError(
            "subdocument candidate population differs"
        )
    return counts


def build_frequency_length_deduplication(
    source_paths: list[Path],
    output_path: Path,
    transform_manifest_path: Path,
    receipt_path: Path,
    *,
    minimum_characters: int = DEFAULT_SEGMENT_CHARACTERS,
    delete_characters: int = DEFAULT_DELETE_CHARACTERS,
    reference_characters: int = DEFAULT_REFERENCE_CHARACTERS,
    effective_shards_numerator: int = DEFAULT_EFFECTIVE_SHARDS_NUMERATOR,
    effective_shards_denominator: int = DEFAULT_EFFECTIVE_SHARDS_DENOMINATOR,
    chunk_records: int = DEFAULT_CHUNK_RECORDS,
    maximum_line_bytes: int = DEFAULT_MAXIMUM_LINE_BYTES,
    maximum_open_chunks: int = DEFAULT_MAXIMUM_OPEN_CHUNKS,
    temporary_root: Path | None = None,
    retention_policy: str = "adaptive_frequency_length",
) -> dict[str, Any]:
    """Count global subdocuments and apply coherent adaptive copy retention."""

    output_paths = (output_path, transform_manifest_path, receipt_path)
    integers = (
        minimum_characters,
        delete_characters,
        reference_characters,
        effective_shards_numerator,
        effective_shards_denominator,
        chunk_records,
        maximum_line_bytes,
        maximum_open_chunks,
    )
    if (
        any(path.exists() or path.is_symlink() for path in output_paths)
        or any(
            not isinstance(value, int) or isinstance(value, bool) for value in integers
        )
        or minimum_characters <= 0
        or delete_characters <= 0
        or reference_characters <= 0
        or effective_shards_denominator <= 0
        or effective_shards_numerator <= effective_shards_denominator
        or chunk_records <= 0
        or maximum_line_bytes <= 0
        or not 2 <= maximum_open_chunks <= 1_024
        or not isinstance(retention_policy, str)
        or retention_policy not in RETENTION_POLICIES
        or (
            temporary_root is not None
            and (not temporary_root.is_dir() or temporary_root.is_symlink())
        )
    ):
        raise FrequencyLengthSubdocumentDeduplicationError(
            "subdocument deduplication geometry or output differs"
        )
    inputs = _input_descriptors(source_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transform_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_stage = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    manifest_stage = transform_manifest_path.with_name(
        f".{transform_manifest_path.name}.tmp.{os.getpid()}"
    )
    initial_runs = merge_passes = final_runs = candidate_initial_runs = 0
    candidate_merge_passes = candidate_final_runs = 0
    try:
        with tempfile.TemporaryDirectory(
            prefix="sai-subdocument-dedup-", dir=temporary_root
        ) as directory:
            scratch = Path(directory)
            index_runs, scan = _build_index(
                source_paths,
                scratch,
                minimum_characters=minimum_characters,
                chunk_records=chunk_records,
                maximum_line_bytes=maximum_line_bytes,
            )
            initial_runs = len(index_runs)
            index_runs, merge_passes = _reduce_runs(
                index_runs,
                scratch,
                _INDEX,
                prefix="index",
                maximum_open_chunks=maximum_open_chunks,
            )
            final_runs = len(index_runs)
            group_path = scratch / "groups.bin"
            groups = _build_groups(
                index_runs,
                source_paths,
                group_path,
                minimum_characters=minimum_characters,
                reference_characters=reference_characters,
                effective_shards_numerator=effective_shards_numerator,
                effective_shards_denominator=effective_shards_denominator,
                retention_policy=retention_policy,
            )
            candidate_runs, initial_candidates = _build_candidate_runs(
                index_runs,
                group_path,
                scratch,
                chunk_records=chunk_records,
            )
            candidate_initial_runs = len(candidate_runs)
            if candidate_runs:
                candidate_runs, candidate_merge_passes = _reduce_runs(
                    candidate_runs,
                    scratch,
                    _CANDIDATE,
                    prefix="candidate",
                    maximum_open_chunks=maximum_open_chunks,
                )
                candidate_final_runs = len(candidate_runs)
                reconstruction = _reconstruct(
                    source_paths,
                    candidate_runs,
                    output_stage,
                    manifest_stage,
                    minimum_characters=minimum_characters,
                    delete_characters=delete_characters,
                    maximum_line_bytes=maximum_line_bytes,
                )
            else:
                output_stage.touch(exist_ok=False)
                manifest_stage.touch(exist_ok=False)
                reconstruction = Counter()
                with output_stage.open("w") as output:
                    for path in source_paths:
                        with path.open() as source:
                            for line in source:
                                if not line.strip():
                                    continue
                                document = normalize_document(json.loads(line))
                                output.write(
                                    json.dumps(
                                        document, ensure_ascii=False, sort_keys=True
                                    )
                                    + "\n"
                                )
                                reconstruction["output_documents"] += 1
                                reconstruction["output_characters"] += len(
                                    document["text"]
                                )
                                reconstruction["output_utf8_bytes"] += len(
                                    document["text"].encode()
                                )
            if reconstruction["candidate_chunks"] != initial_candidates:
                raise FrequencyLengthSubdocumentDeduplicationError(
                    "subdocument candidate custody differs"
                )
        if any(
            sha256_file(path) != descriptor["sha256"]
            for path, descriptor in zip(source_paths, inputs, strict=True)
        ):
            raise FrequencyLengthSubdocumentDeduplicationError(
                "subdocument input changed during replay"
            )
        os.replace(output_stage, output_path)
        os.replace(manifest_stage, transform_manifest_path)
    except BaseException:
        output_stage.unlink(missing_ok=True)
        manifest_stage.unlink(missing_ok=True)
        raise
    counts = Counter(scan)
    counts.update(groups)
    counts.update(reconstruction)
    counts["initial_candidate_chunks"] = initial_candidates
    payload = {
        "schema": SCHEMA,
        "status": "complete_subdocument_deduplication_control",
        "inputs": inputs,
        "policy": {
            "paper": "arxiv:2608.03089v1",
            "retention_policy": retention_policy,
            "segmentation": "natural_boundaries_with_short_forward_merge",
            "minimum_segment_characters": minimum_characters,
            "numeric_normalization": (
                "natural_language_numeric_expressions_to_placeholder"
            ),
            "code_normalization": "identity",
            "code_document_policy": "whole_document_indivisible_fail_closed",
            "markdown_fenced_code_policy": "indivisible",
            "effective_shards": {
                "numerator": effective_shards_numerator,
                "denominator": effective_shards_denominator,
            },
            "reference_characters": reference_characters,
            "minimum_contiguous_delete_characters": delete_characters,
            "retention_function": (
                "keep_exactly_one_initial_occurrence"
                if retention_policy == "keep_one_control"
                else (
                    "ceil(1+(C*(1-1/N)^(C-1)-1)*max(0,1-L/L0)); "
                    "clamped_to_[1,C]; one_when_frequency_budget_leq_one"
                )
            ),
            "occurrence_order": "document_identity_sha256_then_chunk_index",
            "boundary_document_policy": (
                "retain_all_occurrences_sharing_boundary_document"
            ),
            "deletion_policy": (
                "delete_only_maximal_contiguous_candidate_runs_at_or_above_threshold"
            ),
        },
        "counts": dict(sorted(counts.items())),
        "index": {
            "record_bytes": _INDEX.size,
            "initial_runs": initial_runs,
            "merge_passes": merge_passes,
            "final_runs": final_runs,
            "candidate_record_bytes": _CANDIDATE.size,
            "candidate_initial_runs": candidate_initial_runs,
            "candidate_merge_passes": candidate_merge_passes,
            "candidate_final_runs": candidate_final_runs,
            "maximum_records_per_run": chunk_records,
            "maximum_open_runs": maximum_open_chunks,
            "temporary_indexes_removed": True,
            "source_text_persisted_in_indexes": False,
        },
        "output": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "transform_manifest": {
            "path": str(transform_manifest_path.resolve()),
            "bytes": transform_manifest_path.stat().st_size,
            "sha256": sha256_file(transform_manifest_path),
            "records": counts["documents_with_candidate_chunks"],
            "contains_source_text": False,
        },
        "global_normalized_exact_subdocument_counting_complete": True,
        "frequency_length_retention_complete": (
            retention_policy == "adaptive_frequency_length"
        ),
        "keep_one_control_complete": retention_policy == "keep_one_control",
        "coherence_aware_deletion_complete": True,
        "semantic_near_duplicate_filtering_complete": False,
        "matched_unchanged_keep_one_and_adaptive_evaluation_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["policy_sha256"] = canonical_sha256(payload["policy"])
    payload["receipt_sha256"] = canonical_sha256(payload)
    try:
        _atomic_create(receipt_path, payload)
    except BaseException:
        output_path.unlink(missing_ok=True)
        transform_manifest_path.unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transform-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--minimum-segment-characters",
        type=int,
        default=DEFAULT_SEGMENT_CHARACTERS,
    )
    parser.add_argument(
        "--minimum-delete-characters", type=int, default=DEFAULT_DELETE_CHARACTERS
    )
    parser.add_argument(
        "--reference-characters", type=int, default=DEFAULT_REFERENCE_CHARACTERS
    )
    parser.add_argument(
        "--effective-shards-numerator",
        type=int,
        default=DEFAULT_EFFECTIVE_SHARDS_NUMERATOR,
    )
    parser.add_argument(
        "--effective-shards-denominator",
        type=int,
        default=DEFAULT_EFFECTIVE_SHARDS_DENOMINATOR,
    )
    parser.add_argument("--chunk-records", type=int, default=DEFAULT_CHUNK_RECORDS)
    parser.add_argument(
        "--maximum-line-bytes", type=int, default=DEFAULT_MAXIMUM_LINE_BYTES
    )
    parser.add_argument(
        "--maximum-open-chunks", type=int, default=DEFAULT_MAXIMUM_OPEN_CHUNKS
    )
    parser.add_argument("--temporary-root", type=Path)
    parser.add_argument(
        "--retention-policy",
        choices=sorted(RETENTION_POLICIES),
        default="adaptive_frequency_length",
    )
    args = parser.parse_args()
    result = build_frequency_length_deduplication(
        args.source,
        args.output,
        args.transform_manifest,
        args.receipt,
        minimum_characters=args.minimum_segment_characters,
        delete_characters=args.minimum_delete_characters,
        reference_characters=args.reference_characters,
        effective_shards_numerator=args.effective_shards_numerator,
        effective_shards_denominator=args.effective_shards_denominator,
        chunk_records=args.chunk_records,
        maximum_line_bytes=args.maximum_line_bytes,
        maximum_open_chunks=args.maximum_open_chunks,
        temporary_root=args.temporary_root,
        retention_policy=args.retention_policy,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
