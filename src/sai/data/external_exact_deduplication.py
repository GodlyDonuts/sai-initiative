"""Globally deduplicate admitted documents with a bounded external-memory index."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import stat
import struct
import tempfile
import unicodedata
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any, BinaryIO

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-global-normalized-exact-deduplication-receipt-v1"
DROP_SCHEMA = "sai-global-normalized-exact-duplicate-drop-v1"
POLICY = {
    "normalization": "NFKC_casefold_whitespace_collapse",
    "group_key": "sha256_normalized_utf8_with_full_text_collision_replay",
    "survivor": "minimum_document_identity_sha256",
    "output_order": "normalized_text_sha256_then_document_identity_sha256",
}
_RECORD = struct.Struct(">32s32sIQQ")
DEFAULT_CHUNK_RECORDS = 100_000
DEFAULT_MAXIMUM_LINE_BYTES = 16 << 20
DEFAULT_MAXIMUM_OPEN_CHUNKS = 128
MAXIMUM_INPUTS = (1 << 32) - 1


class ExternalExactDeduplicationError(RuntimeError):
    """Input identity, index geometry, collision replay, or output differs."""


def _normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _inputs(paths: list[Path]) -> list[dict[str, Any]]:
    if not paths or len(paths) > MAXIMUM_INPUTS or len(paths) != len(set(paths)):
        raise ExternalExactDeduplicationError("deduplication input set differs")
    identities = set()
    result = []
    for order, path in enumerate(paths):
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError as error:
            raise ExternalExactDeduplicationError(
                "deduplication input is missing or unsafe"
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
            or file_identity in identities
        ):
            raise ExternalExactDeduplicationError(
                "deduplication input is missing, aliased, or unsafe"
            )
        identities.add(file_identity)
        result.append(
            {
                "order": order,
                "path": str(path.resolve()),
                "bytes": metadata.st_size,
                "sha256": sha256_file(path),
            }
        )
    return result


def _write_chunk(
    records: list[tuple[bytes, bytes, int, int, int]],
    root: Path,
    index: int,
) -> Path:
    records.sort()
    path = root / f"chunk-{index:08d}.bin"
    with path.open("xb") as handle:
        for record in records:
            handle.write(_RECORD.pack(*record))
    return path


def _build_index(
    paths: list[Path],
    root: Path,
    *,
    chunk_records: int,
    maximum_line_bytes: int,
) -> tuple[list[Path], dict[str, int]]:
    records: list[tuple[bytes, bytes, int, int, int]] = []
    chunks = []
    rows = blank_lines = 0
    for input_index, path in enumerate(paths):
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline(maximum_line_bytes + 1)
                if not line:
                    break
                if len(line) > maximum_line_bytes:
                    raise ExternalExactDeduplicationError(
                        "deduplication input line exceeds the frozen cap"
                    )
                if not line.strip():
                    blank_lines += 1
                    continue
                try:
                    document = normalize_document(json.loads(line))
                except (json.JSONDecodeError, RuntimeError) as error:
                    raise ExternalExactDeduplicationError(
                        "deduplication input row differs"
                    ) from error
                normalized_sha256 = hashlib.sha256(
                    (normalized := _normalized_text(document["text"])).encode()
                ).digest()
                if not normalized:
                    raise ExternalExactDeduplicationError(
                        "deduplication normalized text is empty"
                    )
                records.append(
                    (
                        normalized_sha256,
                        bytes.fromhex(document["identity_sha256"]),
                        input_index,
                        offset,
                        len(line),
                    )
                )
                rows += 1
                if len(records) == chunk_records:
                    chunks.append(_write_chunk(records, root, len(chunks)))
                    records = []
    if records:
        chunks.append(_write_chunk(records, root, len(chunks)))
    if not rows or not chunks:
        raise ExternalExactDeduplicationError("deduplication input is empty")
    return chunks, {"documents": rows, "blank_lines": blank_lines}


def _chunk_records(handle: BinaryIO) -> Iterator[tuple[bytes, bytes, int, int, int]]:
    while encoded := handle.read(_RECORD.size):
        if len(encoded) != _RECORD.size:
            raise ExternalExactDeduplicationError(
                "deduplication temporary index is truncated"
            )
        yield _RECORD.unpack(encoded)


def _reduce_chunks(
    chunks: list[Path], root: Path, *, maximum_open_chunks: int
) -> tuple[list[Path], int]:
    """Merge index runs in bounded fan-in passes before the document replay."""

    passes = 0
    current = chunks
    while len(current) > maximum_open_chunks:
        reduced = []
        for batch_index, start in enumerate(
            range(0, len(current), maximum_open_chunks)
        ):
            batch = current[start : start + maximum_open_chunks]
            output = root / f"merge-{passes:04d}-{batch_index:08d}.bin"
            with ExitStack() as stack, output.open("xb") as target:
                handles = [stack.enter_context(path.open("rb")) for path in batch]
                for record in heapq.merge(
                    *(_chunk_records(handle) for handle in handles)
                ):
                    target.write(_RECORD.pack(*record))
            reduced.append(output)
        for path in current:
            path.unlink()
        current = reduced
        passes += 1
    return current, passes


def _load_document(
    handles: list[BinaryIO], record: tuple[bytes, bytes, int, int, int]
) -> tuple[dict[str, Any], str]:
    normalized_sha256, identity, input_index, offset, length = record
    if input_index >= len(handles) or length <= 0:
        raise ExternalExactDeduplicationError("deduplication index locator differs")
    handle = handles[input_index]
    handle.seek(offset)
    encoded = handle.read(length)
    if len(encoded) != length:
        raise ExternalExactDeduplicationError("deduplication source replay differs")
    try:
        document = normalize_document(json.loads(encoded))
    except (json.JSONDecodeError, RuntimeError) as error:
        raise ExternalExactDeduplicationError(
            "deduplication source replay differs"
        ) from error
    normalized = _normalized_text(document["text"])
    if (
        not normalized
        or bytes.fromhex(document["identity_sha256"]) != identity
        or hashlib.sha256(normalized.encode()).digest() != normalized_sha256
    ):
        raise ExternalExactDeduplicationError(
            "deduplication indexed source identity differs"
        )
    return document, normalized


def _location(record: tuple[bytes, bytes, int, int, int]) -> dict[str, int]:
    return {
        "input_order": record[2],
        "byte_offset": record[3],
        "line_bytes": record[4],
    }


def _write_duplicate(
    survivor_record: tuple[bytes, bytes, int, int, int],
    survivor_id: str,
    survivor_normalized: str,
    dropped_record: tuple[bytes, bytes, int, int, int],
    handles: list[BinaryIO],
    drops: Any,
    drop_identity: Any,
) -> None:
    dropped, dropped_normalized = _load_document(handles, dropped_record)
    if dropped_normalized != survivor_normalized:
        raise ExternalExactDeduplicationError(
            "normalized SHA-256 collision replay differs"
        )
    mapping = {
        "schema": DROP_SCHEMA,
        "normalized_text_sha256": survivor_record[0].hex(),
        "survivor_identity_sha256": survivor_id,
        "survivor_location": _location(survivor_record),
        "dropped_identity_sha256": dropped["identity_sha256"],
        "dropped_location": _location(dropped_record),
    }
    mapping["record_sha256"] = canonical_sha256(mapping)
    drops.write(json.dumps(mapping, sort_keys=True) + "\n")
    drop_identity.update(bytes.fromhex(mapping["record_sha256"]))


def build_exact_deduplication(
    source_paths: list[Path],
    output_path: Path,
    duplicate_manifest_path: Path,
    receipt_path: Path,
    *,
    chunk_records: int = DEFAULT_CHUNK_RECORDS,
    maximum_line_bytes: int = DEFAULT_MAXIMUM_LINE_BYTES,
    maximum_open_chunks: int = DEFAULT_MAXIMUM_OPEN_CHUNKS,
    temporary_root: Path | None = None,
) -> dict[str, Any]:
    """External-sort normalized identities and emit deterministic global survivors."""

    if (
        any(
            path.exists() or path.is_symlink()
            for path in (output_path, duplicate_manifest_path, receipt_path)
        )
        or isinstance(chunk_records, bool)
        or not isinstance(chunk_records, int)
        or chunk_records <= 0
        or isinstance(maximum_line_bytes, bool)
        or not isinstance(maximum_line_bytes, int)
        or maximum_line_bytes <= 0
        or isinstance(maximum_open_chunks, bool)
        or not isinstance(maximum_open_chunks, int)
        or not 2 <= maximum_open_chunks <= 1_024
        or (
            temporary_root is not None
            and (not temporary_root.is_dir() or temporary_root.is_symlink())
        )
    ):
        raise ExternalExactDeduplicationError(
            "deduplication geometry or output differs"
        )
    inputs = _inputs(source_paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duplicate_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_stage = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    duplicate_stage = duplicate_manifest_path.with_name(
        f".{duplicate_manifest_path.name}.tmp.{os.getpid()}"
    )
    survivors = drops = duplicate_groups = 0
    survivor_identity = hashlib.sha256()
    drop_identity = hashlib.sha256()
    initial_chunk_count = final_chunk_count = merge_passes = 0
    try:
        with tempfile.TemporaryDirectory(
            prefix="sai-global-exact-dedup-",
            dir=temporary_root,
        ) as temporary:
            chunks, scan = _build_index(
                source_paths,
                Path(temporary),
                chunk_records=chunk_records,
                maximum_line_bytes=maximum_line_bytes,
            )
            initial_chunk_count = len(chunks)
            chunks, merge_passes = _reduce_chunks(
                chunks,
                Path(temporary),
                maximum_open_chunks=maximum_open_chunks,
            )
            final_chunk_count = len(chunks)
            if any(
                sha256_file(path) != descriptor["sha256"]
                for path, descriptor in zip(source_paths, inputs, strict=True)
            ):
                raise ExternalExactDeduplicationError(
                    "deduplication input changed during indexing"
                )
            with ExitStack() as stack:
                chunk_handles = [
                    stack.enter_context(path.open("rb")) for path in chunks
                ]
                input_handles = [
                    stack.enter_context(path.open("rb")) for path in source_paths
                ]
                merged = heapq.merge(
                    *(_chunk_records(handle) for handle in chunk_handles)
                )
                with (
                    output_stage.open("x") as output,
                    duplicate_stage.open("x") as duplicate_manifest,
                ):
                    survivor_record = None
                    survivor_id = None
                    survivor_normalized = None
                    group_key = None
                    group_members = 0
                    for record in merged:
                        if group_key != record[0]:
                            duplicate_groups += group_members > 1
                            survivor_record = record
                            survivor, survivor_normalized = _load_document(
                                input_handles, survivor_record
                            )
                            survivor_id = survivor["identity_sha256"]
                            output.write(
                                json.dumps(survivor, ensure_ascii=False, sort_keys=True)
                                + "\n"
                            )
                            survivor_identity.update(bytes.fromhex(survivor_id))
                            survivors += 1
                            group_key = record[0]
                            group_members = 1
                            continue
                        if (
                            survivor_record is None
                            or survivor_id is None
                            or survivor_normalized is None
                        ):
                            raise ExternalExactDeduplicationError(
                                "deduplication group state differs"
                            )
                        _write_duplicate(
                            survivor_record,
                            survivor_id,
                            survivor_normalized,
                            record,
                            input_handles,
                            duplicate_manifest,
                            drop_identity,
                        )
                        drops += 1
                        group_members += 1
                    duplicate_groups += group_members > 1
            if survivors + drops != scan["documents"]:
                raise ExternalExactDeduplicationError(
                    "deduplication document custody differs"
                )
        if any(
            sha256_file(path) != descriptor["sha256"]
            for path, descriptor in zip(source_paths, inputs, strict=True)
        ):
            raise ExternalExactDeduplicationError(
                "deduplication input changed during replay"
            )
        os.replace(output_stage, output_path)
        os.replace(duplicate_stage, duplicate_manifest_path)
    except BaseException:
        output_stage.unlink(missing_ok=True)
        duplicate_stage.unlink(missing_ok=True)
        raise
    payload = {
        "schema": SCHEMA,
        "status": "complete_global_normalized_exact_deduplication",
        "inputs": inputs,
        "policy": POLICY,
        "policy_sha256": canonical_sha256(POLICY),
        "index": {
            "record_encoding": "big_endian_32s_32s_uint32_uint64_uint64",
            "record_bytes": _RECORD.size,
            "maximum_records_per_chunk": chunk_records,
            "initial_chunk_count": initial_chunk_count,
            "maximum_open_chunks": maximum_open_chunks,
            "merge_passes": merge_passes,
            "final_merge_chunk_count": final_chunk_count,
            "temporary_index_removed": True,
            "source_text_persisted_in_index": False,
        },
        "counts": {
            **scan,
            "survivors": survivors,
            "duplicates_dropped": drops,
            "duplicate_groups": duplicate_groups,
        },
        "output": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "ordered_identity_sha256": survivor_identity.hexdigest(),
        },
        "duplicate_manifest": {
            "path": str(duplicate_manifest_path.resolve()),
            "bytes": duplicate_manifest_path.stat().st_size,
            "sha256": sha256_file(duplicate_manifest_path),
            "records": drops,
            "ordered_identity_sha256": drop_identity.hexdigest(),
            "contains_source_text": False,
        },
        "normalized_hash_collisions_replayed_against_full_text": True,
        "exact_document_custody": True,
        "global_normalized_exact_deduplication_complete": True,
        "global_near_duplicate_filtering_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    try:
        _atomic_create(receipt_path, payload)
    except BaseException:
        output_path.unlink(missing_ok=True)
        duplicate_manifest_path.unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duplicate-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--chunk-records", type=int, default=DEFAULT_CHUNK_RECORDS)
    parser.add_argument(
        "--maximum-line-bytes", type=int, default=DEFAULT_MAXIMUM_LINE_BYTES
    )
    parser.add_argument(
        "--maximum-open-chunks", type=int, default=DEFAULT_MAXIMUM_OPEN_CHUNKS
    )
    parser.add_argument("--temporary-root", type=Path)
    args = parser.parse_args()
    result = build_exact_deduplication(
        args.source,
        args.output,
        args.duplicate_manifest,
        args.receipt,
        chunk_records=args.chunk_records,
        maximum_line_bytes=args.maximum_line_bytes,
        maximum_open_chunks=args.maximum_open_chunks,
        temporary_root=args.temporary_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
