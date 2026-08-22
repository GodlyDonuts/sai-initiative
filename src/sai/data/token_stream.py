"""Freeze benchmark-disjoint documents into deterministic packed token shards."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import os
import shutil
import struct
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

SCHEMA = "sai-ordered-token-stream-receipt-v1"
ROW_SCHEMA = "sai-pretraining-document-v1"
ALLOWED_DOMAINS = {"english", "code", "math", "science", "technical"}


class TokenStreamError(RuntimeError):
    """A source, tokenizer projection, packed boundary, or receipt differs."""


class OffsetTokenizer(Protocol):
    eos_token_id: int | None
    vocab_size: int

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> Mapping[str, Any]: ...

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str: ...


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_token_file(path: Path, expected_sha256: str, vocab_size: int) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            if len(chunk) % 4:
                raise TokenStreamError("token shard is not uint32 aligned")
            values = array.array("I")
            values.frombytes(chunk)
            if sys.byteorder != "little":
                values.byteswap()
            if values and max(values) >= vocab_size:
                raise TokenStreamError("token shard contains an out-of-vocabulary ID")
    if digest.hexdigest() != expected_sha256:
        raise TokenStreamError("token stream shard content differs")


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TokenStreamError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise TokenStreamError(f"{field} differs") from error
    return value


def sha256_tree(root: Path) -> str:
    """Hash exact regular-file membership and bytes under a tokenizer root."""

    if not root.is_dir() or root.is_symlink():
        raise TokenStreamError("tokenizer root is missing or unsafe")
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise TokenStreamError(f"tokenizer tree contains a link: {relative}")
        if path.is_dir():
            rows.append({"path": relative, "type": "directory"})
        elif path.is_file():
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        else:
            raise TokenStreamError(f"tokenizer tree contains a special: {relative}")
    if not rows or not any(row["type"] == "file" for row in rows):
        raise TokenStreamError("tokenizer root contains no files")
    return canonical_sha256(rows)


def normalize_document(row: Any) -> dict[str, Any]:
    """Validate and canonicalize one benchmark-disjoint source document."""
    if not isinstance(row, dict) or row.get("schema") != ROW_SCHEMA:
        raise TokenStreamError("pretraining document schema differs")
    text = row.get("text")
    if not isinstance(text, str) or not text:
        raise TokenStreamError("pretraining document text is empty")
    source = row.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("dataset"), str)
        or not source["dataset"]
        or not isinstance(source.get("row_id"), str)
        or not source["row_id"]
        or not isinstance(source.get("license"), str)
        or not source["license"]
        or source.get("domain") not in ALLOWED_DOMAINS
    ):
        raise TokenStreamError("pretraining document provenance differs")
    verification = row.get("verification")
    if (
        not isinstance(verification, dict)
        or verification.get("benchmark_disjoint") is not True
    ):
        raise TokenStreamError("benchmark-disjoint document evidence differs")
    normalized = {
        "schema": ROW_SCHEMA,
        "text": text,
        "source": {
            "dataset": source["dataset"],
            "row_id": source["row_id"],
            "license": source["license"],
            "domain": source["domain"],
        },
        "verification": {
            "benchmark_disjoint": True,
            "evidence_sha256": _sha256(
                verification.get("evidence_sha256"),
                "document decontamination evidence",
            ),
        },
    }
    identity = canonical_sha256(normalized)
    if row.get("identity_sha256") not in (None, identity):
        raise TokenStreamError("declared document identity differs")
    return {**normalized, "identity_sha256": identity}


def _tokenize(
    tokenizer: OffsetTokenizer, text: str, vocabulary_size: int
) -> tuple[list[int], list[int]]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if not isinstance(encoded, Mapping):
        raise TokenStreamError("tokenizer output differs")
    token_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or not isinstance(offsets, list)
        or len(offsets) != len(token_ids)
    ):
        raise TokenStreamError("tokenizer IDs or offsets differ")
    normalized_ids = []
    byte_increments = []
    covered = 0
    for token_id, offset in zip(token_ids, offsets, strict=True):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < vocabulary_size
            or not isinstance(offset, (list, tuple))
            or len(offset) != 2
        ):
            raise TokenStreamError("tokenizer ID or offset value differs")
        start, end = offset
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start <= end <= len(text)
            or start > covered
            or end < covered
        ):
            raise TokenStreamError("tokenizer offsets are not contiguous")
        byte_increments.append(len(text[covered:end].encode("utf-8")))
        normalized_ids.append(token_id)
        covered = end
    if covered != len(text):
        raise TokenStreamError("tokenizer offsets do not cover the source text")
    decoded = tokenizer.decode(
        normalized_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != text or sum(byte_increments) != len(text.encode("utf-8")):
        raise TokenStreamError("tokenizer round trip differs")
    return normalized_ids, byte_increments


def _start_bits(starts: list[bool], sequence_length: int) -> bytes:
    if len(starts) != sequence_length or not starts[0]:
        raise TokenStreamError("packed segment starts differ")
    payload = bytearray((sequence_length + 7) // 8)
    for index, is_start in enumerate(starts):
        if is_start:
            payload[index // 8] |= 1 << (index % 8)
    return bytes(payload)


def decode_segment_starts(payload: bytes, sequence_length: int) -> list[bool]:
    expected_bytes = (sequence_length + 7) // 8
    if sequence_length <= 1 or len(payload) != expected_bytes:
        raise TokenStreamError("segment-start bitset geometry differs")
    unused_bits = expected_bytes * 8 - sequence_length
    if unused_bits and payload[-1] >> (8 - unused_bits):
        raise TokenStreamError("segment-start bitset has nonzero padding")
    starts = [
        bool(payload[index // 8] & (1 << (index % 8)))
        for index in range(sequence_length)
    ]
    if not starts[0]:
        raise TokenStreamError("every packed sequence must begin a segment")
    return starts


def segment_ids_from_start_bits(payload: bytes, sequence_length: int) -> list[int]:
    starts = decode_segment_starts(payload, sequence_length)
    segment = -1
    identities = []
    for is_start in starts:
        if is_start:
            segment += 1
        identities.append(segment)
    return identities


def causal_loss_mask_from_start_bits(
    payload: bytes, sequence_length: int
) -> list[bool]:
    """Return positions whose next-token target stays inside one segment."""

    starts = decode_segment_starts(payload, sequence_length)
    return [not starts[index + 1] for index in range(sequence_length - 1)] + [False]


def _source_receipts(paths: list[Path]) -> list[dict[str, Any]]:
    if not paths:
        raise TokenStreamError("at least one pretraining source is required")
    receipts = []
    resolved_paths = set()
    for order, path in enumerate(paths):
        if not path.is_file() or path.is_symlink():
            raise TokenStreamError(f"pretraining source is missing or unsafe: {path}")
        resolved = str(path.resolve())
        if resolved in resolved_paths:
            raise TokenStreamError("pretraining source paths must be unique")
        resolved_paths.add(resolved)
        size = path.stat().st_size
        if size <= 0:
            raise TokenStreamError("pretraining source is empty")
        receipts.append(
            {
                "order": order,
                "path": resolved,
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    return receipts


def freeze(
    tokenizer: OffsetTokenizer,
    sources: list[Path],
    output: Path,
    *,
    tokenizer_identity_sha256: str,
    sequence_length: int,
    prefix_sequences: set[int],
    sequences_per_shard: int = 4_096,
    source_qualification_sha256: str | None = None,
    curriculum_phases: list[tuple[str, int]] | None = None,
    required_phase_complete_prefixes: set[int] | None = None,
) -> dict[str, Any]:
    """Pack an explicit source order into uint32 tokens and boundary bitsets."""

    tokenizer_identity = _sha256(tokenizer_identity_sha256, "tokenizer identity")
    source_qualification = (
        None
        if source_qualification_sha256 is None
        else _sha256(source_qualification_sha256, "source qualification")
    )
    normalized_curriculum_phases = (
        [] if curriculum_phases is None else list(curriculum_phases)
    )
    required_phase_prefixes = required_phase_complete_prefixes or set()
    if (
        any(
            not isinstance(name, str)
            or not name
            or isinstance(documents, bool)
            or not isinstance(documents, int)
            or documents <= 0
            for name, documents in normalized_curriculum_phases
        )
        or len({name for name, _ in normalized_curriculum_phases})
        != len(normalized_curriculum_phases)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in required_phase_prefixes
        )
        or (required_phase_prefixes and not normalized_curriculum_phases)
    ):
        raise TokenStreamError("curriculum phase geometry differs")
    if output.exists():
        raise TokenStreamError("token stream output already exists")
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 1
        or isinstance(sequences_per_shard, bool)
        or not isinstance(sequences_per_shard, int)
        or sequences_per_shard <= 0
        or not prefix_sequences
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in prefix_sequences
        )
    ):
        raise TokenStreamError("packed stream geometry differs")
    if not required_phase_prefixes.issubset(prefix_sequences):
        raise TokenStreamError("required curriculum prefixes differ")
    base_vocabulary_size = getattr(tokenizer, "vocab_size", None)
    try:
        vocabulary_size = len(tokenizer)  # type: ignore[arg-type]
    except TypeError:
        vocabulary_size = base_vocabulary_size
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if (
        isinstance(base_vocabulary_size, bool)
        or not isinstance(base_vocabulary_size, int)
        or base_vocabulary_size <= 0
        or isinstance(vocabulary_size, bool)
        or not isinstance(vocabulary_size, int)
        or vocabulary_size < base_vocabulary_size
        or isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or not 0 <= eos_token_id < vocabulary_size
    ):
        raise TokenStreamError("tokenizer vocabulary or EOS identity differs")
    source_receipts = _source_receipts(sources)
    source_manifest_sha256 = canonical_sha256(source_receipts)
    target_sequences = max(prefix_sequences)
    stage = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if stage.exists():
        raise TokenStreamError("token stream staging path already exists")
    stage.mkdir(parents=True)

    tokens: list[int] = []
    starts: list[bool] = []
    byte_increments: list[int] = []
    curriculum_labels: list[str] = []
    cumulative_utf8_bytes = 0
    prefix_bytes: dict[str, int] = {}
    sequence_count = 0
    shard_index = 0
    shard_sequences = 0
    token_handle = start_handle = None
    shard_receipts: list[dict[str, Any]] = []
    seen_identities: set[str] = set()
    seen_texts: set[str] = set()
    domain_documents: Counter[str] = Counter()
    curriculum_documents: Counter[str] = Counter()
    curriculum_tokens: Counter[str] = Counter()
    curriculum_bytes: Counter[str] = Counter()
    curriculum_prefixes: dict[str, dict[str, dict[str, int]]] = {}
    documents_scanned = documents_accepted = duplicate_documents = malformed = 0

    def open_shard() -> tuple[Any, Any]:
        token_path = stage / f"shard_{shard_index:05d}.tokens.u32le"
        start_path = stage / f"shard_{shard_index:05d}.starts.bitset"
        return token_path.open("wb"), start_path.open("wb")

    def close_shard() -> None:
        nonlocal token_handle, start_handle, shard_sequences
        if token_handle is None or start_handle is None:
            return
        token_path = Path(token_handle.name)
        start_path = Path(start_handle.name)
        token_handle.close()
        start_handle.close()
        shard_receipts.append(
            {
                "index": len(shard_receipts),
                "sequences": shard_sequences,
                "tokens": {
                    "path": token_path.name,
                    "bytes": token_path.stat().st_size,
                    "sha256": sha256_file(token_path),
                },
                "segment_starts": {
                    "path": start_path.name,
                    "bytes": start_path.stat().st_size,
                    "sha256": sha256_file(start_path),
                },
            }
        )
        token_handle = start_handle = None
        shard_sequences = 0

    def write_sequence() -> None:
        nonlocal token_handle, start_handle, shard_index, shard_sequences
        nonlocal sequence_count, cumulative_utf8_bytes
        if token_handle is None:
            token_handle, start_handle = open_shard()
        if len(tokens) != sequence_length:
            raise TokenStreamError("attempted to write a partial packed sequence")
        token_handle.write(struct.pack(f"<{sequence_length}I", *tokens))
        start_handle.write(_start_bits(starts, sequence_length))
        cumulative_utf8_bytes += sum(byte_increments)
        if normalized_curriculum_phases:
            for phase, byte_count in zip(
                curriculum_labels, byte_increments, strict=True
            ):
                curriculum_tokens[phase] += 1
                curriculum_bytes[phase] += byte_count
        sequence_count += 1
        shard_sequences += 1
        if sequence_count in prefix_sequences:
            prefix_bytes[str(sequence_count)] = cumulative_utf8_bytes
            if normalized_curriculum_phases:
                curriculum_prefixes[str(sequence_count)] = {
                    phase: {
                        "tokens": curriculum_tokens[phase],
                        "utf8_bytes": curriculum_bytes[phase],
                    }
                    for phase, _ in normalized_curriculum_phases
                }
        tokens.clear()
        starts.clear()
        byte_increments.clear()
        curriculum_labels.clear()
        if shard_sequences == sequences_per_shard:
            close_shard()
            shard_index += 1

    try:
        phase_boundaries: list[tuple[str, int]] = []
        phase_total = 0
        for phase_name, phase_documents in normalized_curriculum_phases:
            phase_total += phase_documents
            phase_boundaries.append((phase_name, phase_total))

        def curriculum_phase(document_index: int) -> str:
            if not phase_boundaries:
                return ""
            for phase_name, boundary in phase_boundaries:
                if document_index < boundary:
                    return phase_name
            raise TokenStreamError("curriculum source exceeds declared phases")

        complete = False
        for source in sources:
            with source.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    documents_scanned += 1
                    try:
                        row = normalize_document(json.loads(line))
                    except (json.JSONDecodeError, TokenStreamError):
                        malformed += 1
                        continue
                    text_identity = hashlib.sha256(row["text"].encode()).hexdigest()
                    if (
                        row["identity_sha256"] in seen_identities
                        or text_identity in seen_texts
                    ):
                        duplicate_documents += 1
                        continue
                    document_tokens, document_bytes = _tokenize(
                        tokenizer, row["text"], vocabulary_size
                    )
                    phase = curriculum_phase(documents_accepted)
                    seen_identities.add(row["identity_sha256"])
                    seen_texts.add(text_identity)
                    domain_documents[row["source"]["domain"]] += 1
                    if phase:
                        curriculum_documents[phase] += 1
                    documents_accepted += 1
                    document_tokens.append(eos_token_id)
                    document_bytes.append(0)
                    for document_index, (token_id, byte_count) in enumerate(
                        zip(document_tokens, document_bytes, strict=True)
                    ):
                        starts.append(not tokens or document_index == 0)
                        tokens.append(token_id)
                        byte_increments.append(byte_count)
                        if phase:
                            curriculum_labels.append(phase)
                        if len(tokens) == sequence_length:
                            write_sequence()
                            if sequence_count == target_sequences:
                                complete = True
                                break
                    if complete:
                        break
                if complete:
                    break
        if not complete or set(prefix_bytes) != {
            str(value) for value in prefix_sequences
        }:
            raise TokenStreamError(
                "pretraining sources cannot fill every required prefix"
            )
        curriculum_phase_names = [phase for phase, _ in normalized_curriculum_phases]
        for prefix in required_phase_prefixes:
            evidence = curriculum_prefixes.get(str(prefix), {})
            if any(
                evidence.get(phase, {}).get("tokens", 0) <= 0
                or evidence.get(phase, {}).get("utf8_bytes", 0) <= 0
                for phase in curriculum_phase_names
            ):
                raise TokenStreamError(
                    "required training prefix lacks a curriculum phase"
                )
        close_shard()
        report_unsigned: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "complete",
            "training_authorized": False,
            "tokenizer_identity_sha256": tokenizer_identity,
            "source_manifest_sha256": source_manifest_sha256,
            "source_receipts": source_receipts,
            "sequence_length": sequence_length,
            "sequences": sequence_count,
            "valid_tokens": sequence_count * sequence_length,
            "admitted_utf8_bytes": cumulative_utf8_bytes,
            "prefix_utf8_bytes": prefix_bytes,
            "benchmark_disjoint": True,
            "cross_document_targets_masked": True,
            "token_encoding": "little_endian_uint32",
            "segment_start_encoding": "little_endian_bitset_lsb_first",
            "eos_token_id": eos_token_id,
            "vocab_size": vocabulary_size,
            "sequences_per_shard": sequences_per_shard,
            "shards": shard_receipts,
            "documents": {
                "scanned": documents_scanned,
                "accepted": documents_accepted,
                "duplicates_dropped": duplicate_documents,
                "malformed_or_unverified_dropped": malformed,
                "accepted_by_domain": dict(sorted(domain_documents.items())),
            },
        }
        if normalized_curriculum_phases:
            report_unsigned["curriculum"] = {
                "declared_phase_documents": {
                    phase: documents
                    for phase, documents in normalized_curriculum_phases
                },
                "consumed_phase_documents": {
                    phase: curriculum_documents[phase]
                    for phase, _ in normalized_curriculum_phases
                },
                "consumed_phase_tokens": {
                    phase: curriculum_tokens[phase]
                    for phase, _ in normalized_curriculum_phases
                },
                "consumed_phase_utf8_bytes": {
                    phase: curriculum_bytes[phase]
                    for phase, _ in normalized_curriculum_phases
                },
                "prefixes": curriculum_prefixes,
                "required_all_phase_prefixes": sorted(required_phase_prefixes),
                "all_required_prefixes_cover_every_phase": True,
            }
        if source_qualification is not None:
            report_unsigned["source_qualification_sha256"] = source_qualification
        report_unsigned["ordered_stream_identity_sha256"] = canonical_sha256(
            report_unsigned
        )
        report_path = stage / "stream_receipt.json"
        report_path.write_text(
            json.dumps(report_unsigned, indent=2, sort_keys=True) + "\n"
        )
        os.replace(stage, output)
        return report_unsigned
    except BaseException:
        if token_handle is not None:
            token_handle.close()
        if start_handle is not None:
            start_handle.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def validate_frozen_stream(
    output: Path, *, verify_sources: bool = True
) -> dict[str, Any]:
    if not output.is_dir() or output.is_symlink():
        raise TokenStreamError("token stream root is missing or unsafe")
    report_path = output / "stream_receipt.json"
    if not report_path.is_file() or report_path.is_symlink():
        raise TokenStreamError("token stream receipt is missing or unsafe")
    try:
        report = json.loads(report_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TokenStreamError("token stream receipt is unreadable") from error
    if (
        not isinstance(report, dict)
        or report.get("schema") != SCHEMA
        or report.get("status") != "complete"
        or report.get("training_authorized") is not False
        or report.get("benchmark_disjoint") is not True
        or report.get("cross_document_targets_masked") is not True
    ):
        raise TokenStreamError("token stream receipt identity differs")
    sequence_length = report.get("sequence_length")
    sequences = report.get("sequences")
    sequences_per_shard = report.get("sequences_per_shard")
    vocab_size = report.get("vocab_size")
    eos_token_id = report.get("eos_token_id")
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 1
        or isinstance(sequences, bool)
        or not isinstance(sequences, int)
        or sequences <= 0
        or report.get("valid_tokens") != sequence_length * sequences
        or isinstance(sequences_per_shard, bool)
        or not isinstance(sequences_per_shard, int)
        or sequences_per_shard <= 0
        or isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 0
        or isinstance(eos_token_id, bool)
        or not isinstance(eos_token_id, int)
        or not 0 <= eos_token_id < vocab_size
        or report.get("token_encoding") != "little_endian_uint32"
        or report.get("segment_start_encoding") != "little_endian_bitset_lsb_first"
    ):
        raise TokenStreamError("token stream geometry differs")
    _sha256(report.get("tokenizer_identity_sha256"), "tokenizer identity")
    _sha256(report.get("source_manifest_sha256"), "source manifest")
    if "source_qualification_sha256" in report:
        _sha256(report["source_qualification_sha256"], "source qualification")
    admitted_bytes = report.get("admitted_utf8_bytes")
    prefixes = report.get("prefix_utf8_bytes")
    if (
        isinstance(admitted_bytes, bool)
        or not isinstance(admitted_bytes, int)
        or admitted_bytes <= 0
        or not isinstance(prefixes, dict)
        or not prefixes
    ):
        raise TokenStreamError("token stream UTF-8 byte ledger differs")
    previous_count = previous_bytes = 0
    try:
        ordered_prefixes = sorted((int(key), value) for key, value in prefixes.items())
    except (TypeError, ValueError) as error:
        raise TokenStreamError("token stream prefix keys differ") from error
    for count, byte_count in ordered_prefixes:
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or count <= previous_count
            or count > sequences
            or byte_count <= 0
            or byte_count < previous_bytes
        ):
            raise TokenStreamError("token stream prefix ledger is not monotonic")
        previous_count, previous_bytes = count, byte_count
    if previous_count != sequences or previous_bytes != admitted_bytes:
        raise TokenStreamError("token stream full prefix differs")
    documents = report.get("documents")
    if not isinstance(documents, dict) or set(documents) != {
        "scanned",
        "accepted",
        "duplicates_dropped",
        "malformed_or_unverified_dropped",
        "accepted_by_domain",
    }:
        raise TokenStreamError("token stream document accounting differs")
    for key in (
        "scanned",
        "accepted",
        "duplicates_dropped",
        "malformed_or_unverified_dropped",
    ):
        if (
            isinstance(documents[key], bool)
            or not isinstance(documents[key], int)
            or documents[key] < 0
        ):
            raise TokenStreamError("token stream document counts differ")
    domains = documents["accepted_by_domain"]
    if (
        documents["accepted"] <= 0
        or documents["scanned"]
        != documents["accepted"]
        + documents["duplicates_dropped"]
        + documents["malformed_or_unverified_dropped"]
        or not isinstance(domains, dict)
        or not set(domains).issubset(ALLOWED_DOMAINS)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in domains.values()
        )
        or sum(domains.values()) != documents["accepted"]
    ):
        raise TokenStreamError("token stream document accounting is inconsistent")
    curriculum = report.get("curriculum")
    if curriculum is not None:
        if not isinstance(curriculum, dict) or set(curriculum) != {
            "declared_phase_documents",
            "consumed_phase_documents",
            "consumed_phase_tokens",
            "consumed_phase_utf8_bytes",
            "prefixes",
            "required_all_phase_prefixes",
            "all_required_prefixes_cover_every_phase",
        }:
            raise TokenStreamError("token stream curriculum evidence differs")
        declared = curriculum["declared_phase_documents"]
        consumed_documents = curriculum["consumed_phase_documents"]
        consumed_tokens = curriculum["consumed_phase_tokens"]
        consumed_bytes = curriculum["consumed_phase_utf8_bytes"]
        phase_names = set(declared) if isinstance(declared, dict) else set()
        if (
            not phase_names
            or not all(
                isinstance(mapping, dict) and set(mapping) == phase_names
                for mapping in (consumed_documents, consumed_tokens, consumed_bytes)
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in declared.values()
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for mapping in (consumed_documents, consumed_tokens, consumed_bytes)
                for value in mapping.values()
            )
            or sum(declared.values()) < documents["accepted"]
            or sum(consumed_documents.values()) != documents["accepted"]
            or sum(consumed_tokens.values()) != report["valid_tokens"]
            or sum(consumed_bytes.values()) != admitted_bytes
        ):
            raise TokenStreamError("token stream curriculum totals differ")
        curriculum_prefixes = curriculum["prefixes"]
        if not isinstance(curriculum_prefixes, dict) or set(curriculum_prefixes) != set(
            prefixes
        ):
            raise TokenStreamError("token stream curriculum prefixes differ")
        for prefix, phase_evidence in curriculum_prefixes.items():
            if (
                not isinstance(phase_evidence, dict)
                or set(phase_evidence) != phase_names
            ):
                raise TokenStreamError("token stream curriculum phase set differs")
            for values in phase_evidence.values():
                if (
                    not isinstance(values, dict)
                    or set(values) != {"tokens", "utf8_bytes"}
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in values.values()
                    )
                ):
                    raise TokenStreamError(
                        "token stream curriculum prefix values differ"
                    )
            if (
                sum(values["tokens"] for values in phase_evidence.values())
                != int(prefix) * sequence_length
                or sum(values["utf8_bytes"] for values in phase_evidence.values())
                != prefixes[prefix]
            ):
                raise TokenStreamError("token stream curriculum prefix totals differ")
        required = curriculum["required_all_phase_prefixes"]
        if (
            not isinstance(required, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or str(value) not in curriculum_prefixes
                for value in required
            )
            or required != sorted(set(required))
            or curriculum["all_required_prefixes_cover_every_phase"] is not True
        ):
            raise TokenStreamError("token stream curriculum requirement differs")
        for prefix in required:
            if any(
                curriculum_prefixes[str(prefix)][phase]["tokens"] <= 0
                or curriculum_prefixes[str(prefix)][phase]["utf8_bytes"] <= 0
                for phase in phase_names
            ):
                raise TokenStreamError("token stream curriculum phase is absent")
    shards = report.get("shards")
    if not isinstance(shards, list) or not shards:
        raise TokenStreamError("token stream shards are missing")
    expected_members = {"stream_receipt.json"}
    observed_sequences = 0
    start_bytes_per_sequence = (sequence_length + 7) // 8
    for expected_index, shard in enumerate(shards):
        if not isinstance(shard, dict) or shard.get("index") != expected_index:
            raise TokenStreamError("token stream shard order differs")
        shard_sequences = shard.get("sequences")
        if (
            isinstance(shard_sequences, bool)
            or not isinstance(shard_sequences, int)
            or shard_sequences <= 0
        ):
            raise TokenStreamError("token stream shard geometry differs")
        if expected_index < len(shards) - 1 and shard_sequences != sequences_per_shard:
            raise TokenStreamError("nonfinal token shard is partial")
        if shard_sequences > sequences_per_shard:
            raise TokenStreamError("token shard exceeds its sequence geometry")
        observed_sequences += shard_sequences
        for key, expected_bytes in (
            ("tokens", shard_sequences * sequence_length * 4),
            ("segment_starts", shard_sequences * start_bytes_per_sequence),
        ):
            receipt = shard.get(key)
            if not isinstance(receipt, dict) or set(receipt) != {
                "path",
                "bytes",
                "sha256",
            }:
                raise TokenStreamError("token stream shard receipt differs")
            relative = receipt["path"]
            if (
                not isinstance(relative, str)
                or Path(relative).name != relative
                or relative in expected_members
                or receipt["bytes"] != expected_bytes
            ):
                raise TokenStreamError("token stream shard path or bytes differ")
            path = output / relative
            expected_hash = _sha256(receipt["sha256"], "shard sha256")
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != expected_bytes
            ):
                raise TokenStreamError("token stream shard content differs")
            if key == "tokens":
                _validate_token_file(path, expected_hash, vocab_size)
            else:
                if sha256_file(path) != expected_hash:
                    raise TokenStreamError("token stream shard content differs")
                with path.open("rb") as handle:
                    for _ in range(shard_sequences):
                        decode_segment_starts(
                            handle.read(start_bytes_per_sequence), sequence_length
                        )
                    if handle.read(1):
                        raise TokenStreamError("segment-start shard has trailing bytes")
            expected_members.add(relative)
    observed_members = {
        path.name
        for path in output.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if observed_members != expected_members or observed_sequences != sequences:
        raise TokenStreamError("token stream membership or sequence total differs")
    if any(path.is_symlink() or not path.is_file() for path in output.iterdir()):
        raise TokenStreamError("token stream contains links, directories, or specials")
    receipts = report.get("source_receipts")
    if not isinstance(receipts, list) or not receipts:
        raise TokenStreamError("token stream source receipts are missing")
    for expected_order, receipt in enumerate(receipts):
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"order", "path", "bytes", "sha256"}
            or receipt.get("order") != expected_order
            or not isinstance(receipt.get("path"), str)
            or not receipt["path"]
            or isinstance(receipt.get("bytes"), bool)
            or not isinstance(receipt.get("bytes"), int)
            or receipt["bytes"] <= 0
        ):
            raise TokenStreamError("token stream source order differs")
        _sha256(receipt.get("sha256"), "source sha256")
    if canonical_sha256(receipts) != report.get("source_manifest_sha256"):
        raise TokenStreamError("token stream source manifest differs")
    if verify_sources:
        for receipt in receipts:
            path = Path(receipt.get("path", ""))
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != receipt.get("bytes")
                or sha256_file(path) != _sha256(receipt.get("sha256"), "source sha256")
            ):
                raise TokenStreamError("token stream source content differs")
    unsigned = {
        key: value
        for key, value in report.items()
        if key != "ordered_stream_identity_sha256"
    }
    if canonical_sha256(unsigned) != _sha256(
        report.get("ordered_stream_identity_sha256"), "ordered stream identity"
    ):
        raise TokenStreamError("ordered stream identity differs")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-sha256", required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=2_048)
    parser.add_argument("--prefix-sequences", type=int, action="append", required=True)
    parser.add_argument("--sequences-per-shard", type=int, default=4_096)
    parser.add_argument("--source-qualification-sha256")
    parser.add_argument("--curriculum-receipt", type=Path)
    parser.add_argument(
        "--require-all-curriculum-phases-at-prefix", type=int, action="append"
    )
    args = parser.parse_args()
    observed_identity = sha256_tree(args.tokenizer_root)
    if observed_identity != _sha256(
        args.expected_tokenizer_sha256, "expected tokenizer identity"
    ):
        raise TokenStreamError("tokenizer tree identity differs")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TokenStreamError(
            "Transformers is required to load a tokenizer"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_root,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise TokenStreamError("a fast tokenizer with offsets is required")
    curriculum_phases = None
    if args.curriculum_receipt is not None:
        path = args.curriculum_receipt
        if not path.is_file() or path.is_symlink():
            raise TokenStreamError("curriculum receipt is missing or unsafe")
        curriculum = json.loads(path.read_text())
        unsigned = {
            key: value for key, value in curriculum.items() if key != "receipt_sha256"
        }
        phases = curriculum.get("phases")
        if (
            curriculum.get("schema") != "sai-curriculum-order-receipt-v1"
            or curriculum.get("status") != "qualified"
            or curriculum.get("curriculum_qualified") is not True
            or curriculum.get("training_authorized") is not False
            or curriculum.get("receipt_sha256") != canonical_sha256(unsigned)
            or not isinstance(phases, dict)
            or len(args.source) != 1
            or curriculum.get("output", {}).get("path") != str(args.source[0].resolve())
            or curriculum.get("output", {}).get("bytes")
            != args.source[0].stat().st_size
            or curriculum.get("output", {}).get("sha256") != sha256_file(args.source[0])
            or args.source_qualification_sha256 != sha256_file(path)
        ):
            raise TokenStreamError("curriculum receipt differs")
        ordered_phases = sorted(
            phases.items(), key=lambda item: item[1].get("index", -1)
        )
        if [row.get("index") for _, row in ordered_phases] != list(
            range(len(ordered_phases))
        ):
            raise TokenStreamError("curriculum phase order differs")
        curriculum_phases = [
            (name, row.get("documents")) for name, row in ordered_phases
        ]
    report = freeze(
        tokenizer,
        args.source,
        args.output,
        tokenizer_identity_sha256=observed_identity,
        sequence_length=args.sequence_length,
        prefix_sequences=set(args.prefix_sequences),
        sequences_per_shard=args.sequences_per_shard,
        source_qualification_sha256=args.source_qualification_sha256,
        curriculum_phases=curriculum_phases,
        required_phase_complete_prefixes=set(
            args.require_all_curriculum_phases_at_prefix or []
        ),
    )
    print(
        json.dumps(
            {
                "ordered_stream_identity_sha256": report[
                    "ordered_stream_identity_sha256"
                ],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
