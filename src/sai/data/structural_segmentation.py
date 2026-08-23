"""Losslessly segment long raw documents while preserving exact lineage."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.decontamination import RAW_SCHEMA
from sai.data.source_lineage import SEGMENT_SCHEMA, parent_row_id, source_locator
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-structural-segmentation-receipt-v1"
LINEAGE_SCHEMA = "sai-structural-segmentation-record-v1"
DEFAULT_MINIMUM_BYTES = 200
DEFAULT_MAXIMUM_BYTES = 128 * 1024

_PARAGRAPH = re.compile(r"\n[ \t]*\n+")
_LINE = re.compile(r"\n")
_SENTENCE = re.compile(r"(?<=[.!?])(?:[ \t]+|\n)")
_CLAUSE = re.compile(r"(?<=[;:,])(?:[ \t]+|\n)")
_WORD = re.compile(r"\s+")
_BOUNDARIES = (
    ("paragraph", _PARAGRAPH),
    ("line", _LINE),
    ("sentence", _SENTENCE),
    ("clause", _CLAUSE),
    ("word", _WORD),
)


class StructuralSegmentationError(RuntimeError):
    """The input, segmentation geometry, lineage, or output differs."""


def _byte_prefix(text: str) -> list[int]:
    prefix = [0]
    for character in text:
        prefix.append(prefix[-1] + len(character.encode()))
    return prefix


def _candidate_boundaries(text: str) -> dict[int, str]:
    candidates: dict[int, str] = {}
    for kind, pattern in _BOUNDARIES:
        for match in pattern.finditer(text):
            candidates.setdefault(match.end(), kind)
    return candidates


def segment_text(
    text: str,
    *,
    minimum_bytes: int = DEFAULT_MINIMUM_BYTES,
    maximum_bytes: int = DEFAULT_MAXIMUM_BYTES,
) -> list[dict[str, Any]]:
    """Return ordered, byte-bounded spans whose exact concatenation is ``text``."""

    if (
        not isinstance(text, str)
        or not text
        or isinstance(minimum_bytes, bool)
        or not isinstance(minimum_bytes, int)
        or isinstance(maximum_bytes, bool)
        or not isinstance(maximum_bytes, int)
        or minimum_bytes <= 0
        or maximum_bytes < max(4, minimum_bytes * 2)
    ):
        raise StructuralSegmentationError("segmentation geometry differs")
    prefix = _byte_prefix(text)
    if prefix[-1] <= maximum_bytes:
        return [
            {
                "text": text,
                "character_start": 0,
                "character_end": len(text),
                "utf8_start": 0,
                "utf8_end": prefix[-1],
                "end_boundary": "document_end",
            }
        ]
    candidates = _candidate_boundaries(text)
    spans = []
    start = 0
    while start < len(text):
        byte_limit = prefix[start] + maximum_bytes
        limit = bisect.bisect_right(prefix, byte_limit) - 1
        if limit <= start:
            raise StructuralSegmentationError("a character exceeds the byte budget")
        if limit == len(text):
            end = limit
            kind = "document_end"
        else:
            eligible = [
                index
                for index in candidates
                if start < index <= limit
                and prefix[index] - prefix[start] >= minimum_bytes
            ]
            if eligible:
                best_priority = min(
                    next(
                        order
                        for order, (name, _) in enumerate(_BOUNDARIES)
                        if name == candidates[index]
                    )
                    for index in eligible
                )
                preferred = [
                    index
                    for index in eligible
                    if next(
                        order
                        for order, (name, _) in enumerate(_BOUNDARIES)
                        if name == candidates[index]
                    )
                    == best_priority
                ]
                end = max(preferred)
                kind = candidates[end]
            else:
                end = limit
                kind = "utf8_budget"
        spans.append(
            {
                "text": text[start:end],
                "character_start": start,
                "character_end": end,
                "utf8_start": prefix[start],
                "utf8_end": prefix[end],
                "end_boundary": kind,
            }
        )
        start = end
    if "".join(span["text"] for span in spans) != text:
        raise StructuralSegmentationError("segmentation is not lossless")
    if any(len(span["text"].encode()) > maximum_bytes for span in spans):
        raise StructuralSegmentationError("segment exceeds the byte budget")
    return spans


def _raw_row(row: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(row, dict) or row.get("schema") != RAW_SCHEMA:
        raise StructuralSegmentationError("raw document schema differs")
    text = row.get("text")
    source = row.get("source")
    if not isinstance(text, str) or not text or not isinstance(source, dict):
        raise StructuralSegmentationError("raw document differs")
    source_locator(source)
    if "segment" in source:
        raise StructuralSegmentationError("raw document is already segmented")
    for field in ("license", "declared_license", "domain"):
        if not isinstance(source.get(field), str) or not source[field]:
            raise StructuralSegmentationError("raw document provenance differs")
    return text, source


def build_segments(
    source_path: Path,
    output_path: Path,
    lineage_path: Path,
    receipt_path: Path,
    *,
    minimum_bytes: int = DEFAULT_MINIMUM_BYTES,
    maximum_bytes: int = DEFAULT_MAXIMUM_BYTES,
) -> dict[str, Any]:
    """Segment a raw JSONL population and seal text-free reconstruction evidence."""

    if (
        not source_path.is_file()
        or source_path.is_symlink()
        or source_path.stat().st_nlink != 1
        or any(
            path.exists() or path.is_symlink()
            for path in (output_path, lineage_path, receipt_path)
        )
    ):
        raise StructuralSegmentationError("segmentation input or output differs")
    source_sha256 = sha256_file(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_path.parent.mkdir(parents=True, exist_ok=True)
    output_stage = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    lineage_stage = lineage_path.with_name(f".{lineage_path.name}.tmp.{os.getpid()}")
    counters: Counter[str] = Counter()
    parent_identity = hashlib.sha256()
    child_identity = hashlib.sha256()
    try:
        with (
            source_path.open() as source,
            output_stage.open("w") as output,
            lineage_stage.open("w") as lineage,
        ):
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                    text, raw_source = _raw_row(raw)
                except (json.JSONDecodeError, StructuralSegmentationError) as error:
                    raise StructuralSegmentationError(
                        f"raw document row {line_number} differs"
                    ) from error
                parent_text_sha256 = hashlib.sha256(text.encode()).hexdigest()
                parent_id = parent_row_id(raw_source)
                spans = segment_text(
                    text,
                    minimum_bytes=minimum_bytes,
                    maximum_bytes=maximum_bytes,
                )
                counters["input_documents"] += 1
                counters["input_text_utf8_bytes"] += len(text.encode())
                counters["output_segments"] += len(spans)
                counters["segmented_documents"] += len(spans) > 1
                parent_identity.update(bytes.fromhex(parent_id))
                records = []
                for index, span in enumerate(spans):
                    segment_text_sha256 = hashlib.sha256(
                        span["text"].encode()
                    ).hexdigest()
                    if len(spans) == 1:
                        child_source = raw_source
                    else:
                        segment = {
                            "schema": SEGMENT_SCHEMA,
                            "parent_row_id": parent_id,
                            "parent_text_sha256": parent_text_sha256,
                            "index": index,
                            "count": len(spans),
                            "utf8_start": span["utf8_start"],
                            "utf8_end": span["utf8_end"],
                            "segment_text_sha256": segment_text_sha256,
                        }
                        child_source = {**raw_source, "segment": segment}
                    child = {**raw, "text": span["text"], "source": child_source}
                    output.write(
                        json.dumps(child, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    child_record = {
                        "index": index,
                        "utf8_start": span["utf8_start"],
                        "utf8_end": span["utf8_end"],
                        "text_sha256": segment_text_sha256,
                        "end_boundary": span["end_boundary"],
                    }
                    records.append(child_record)
                    counters[f"end_boundary_{span['end_boundary']}"] += 1
                    child_identity.update(bytes.fromhex(canonical_sha256(child_record)))
                record = {
                    "schema": LINEAGE_SCHEMA,
                    "parent_row_id": parent_id,
                    "parent_text_sha256": parent_text_sha256,
                    "parent_utf8_bytes": len(text.encode()),
                    "segment_count": len(spans),
                    "segments": records,
                    "lossless_reconstruction_verified": True,
                }
                record["record_sha256"] = canonical_sha256(record)
                lineage.write(json.dumps(record, sort_keys=True) + "\n")
        if not counters["input_documents"]:
            raise StructuralSegmentationError("segmentation input is empty")
        if sha256_file(source_path) != source_sha256:
            raise StructuralSegmentationError("segmentation input changed")
        os.replace(output_stage, output_path)
        os.replace(lineage_stage, lineage_path)
    except BaseException:
        output_stage.unlink(missing_ok=True)
        lineage_stage.unlink(missing_ok=True)
        raise
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_representation",
        "source": {
            "path": str(source_path.resolve()),
            "bytes": source_path.stat().st_size,
            "sha256": source_sha256,
        },
        "policy": {
            "minimum_segment_bytes": minimum_bytes,
            "maximum_segment_bytes": maximum_bytes,
            "boundary_preference": [name for name, _ in _BOUNDARIES],
            "lossless_character_fallback": True,
            "normalization_or_rewriting_allowed": False,
        },
        "counts": dict(sorted(counters.items())),
        "parent_ordered_identity_sha256": parent_identity.hexdigest(),
        "child_ordered_identity_sha256": child_identity.hexdigest(),
        "output": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
        },
        "lineage": {
            "path": str(lineage_path.resolve()),
            "bytes": lineage_path.stat().st_size,
            "sha256": sha256_file(lineage_path),
            "contains_source_text": False,
        },
        "exact_parent_reconstruction_verified": True,
        "rights_and_attribution_inherited_from_parent": True,
        "benchmark_decontamination_complete": False,
        "global_deduplication_complete": False,
        "representation_verification_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_stage = receipt_path.with_name(f".{receipt_path.name}.tmp.{os.getpid()}")
    try:
        receipt_stage.write_text(json.dumps(payload, sort_keys=True) + "\n")
        os.replace(receipt_stage, receipt_path)
    except BaseException:
        receipt_stage.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        lineage_path.unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--minimum-bytes", type=int, default=DEFAULT_MINIMUM_BYTES)
    parser.add_argument("--maximum-bytes", type=int, default=DEFAULT_MAXIMUM_BYTES)
    args = parser.parse_args()
    result = build_segments(
        args.source,
        args.output,
        args.lineage,
        args.receipt,
        minimum_bytes=args.minimum_bytes,
        maximum_bytes=args.maximum_bytes,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
