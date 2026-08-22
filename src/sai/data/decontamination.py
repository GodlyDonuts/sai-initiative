"""Build and replay benchmark-disjoint Sai pretraining documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sai.data.token_stream import ROW_SCHEMA, canonical_sha256, sha256_file

RAW_SCHEMA = "sai-raw-pretraining-document-v1"
RECEIPT_SCHEMA = "sai-decontamination-receipt-v1"
POLICY = {
    "unicode_normalization": "NFKC_casefold",
    "word_shingle_tokens": 13,
    "code_shingle_tokens": 8,
    "minimum_boundary_string_characters": 8,
    "decision": "reject_on_any_exact_word_or_code_shingle_overlap",
}
_WORD = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_CODE = re.compile(
    r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|!=|<=|>=|:=|->|\*\*|//|<<|>>|&&|\|\||\S"
)
_WORKER_WORD_BOUNDARY: set[bytes] | None = None
_WORKER_CODE_BOUNDARY: set[bytes] | None = None


class DecontaminationError(RuntimeError):
    """The source, boundary, policy, output, or replay evidence differs."""


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return (
            [value]
            if len(value.strip()) >= POLICY["minimum_boundary_string_characters"]
            else []
        )
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    if isinstance(value, dict):
        return [text for key in sorted(value) for text in _strings(value[key])]
    return []


def _shingles(tokens: list[str], width: int) -> set[bytes]:
    if len(tokens) < width:
        return set()
    return {
        bytes.fromhex(canonical_sha256(tokens[index : index + width]))
        for index in range(len(tokens) - width + 1)
    }


def _overlap_count(tokens: list[str], width: int, boundary: set[bytes]) -> int:
    """Count unique matching shingles without retaining nonmatching source keys."""

    if len(tokens) < width:
        return 0
    matches = {
        digest
        for index in range(len(tokens) - width + 1)
        if (digest := bytes.fromhex(canonical_sha256(tokens[index : index + width])))
        in boundary
    }
    return len(matches)


def _text_shingles(text: str) -> tuple[set[bytes], set[bytes]]:
    normalized = _normalize(text)
    return (
        _shingles(_WORD.findall(normalized), POLICY["word_shingle_tokens"]),
        _shingles(_CODE.findall(normalized), POLICY["code_shingle_tokens"]),
    )


def boundary_index(
    paths: list[Path],
) -> tuple[set[bytes], set[bytes], list[dict[str, Any]]]:
    if not paths:
        raise DecontaminationError("at least one benchmark boundary is required")
    words: set[bytes] = set()
    code: set[bytes] = set()
    receipts = []
    resolved = set()
    for order, path in enumerate(paths):
        if not path.is_file() or path.is_symlink():
            raise DecontaminationError("benchmark boundary is missing or unsafe")
        absolute = str(path.resolve())
        if absolute in resolved:
            raise DecontaminationError("benchmark boundary paths are duplicated")
        resolved.add(absolute)
        rows = strings = 0
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DecontaminationError(
                        "benchmark JSONL is malformed"
                    ) from error
                for text in _strings(row):
                    strings += 1
                    word_shingles, code_shingles = _text_shingles(text)
                    words.update(word_shingles)
                    code.update(code_shingles)
        if not rows or not strings:
            raise DecontaminationError("benchmark boundary contains no usable text")
        receipts.append(
            {
                "order": order,
                "path": absolute,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "rows": rows,
                "strings": strings,
            }
        )
    if not words and not code:
        raise DecontaminationError("benchmark boundary produced no shingles")
    return words, code, receipts


def _raw_row(row: Any, *, source_path_sha256: str, line_number: int) -> dict[str, Any]:
    if not isinstance(row, dict) or row.get("schema") != RAW_SCHEMA:
        raise DecontaminationError("raw pretraining row schema differs")
    text = row.get("text")
    source = row.get("source")
    if (
        not isinstance(text, str)
        or not text
        or not isinstance(source, dict)
        or not isinstance(source.get("dataset"), str)
        or not source["dataset"]
        or not isinstance(source.get("revision"), str)
        or not source["revision"]
        or not isinstance(source.get("source_file"), str)
        or not source["source_file"]
        or not isinstance(source.get("row_index"), int)
        or isinstance(source.get("row_index"), bool)
        or source["row_index"] < 0
        or not isinstance(source.get("license"), str)
        or not source["license"]
        or source.get("domain")
        not in {"english", "code", "math", "science", "technical"}
    ):
        raise DecontaminationError("raw pretraining row provenance differs")
    identity = canonical_sha256(
        {
            "source_path_sha256": source_path_sha256,
            "line_number": line_number,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source": source,
        }
    )
    return {"text": text, "source": source, "raw_identity_sha256": identity}


def _candidate(
    item: tuple[int, str, str],
) -> tuple[dict[str, Any], bytes, int, int]:
    """Compute the expensive per-row evidence without changing source order."""

    line_number, line, source_sha256 = item
    if _WORKER_WORD_BOUNDARY is None or _WORKER_CODE_BOUNDARY is None:
        raise DecontaminationError("decontamination worker boundary is unavailable")
    try:
        raw = _raw_row(
            json.loads(line),
            source_path_sha256=source_sha256,
            line_number=line_number,
        )
    except json.JSONDecodeError as error:
        raise DecontaminationError("raw source JSONL is malformed") from error
    normalized_text = _normalize(raw["text"])
    normalized_text_sha256 = hashlib.sha256(normalized_text.encode()).digest()
    word_overlap_count = _overlap_count(
        _WORD.findall(normalized_text),
        POLICY["word_shingle_tokens"],
        _WORKER_WORD_BOUNDARY,
    )
    code_overlap_count = _overlap_count(
        _CODE.findall(normalized_text),
        POLICY["code_shingle_tokens"],
        _WORKER_CODE_BOUNDARY,
    )
    return raw, normalized_text_sha256, word_overlap_count, code_overlap_count


def _source_items(source: Path, source_sha256: str):
    with source.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, line, source_sha256


def _compute(
    source: Path,
    boundaries: list[Path],
    on_accepted: Callable[[dict[str, Any]], None],
    *,
    workers: int = 1,
) -> dict[str, Any]:
    if (
        not source.is_file()
        or source.is_symlink()
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 64
    ):
        raise DecontaminationError("raw source is missing or unsafe")
    source_sha256 = sha256_file(source)
    words, code, boundary_receipts = boundary_index(boundaries)
    boundary_manifest_sha256 = canonical_sha256(boundary_receipts)
    policy_sha256 = canonical_sha256(POLICY)
    seen_text: set[bytes] = set()
    accepted_identity_digest = hashlib.sha256()
    dropped_evidence_digest = hashlib.sha256()
    scanned = accepted = dropped = 0
    global _WORKER_WORD_BOUNDARY, _WORKER_CODE_BOUNDARY
    _WORKER_WORD_BOUNDARY = words
    _WORKER_CODE_BOUNDARY = code
    items = _source_items(source, source_sha256)
    pool = None
    try:
        if workers == 1:
            candidates = map(_candidate, items)
        else:
            if os.name != "posix":
                raise DecontaminationError(
                    "parallel decontamination requires a POSIX fork runtime"
                )
            context = multiprocessing.get_context("fork")
            pool = context.Pool(processes=workers)
            candidates = pool.imap(_candidate, items, chunksize=64)
        for (
            raw,
            normalized_text_sha256,
            word_overlap_count,
            code_overlap_count,
        ) in candidates:
            scanned += 1
            duplicate = normalized_text_sha256 in seen_text
            seen_text.add(normalized_text_sha256)
            decision = {
                "raw_identity_sha256": raw["raw_identity_sha256"],
                "boundary_manifest_sha256": boundary_manifest_sha256,
                "policy_sha256": policy_sha256,
                "word_overlap_count": word_overlap_count,
                "code_overlap_count": code_overlap_count,
                "normalized_exact_duplicate": duplicate,
            }
            evidence_sha256 = canonical_sha256(decision)
            if word_overlap_count or code_overlap_count or duplicate:
                dropped += 1
                dropped_evidence_digest.update(bytes.fromhex(evidence_sha256))
                continue
            source_row = raw["source"]
            output = {
                "schema": ROW_SCHEMA,
                "text": raw["text"],
                "source": {
                    "dataset": source_row["dataset"],
                    "row_id": canonical_sha256(
                        {
                            "dataset": source_row["dataset"],
                            "revision": source_row["revision"],
                            "source_file": source_row["source_file"],
                            "row_index": source_row["row_index"],
                        }
                    ),
                    "license": source_row["license"],
                    "domain": source_row["domain"],
                },
                "verification": {
                    "benchmark_disjoint": True,
                    "evidence_sha256": evidence_sha256,
                },
            }
            identity = canonical_sha256(output)
            accepted += 1
            accepted_identity_digest.update(bytes.fromhex(identity))
            on_accepted({**output, "identity_sha256": identity})
        if pool is not None:
            pool.close()
            pool.join()
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    finally:
        _WORKER_WORD_BOUNDARY = None
        _WORKER_CODE_BOUNDARY = None
    if not scanned or not accepted:
        raise DecontaminationError("decontamination admitted no documents")
    return {
        "source": {
            "path": str(source.resolve()),
            "bytes": source.stat().st_size,
            "sha256": source_sha256,
        },
        "boundaries": boundary_receipts,
        "boundary_manifest_sha256": boundary_manifest_sha256,
        "policy": POLICY,
        "policy_sha256": policy_sha256,
        "scanned": scanned,
        "accepted": accepted,
        "dropped": dropped,
        "identity_accumulation": "ordered_raw_sha256_bytes",
        "accepted_identity_sha256": accepted_identity_digest.hexdigest(),
        "dropped_evidence_sha256": dropped_evidence_digest.hexdigest(),
    }


def build(
    source: Path,
    boundaries: list[Path],
    output: Path,
    receipt: Path,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    if output.exists() or receipt.exists():
        raise DecontaminationError("decontamination output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if stage.exists():
        raise DecontaminationError("decontamination staging output already exists")
    try:
        with stage.open("w") as output_handle:
            metadata = _compute(
                source,
                boundaries,
                lambda row: output_handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                ),
                workers=workers,
            )
    except BaseException:
        stage.unlink(missing_ok=True)
        raise
    output_sha256 = sha256_file(stage)
    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        **metadata,
        "output": {
            "path": str(output.resolve()),
            "bytes": stage.stat().st_size,
            "sha256": output_sha256,
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_stage = receipt.with_name(f".{receipt.name}.partial.{os.getpid()}")
    receipt_stage.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    os.replace(stage, output)
    os.replace(receipt_stage, receipt)
    return payload


def validate(receipt: Path) -> dict[str, Any]:
    if not receipt.is_file() or receipt.is_symlink():
        raise DecontaminationError("decontamination receipt is missing or unsafe")
    payload = json.loads(receipt.read_text())
    if not isinstance(payload, dict) or payload.get("schema") != RECEIPT_SCHEMA:
        raise DecontaminationError("decontamination receipt schema differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("receipt_sha256") != canonical_sha256(unsigned):
        raise DecontaminationError("decontamination receipt hash differs")
    source = Path(payload.get("source", {}).get("path", ""))
    boundaries = [Path(row.get("path", "")) for row in payload.get("boundaries", [])]
    output = Path(payload.get("output", {}).get("path", ""))
    if (
        not output.is_file()
        or output.is_symlink()
        or payload.get("output", {}).get("path") != str(output.resolve())
        or payload.get("output", {}).get("bytes") != output.stat().st_size
        or payload.get("output", {}).get("sha256") != sha256_file(output)
    ):
        raise DecontaminationError("decontaminated output differs")
    with output.open() as output_handle:

        def compare_row(row: dict[str, Any]) -> None:
            observed = output_handle.readline()
            expected = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            if observed != expected:
                raise DecontaminationError("decontaminated output differs")

        metadata = _compute(source, boundaries, compare_row)
        if output_handle.read(1):
            raise DecontaminationError("decontaminated output differs")
    for key, value in metadata.items():
        if payload.get(key) != value:
            raise DecontaminationError("decontamination replay differs")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source", type=Path, required=True)
    build_parser.add_argument("--boundary", type=Path, action="append", required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--receipt", type=Path, required=True)
    build_parser.add_argument("--workers", type=int, default=1)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        payload = build(
            args.source,
            args.boundary,
            args.output,
            args.receipt,
            workers=args.workers,
        )
    else:
        payload = validate(args.receipt)
    print(
        json.dumps(
            {"status": payload["status"], "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
