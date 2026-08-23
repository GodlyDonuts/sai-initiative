"""Build and replay benchmark-disjoint Sai pretraining documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import multiprocessing
import os
import re
import stat
import unicodedata
from collections.abc import Callable, Container
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
_WORKER_WORD_BOUNDARY: Container[bytes] | None = None
_WORKER_CODE_BOUNDARY: Container[bytes] | None = None


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


def _overlap_count(tokens: list[str], width: int, boundary: Container[bytes]) -> int:
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


class SortedDigestBoundary:
    """Memory-map a strictly ordered fixed-width SHA-256 membership index."""

    def __init__(
        self, path: Path, *, expected_bytes: int, expected_sha256: str
    ) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError as error:
            raise DecontaminationError("boundary index is missing or unsafe") from error
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size != expected_bytes
                or metadata.st_size <= 0
                or metadata.st_size % 32
                or sha256_file(path) != expected_sha256
            ):
                raise DecontaminationError("boundary index bytes differ")
            self._mapping = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(descriptor)
        self._rows = expected_bytes // 32
        previous = None
        for index in range(self._rows):
            value = self._mapping[index * 32 : (index + 1) * 32]
            if previous is not None and value <= previous:
                self.close()
                raise DecontaminationError("boundary index ordering differs")
            previous = value

    def __contains__(self, value: object) -> bool:
        if not isinstance(value, bytes) or len(value) != 32:
            return False
        low = 0
        high = self._rows
        while low < high:
            middle = (low + high) // 2
            observed = self._mapping[middle * 32 : (middle + 1) * 32]
            if observed < value:
                low = middle + 1
            else:
                high = middle
        return low < self._rows and self._mapping[low * 32 : (low + 1) * 32] == value

    def close(self) -> None:
        self._mapping.close()


class CombinedDigestBoundary:
    """Present file and binary boundaries through one exact-membership surface."""

    def __init__(self, members: list[Container[bytes]]) -> None:
        if not members:
            raise DecontaminationError("at least one benchmark boundary is required")
        self.members = members

    def __contains__(self, value: object) -> bool:
        return any(value in member for member in self.members)


def binary_boundary_index(
    roots: list[Path],
) -> tuple[
    list[SortedDigestBoundary],
    list[SortedDigestBoundary],
    list[dict[str, Any]],
]:
    """Replay source-safe exact-shingle index receipts and memory-map their bytes."""

    words = []
    code = []
    receipts = []
    resolved = set()
    try:
        for order, root in enumerate(roots):
            absolute = str(root.resolve())
            receipt_path = root / "receipt.json"
            if (
                absolute in resolved
                or not root.is_dir()
                or root.is_symlink()
                or not receipt_path.is_file()
                or receipt_path.is_symlink()
            ):
                raise DecontaminationError("boundary index root is missing or unsafe")
            resolved.add(absolute)
            payload = json.loads(receipt_path.read_text())
            unsigned = {
                key: value for key, value in payload.items() if key != "receipt_sha256"
            }
            if (
                payload.get("schema") != "sai-official-benchmark-boundary-index-v1"
                or payload.get("status") != "complete"
                or payload.get("receipt_sha256") != canonical_sha256(unsigned)
                or payload.get("policy") != POLICY
                or payload.get("policy_sha256") != canonical_sha256(POLICY)
                or payload.get("benchmark_contamination_gate_ready") is not True
                or payload.get("raw_benchmark_text_persisted") is not False
            ):
                raise DecontaminationError("boundary index receipt differs")
            for key, target in (("word_index", words), ("code_index", code)):
                index = payload.get(key)
                if (
                    not isinstance(index, dict)
                    or index.get("digest_bytes") != 32
                    or not isinstance(index.get("file"), str)
                    or Path(index["file"]).name != index["file"]
                    or not isinstance(index.get("bytes"), int)
                    or not isinstance(index.get("unique_shingles"), int)
                    or index["bytes"] != index["unique_shingles"] * 32
                    or not isinstance(index.get("sha256"), str)
                ):
                    raise DecontaminationError("boundary index descriptor differs")
                target.append(
                    SortedDigestBoundary(
                        root / index["file"],
                        expected_bytes=index["bytes"],
                        expected_sha256=index["sha256"],
                    )
                )
            receipts.append(
                {
                    "order": order,
                    "boundary_index_root": absolute,
                    "receipt_sha256": payload["receipt_sha256"],
                    "receipt_file_sha256": sha256_file(receipt_path),
                    "word_index_sha256": payload["word_index"]["sha256"],
                    "code_index_sha256": payload["code_index"]["sha256"],
                }
            )
        return words, code, receipts
    except BaseException:
        for member in [*words, *code]:
            member.close()
        raise


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
    boundary_indexes: list[Path] | None = None,
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
    boundary_indexes = boundary_indexes or []
    if not boundaries and not boundary_indexes:
        raise DecontaminationError("at least one benchmark boundary is required")
    word_members: list[Container[bytes]] = []
    code_members: list[Container[bytes]] = []
    boundary_receipts = []
    if boundaries:
        file_words, file_code, file_receipts = boundary_index(boundaries)
        word_members.append(file_words)
        code_members.append(file_code)
        boundary_receipts.extend(file_receipts)
    binary_words, binary_code, binary_receipts = binary_boundary_index(boundary_indexes)
    word_members.extend(binary_words)
    code_members.extend(binary_code)
    boundary_receipts.extend(binary_receipts)
    words = CombinedDigestBoundary(word_members)
    code = CombinedDigestBoundary(code_members)
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
        for member in [*binary_words, *binary_code]:
            member.close()
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
    boundary_indexes: list[Path] | None = None,
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
                boundary_indexes=boundary_indexes,
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
    boundary_rows = payload.get("boundaries", [])
    boundaries = [
        Path(row["path"]) for row in boundary_rows if isinstance(row.get("path"), str)
    ]
    boundary_indexes = [
        Path(row["boundary_index_root"])
        for row in boundary_rows
        if isinstance(row.get("boundary_index_root"), str)
    ]
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

        metadata = _compute(
            source, boundaries, compare_row, boundary_indexes=boundary_indexes
        )
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
    build_parser.add_argument("--boundary", type=Path, action="append", default=[])
    build_parser.add_argument(
        "--boundary-index", type=Path, action="append", default=[]
    )
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
            boundary_indexes=args.boundary_index,
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
