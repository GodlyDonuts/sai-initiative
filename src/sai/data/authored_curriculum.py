"""Build a provenance-bound, source-ordered authored curriculum candidate.

The output is deliberately not a pretraining-document population.  It preserves
the publisher's chapter bytes and ordering so semantic review, licensing review,
deduplication, and benchmark decontamination can happen before admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tarfile
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-authored-curriculum-receipt-v1"
ROW_SCHEMA = "sai-authored-curriculum-candidate-v1"
RUST_SOURCE = "rust-book"
PYTHON_SOURCE = "cpython-tutorial"
_RUST_LINK = re.compile(
    r"^(?P<indent>\s*)-?\s*\[(?P<title>[^]]+)\]\((?P<path>[^)]+\.md)\)\s*$"
)
_RST_ENTRY = re.compile(r"^\s{3}(?P<path>[A-Za-z0-9_.-]+\.rst)\s*$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "policy",
    "sources",
    "summary",
    "output",
    "limitations",
    "receipt_sha256",
}
_ROW_KEYS = {
    "schema",
    "identity_sha256",
    "source_name",
    "source_revision",
    "source_path",
    "source_order_index",
    "source_hierarchy_depth",
    "candidate_stage",
    "required_prior_concepts",
    "title",
    "license_identifiers",
    "source_bytes",
    "source_sha256",
    "text",
    "admission_status",
}


class AuthoredCurriculumError(RuntimeError):
    """The authored source or its declared progression is unsafe or differs."""


@dataclass(frozen=True)
class _Chapter:
    source_path: str
    title: str
    depth: int
    stage: int
    required_prior_concepts: tuple[str, ...]


def _safe_member_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AuthoredCurriculumError("archive contains an unsafe path")
    return path


def _regular_members(
    archive: Path,
) -> tuple[tarfile.TarFile, dict[str, tarfile.TarInfo]]:
    try:
        handle = tarfile.open(archive, "r:gz")
    except (OSError, tarfile.TarError) as error:
        raise AuthoredCurriculumError("authored archive is unreadable") from error
    members: dict[str, tarfile.TarInfo] = {}
    try:
        for member in handle.getmembers():
            _safe_member_name(member.name)
            if member.name in members:
                raise AuthoredCurriculumError("archive contains duplicate paths")
            members[member.name] = member
    except Exception:
        handle.close()
        raise
    return handle, members


def _read_member(
    handle: tarfile.TarFile, members: dict[str, tarfile.TarInfo], name: str
) -> bytes:
    member = members.get(name)
    if member is None or not member.isfile() or member.issym() or member.islnk():
        raise AuthoredCurriculumError("required archive member is missing or unsafe")
    extracted = handle.extractfile(member)
    if extracted is None:
        raise AuthoredCurriculumError("required archive member is unreadable")
    encoded = extracted.read()
    if len(encoded) != member.size:
        raise AuthoredCurriculumError("required archive member changed while reading")
    try:
        encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuthoredCurriculumError("authored chapter is not UTF-8") from error
    return encoded


def _single_prefix(members: Iterable[str], marker: str) -> str:
    prefixes = {name.split("/", 1)[0] for name in members if name.endswith(marker)}
    if len(prefixes) != 1:
        raise AuthoredCurriculumError("archive root differs")
    return prefixes.pop()


def _rust_stage(path: str) -> int:
    if path in {"title-page.md", "foreword.md"} or path.startswith(
        ("ch00-", "ch01-", "ch02-", "ch03-")
    ):
        return 0
    if path.startswith(("ch04-", "ch05-", "ch06-", "ch07-", "ch08-", "ch09-")):
        return 1
    if path.startswith(("ch10-", "ch11-", "ch12-", "ch13-", "ch14-")):
        return 2
    return 3


def _parse_rust_summary(text: str) -> list[_Chapter]:
    chapters: list[_Chapter] = []
    seen: set[str] = set()
    for line in text.splitlines():
        match = _RUST_LINK.match(line)
        if match is None:
            continue
        path = match.group("path")
        if "://" in path or path.startswith("/") or path in seen:
            raise AuthoredCurriculumError("Rust summary path differs")
        seen.add(path)
        indent = len(match.group("indent"))
        chapters.append(
            _Chapter(
                source_path=f"src/{path}",
                title=match.group("title"),
                depth=indent // 2,
                stage=_rust_stage(path),
                required_prior_concepts=(),
            )
        )
    if len(chapters) < 80 or chapters[0].source_path != "src/title-page.md":
        raise AuthoredCurriculumError("Rust summary is incomplete")
    return chapters


def _parse_python_tutorial(text: str) -> list[_Chapter]:
    chapters: list[_Chapter] = []
    in_toctree = False
    for line in text.splitlines():
        if line.strip() == ".. toctree::":
            if in_toctree:
                raise AuthoredCurriculumError(
                    "Python tutorial has multiple root toctrees"
                )
            in_toctree = True
            continue
        if not in_toctree:
            continue
        match = _RST_ENTRY.match(line)
        if match:
            path = match.group("path")
            chapters.append(
                _Chapter(
                    source_path=f"Doc/tutorial/{path}",
                    title=PurePosixPath(path).stem.replace("-", " ").replace("_", " "),
                    depth=0,
                    stage=1 if len(chapters) < 8 else 2,
                    required_prior_concepts=("programming_foundations",),
                )
            )
        elif chapters and line.strip() and not line.startswith(" "):
            break
    if len(chapters) != 16 or chapters[0].source_path != "Doc/tutorial/appetite.rst":
        raise AuthoredCurriculumError("Python tutorial order differs")
    return chapters


def _rst_title(text: str, fallback: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines[:-1]):
        value = line.strip()
        underline = lines[index + 1].strip()
        if (
            value
            and len(underline) >= len(value)
            and set(underline) <= set('#*=-^"~`:+')
        ):
            return value
    return fallback


def _rows_for_source(
    *,
    archive: Path,
    revision: str,
    source_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not _HEX40.fullmatch(revision):
        raise AuthoredCurriculumError("source revision differs")
    handle, members = _regular_members(archive)
    try:
        if source_name == RUST_SOURCE:
            root = _single_prefix(members, "/src/SUMMARY.md")
            summary_name = f"{root}/src/SUMMARY.md"
            chapters = _parse_rust_summary(
                _read_member(handle, members, summary_name).decode()
            )
            licenses = [f"{root}/LICENSE-APACHE", f"{root}/LICENSE-MIT"]
            license_ids = ["Apache-2.0", "MIT"]
        elif source_name == PYTHON_SOURCE:
            root = _single_prefix(members, "/Doc/tutorial/index.rst")
            summary_name = f"{root}/Doc/tutorial/index.rst"
            chapters = _parse_python_tutorial(
                _read_member(handle, members, summary_name).decode()
            )
            licenses = [f"{root}/LICENSE"]
            license_ids = ["PSF-2.0"]
        else:
            raise AuthoredCurriculumError("authored source name differs")
        license_records = []
        for name, identifier in zip(licenses, license_ids, strict=True):
            encoded = _read_member(handle, members, name)
            license_records.append(
                {
                    "spdx_identifier": identifier,
                    "archive_path": name.removeprefix(f"{root}/"),
                    "bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
        rows = []
        for order_index, chapter in enumerate(chapters):
            archive_path = f"{root}/{chapter.source_path}"
            encoded = _read_member(handle, members, archive_path)
            text = encoded.decode()
            title = (
                _rst_title(text, chapter.title)
                if source_name == PYTHON_SOURCE
                else chapter.title
            )
            identity = canonical_sha256(
                {
                    "source_name": source_name,
                    "revision": revision,
                    "source_path": chapter.source_path,
                    "source_sha256": hashlib.sha256(encoded).hexdigest(),
                    "order_index": order_index,
                }
            )
            rows.append(
                {
                    "schema": ROW_SCHEMA,
                    "identity_sha256": identity,
                    "source_name": source_name,
                    "source_revision": revision,
                    "source_path": chapter.source_path,
                    "source_order_index": order_index,
                    "source_hierarchy_depth": chapter.depth,
                    "candidate_stage": chapter.stage,
                    "required_prior_concepts": list(chapter.required_prior_concepts),
                    "title": title,
                    "license_identifiers": license_ids,
                    "source_bytes": len(encoded),
                    "source_sha256": hashlib.sha256(encoded).hexdigest(),
                    "text": text,
                    "admission_status": "candidate_only",
                }
            )
        archive_record = {
            "source_name": source_name,
            "revision": revision,
            "archive_file": archive.name,
            "archive_bytes": archive.stat().st_size,
            "archive_sha256": sha256_file(archive),
            "order_manifest_path": summary_name.removeprefix(f"{root}/"),
            "order_manifest_sha256": hashlib.sha256(
                _read_member(handle, members, summary_name)
            ).hexdigest(),
            "licenses": license_records,
            "chapter_count": len(rows),
        }
        return rows, archive_record
    finally:
        handle.close()


def _write_create_only(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise AuthoredCurriculumError("output already exists") from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_regular_bytes(path: Path, *, maximum_bytes: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise AuthoredCurriculumError(
            "authored artifact is missing or unsafe"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise AuthoredCurriculumError("authored artifact is missing or unsafe")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
    )
    if len(encoded) != before.st_size or before_identity != after_identity:
        raise AuthoredCurriculumError("authored artifact changed while reading")
    return encoded


def build(
    *,
    rust_archive: Path,
    rust_revision: str,
    python_archive: Path,
    python_revision: str,
    output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Build an exact source-ordered candidate and its fail-closed receipt."""

    if (
        output.exists()
        or receipt_output.exists()
        or output.resolve() == receipt_output.resolve()
    ):
        raise AuthoredCurriculumError("authored output boundary differs")
    rust_rows, rust_source = _rows_for_source(
        archive=rust_archive, revision=rust_revision, source_name=RUST_SOURCE
    )
    python_rows, python_source = _rows_for_source(
        archive=python_archive, revision=python_revision, source_name=PYTHON_SOURCE
    )
    rows = rust_rows + python_rows
    encoded = b"".join(
        json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
        for row in rows
    )
    output_sha256 = hashlib.sha256(encoded).hexdigest()
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "candidate_only",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "policy": {
            "preserve_publisher_chapter_order": True,
            "preserve_exact_chapter_bytes": True,
            "cross_chapter_chunking": "forbidden",
            "rust_stage_assignment": "publisher_order_chapter_bands_candidate_only",
            "python_tutorial_requires_prior": "programming_foundations",
            "python_reference_and_library_excluded": True,
            "human_semantic_review_required": True,
            "license_review_required": True,
            "near_deduplication_required": True,
            "benchmark_decontamination_required": True,
        },
        "sources": [rust_source, python_source],
        "summary": {
            "rows": len(rows),
            "source_bytes": sum(row["source_bytes"] for row in rows),
            "stage_counts": {
                str(stage): sum(row["candidate_stage"] == stage for row in rows)
                for stage in range(4)
            },
            "python_rows_with_programming_foundations_prerequisite": len(python_rows),
        },
        "output": {
            "file": output.name,
            "bytes": len(encoded),
            "rows": len(rows),
            "sha256": output_sha256,
        },
        "limitations": [
            "candidate_rows_are_not_pretraining_documents",
            "candidate_stages_are_not_semantically_qualified",
            "no_benchmark_decontamination_has_been_applied",
            "no_training_or_architecture_promotion_is_authorized",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    receipt_encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False).encode()
        + b"\n"
    )
    _write_create_only(output, encoded)
    try:
        _write_create_only(receipt_output, receipt_encoded)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return payload


def validate(output: Path, receipt_output: Path) -> dict[str, Any]:
    """Replay the published candidate's identities, order, hashes, and holds."""

    try:
        receipt_encoded = _read_regular_bytes(receipt_output, maximum_bytes=1 << 20)
        output_encoded = _read_regular_bytes(output, maximum_bytes=1 << 30)
        payload = json.loads(receipt_encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthoredCurriculumError("authored receipt is unreadable") from error
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        set(payload) != _TOP_KEYS
        or payload.get("schema") != SCHEMA
        or payload.get("status") != "candidate_only"
        or payload.get("training_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or payload.get("output", {}).get("sha256")
        != hashlib.sha256(output_encoded).hexdigest()
        or payload.get("output", {}).get("bytes") != len(output_encoded)
    ):
        raise AuthoredCurriculumError("authored receipt differs")
    try:
        rows = [json.loads(line) for line in output_encoded.decode().splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthoredCurriculumError("authored rows are unreadable") from error
    if (
        len(rows) != payload["output"]["rows"]
        or len(rows) != payload["summary"]["rows"]
    ):
        raise AuthoredCurriculumError("authored rows differ")
    source_records = payload.get("sources")
    if not isinstance(source_records, list) or [
        record.get("source_name") for record in source_records
    ] != [RUST_SOURCE, PYTHON_SOURCE]:
        raise AuthoredCurriculumError("authored sources differ")
    revisions = {record["source_name"]: record["revision"] for record in source_records}
    license_ids = {
        record["source_name"]: [
            license_["spdx_identifier"] for license_ in record["licenses"]
        ]
        for record in source_records
    }
    offsets: dict[str, int] = {RUST_SOURCE: 0, PYTHON_SOURCE: 0}
    seen_paths: set[tuple[str, str]] = set()
    seen_python = False
    for row in rows:
        source = row.get("source_name")
        text = row.get("text")
        path = row.get("source_path")
        if source == PYTHON_SOURCE:
            seen_python = True
        if (
            set(row) != _ROW_KEYS
            or row.get("schema") != ROW_SCHEMA
            or source not in offsets
            or (seen_python and source == RUST_SOURCE)
            or row.get("source_order_index") != offsets[source]
            or row.get("source_revision") != revisions[source]
            or not isinstance(path, str)
            or (source, path) in seen_paths
            or not isinstance(text, str)
            or row.get("source_bytes") != len(text.encode())
            or row.get("source_sha256") != hashlib.sha256(text.encode()).hexdigest()
            or row.get("license_identifiers") != license_ids[source]
            or row.get("candidate_stage") not in range(4)
            or row.get("admission_status") != "candidate_only"
        ):
            raise AuthoredCurriculumError("authored row differs")
        expected_prior = ["programming_foundations"] if source == PYTHON_SOURCE else []
        expected_identity = canonical_sha256(
            {
                "source_name": source,
                "revision": revisions[source],
                "source_path": path,
                "source_sha256": row["source_sha256"],
                "order_index": offsets[source],
            }
        )
        if (
            row.get("required_prior_concepts") != expected_prior
            or row.get("identity_sha256") != expected_identity
        ):
            raise AuthoredCurriculumError("authored row identity differs")
        seen_paths.add((source, path))
        offsets[source] += 1
    expected_counts = {
        record["source_name"]: record["chapter_count"] for record in source_records
    }
    stage_counts = {
        str(stage): sum(row["candidate_stage"] == stage for row in rows)
        for stage in range(4)
    }
    if (
        offsets != expected_counts
        or payload["summary"].get("source_bytes")
        != sum(row["source_bytes"] for row in rows)
        or payload["summary"].get("stage_counts") != stage_counts
        or payload["summary"].get(
            "python_rows_with_programming_foundations_prerequisite"
        )
        != offsets[PYTHON_SOURCE]
    ):
        raise AuthoredCurriculumError("authored summary differs")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--rust-archive", type=Path, required=True)
    build_parser.add_argument("--rust-revision", required=True)
    build_parser.add_argument("--python-archive", type=Path, required=True)
    build_parser.add_argument("--python-revision", required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--receipt-output", type=Path, required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, required=True)
    validate_parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        payload = build(
            rust_archive=args.rust_archive,
            rust_revision=args.rust_revision,
            python_archive=args.python_archive,
            python_revision=args.python_revision,
            output=args.output,
            receipt_output=args.receipt_output,
        )
    else:
        payload = validate(args.output, args.receipt_output)
    print(
        json.dumps(
            {"status": payload["status"], "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
