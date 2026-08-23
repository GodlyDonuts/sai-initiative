"""Build a host-diverse, benchmark-clean OpenCoder code-web teacher audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sai.data.benchmark_contamination_screen import _Union
from sai.data.decontamination import (
    _CODE,
    _WORD,
    _code_overlap_count,
    _normalize,
    _overlap_count,
    binary_boundary_index,
)
from sai.data.decontamination import (
    POLICY as DECONTAMINATION_POLICY,
)
from sai.data.reservoir_audit_population import (
    SCHEMA,
    _candidate_and_lineage,
    _write_jsonl,
)
from sai.data.source_quality_gate import (
    POLICY_SHA256 as MECHANICAL_POLICY_SHA256,
)
from sai.data.source_quality_gate import mechanical_quality_evidence
from sai.data.token_stream import canonical_sha256, sha256_file

REPOSITORY = "OpenCoder-LLM/opc-fineweb-code-corpus"
REVISION = "9e8e48e666c226294d6f9e6c2e13f2c84c1c06f3"
SOURCE_MEMBER = "data/train-00503-of-00510.parquet"
SOURCE_BYTES = 286_437_437
SOURCE_SHA256 = "b10da422ed33595620f3b375fd697a70d16829ae38b9cebbfad29eacad501eb7"
SOURCE_ROWS = 197_882
SOURCE_LICENSE = "MIT"
SOURCE_CARD = "README.md"
SOURCE_CARD_BYTES = 2_791
SOURCE_CARD_SHA256 = "f3bb986ed02b934b222589876d7490477a195bf8b635693693db35939abd8e34"
SEED = 20260826
TARGET_ROWS = 2_048
SCREEN_ROWS = 8_192
MINIMUM_TEXT_BYTES = 512
MAXIMUM_TEXT_BYTES = 512 * 1024
MAXIMUM_SCREEN_ROWS_PER_HOST = 4
MAXIMUM_SELECTED_ROWS_PER_HOST = 2
EXPECTED_COLUMNS = (
    "url",
    "tag",
    "text",
    "file_path",
    "dump",
    "file_size_in_byte",
    "line_count",
)


class OpenCoderCodeAuditPopulationError(RuntimeError):
    """The pinned source, selection, quality, or boundary replay differs."""


def _host(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        result = (urlsplit(value).hostname or "").strip(".").lower()
    except ValueError:
        return None
    return result or None


def _selection_key(
    *, row_index: int, host: str, content_sha256: str, url_sha256: str
) -> str:
    return hashlib.sha256(
        (
            f"{SEED}:{REPOSITORY}:{REVISION}:{SOURCE_MEMBER}:"
            f"{row_index}:{host}:{content_sha256}:{url_sha256}"
        ).encode()
    ).hexdigest()


def select_screen_rows(
    metadata: list[dict[str, Any]],
    *,
    screen_rows: int = SCREEN_ROWS,
    maximum_rows_per_host: int = MAXIMUM_SCREEN_ROWS_PER_HOST,
) -> list[dict[str, Any]]:
    """Select exact-content-unique rows with a strict per-host ceiling."""

    if (
        isinstance(screen_rows, bool)
        or not isinstance(screen_rows, int)
        or screen_rows <= 0
        or isinstance(maximum_rows_per_host, bool)
        or not isinstance(maximum_rows_per_host, int)
        or maximum_rows_per_host <= 0
    ):
        raise OpenCoderCodeAuditPopulationError("code audit selection differs")
    by_content: dict[str, dict[str, Any]] = {}
    for row in metadata:
        if not isinstance(row, dict):
            raise OpenCoderCodeAuditPopulationError("code audit metadata differs")
        content = row.get("content_sha256")
        key = row.get("selection_key")
        host = row.get("host")
        if (
            not isinstance(content, str)
            or len(content) != 64
            or not isinstance(key, str)
            or len(key) != 64
            or not isinstance(host, str)
            or not host
        ):
            raise OpenCoderCodeAuditPopulationError("code audit metadata differs")
        previous = by_content.get(content)
        if previous is None or (key, row["row_index"]) < (
            previous["selection_key"],
            previous["row_index"],
        ):
            by_content[content] = row
    selected = []
    host_counts: Counter[str] = Counter()
    for row in sorted(
        by_content.values(), key=lambda item: (item["selection_key"], item["row_index"])
    ):
        if host_counts[row["host"]] >= maximum_rows_per_host:
            continue
        selected.append(row)
        host_counts[row["host"]] += 1
        if len(selected) == screen_rows:
            break
    if len(selected) != screen_rows:
        raise OpenCoderCodeAuditPopulationError(
            f"code audit screen underfilled: {len(selected)} of {screen_rows}"
        )
    return selected


def select_final_rows(
    rows: list[dict[str, Any]],
    *,
    target_rows: int = TARGET_ROWS,
    maximum_rows_per_host: int = MAXIMUM_SELECTED_ROWS_PER_HOST,
) -> list[dict[str, Any]]:
    """Freeze the clean mechanical-pass rows under a tighter host ceiling."""

    if (
        isinstance(target_rows, bool)
        or not isinstance(target_rows, int)
        or target_rows <= 0
        or isinstance(maximum_rows_per_host, bool)
        or not isinstance(maximum_rows_per_host, int)
        or maximum_rows_per_host <= 0
    ):
        raise OpenCoderCodeAuditPopulationError("code audit final geometry differs")
    selected = []
    hosts: Counter[str] = Counter()
    contents = set()
    for row in sorted(
        rows, key=lambda item: (item["selection_key"], item["row_index"])
    ):
        if (
            row["content_sha256"] in contents
            or hosts[row["host"]] >= maximum_rows_per_host
        ):
            continue
        selected.append(row)
        contents.add(row["content_sha256"])
        hosts[row["host"]] += 1
        if len(selected) == target_rows:
            break
    if len(selected) != target_rows:
        raise OpenCoderCodeAuditPopulationError(
            f"code audit population underfilled: {len(selected)} of {target_rows}"
        )
    return selected


def _scan_metadata(
    source_path: Path,
) -> tuple[list[dict[str, Any]], dict[int, tuple[int, int]]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - dependency declared by workspace
        raise OpenCoderCodeAuditPopulationError("pyarrow is required") from error
    source = parquet.ParquetFile(source_path)
    if (
        tuple(source.schema_arrow.names) != EXPECTED_COLUMNS
        or source.metadata.num_rows != SOURCE_ROWS
        or source.num_row_groups <= 0
    ):
        raise OpenCoderCodeAuditPopulationError("code audit parquet schema differs")
    metadata = []
    locations: dict[int, tuple[int, int]] = {}
    row_index = 0
    for group_index in range(source.num_row_groups):
        table = source.read_row_group(
            group_index,
            columns=["url", "tag", "text", "file_path"],
            use_threads=False,
        )
        for row_in_group in range(table.num_rows):
            url = table["url"][row_in_group].as_py()
            tag = table["tag"][row_in_group].as_py()
            text = table["text"][row_in_group].as_py()
            file_path = table["file_path"][row_in_group].as_py()
            encoded = text.strip().encode() if isinstance(text, str) else b""
            host = _host(url)
            if (
                tag == "code"
                and host is not None
                and isinstance(file_path, str)
                and file_path
                and MINIMUM_TEXT_BYTES <= len(encoded) <= MAXIMUM_TEXT_BYTES
            ):
                content_sha256 = hashlib.sha256(encoded).hexdigest()
                url_sha256 = hashlib.sha256(url.encode()).hexdigest()
                metadata.append(
                    {
                        "row_index": row_index,
                        "host": host,
                        "content_sha256": content_sha256,
                        "url_sha256": url_sha256,
                        "file_path_sha256": hashlib.sha256(
                            file_path.encode()
                        ).hexdigest(),
                        "full_text_bytes": len(encoded),
                        "selection_key": _selection_key(
                            row_index=row_index,
                            host=host,
                            content_sha256=content_sha256,
                            url_sha256=url_sha256,
                        ),
                    }
                )
                locations[row_index] = (group_index, row_in_group)
            row_index += 1
    if row_index != SOURCE_ROWS:
        raise OpenCoderCodeAuditPopulationError("code audit source coverage differs")
    return metadata, locations


def _load_selected_texts(
    source_path: Path,
    selected: list[dict[str, Any]],
    locations: dict[int, tuple[int, int]],
) -> dict[int, str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover
        raise OpenCoderCodeAuditPopulationError("pyarrow is required") from error
    by_group: dict[int, list[tuple[int, int]]] = {}
    for row in selected:
        row_index = row["row_index"]
        if row_index not in locations:
            raise OpenCoderCodeAuditPopulationError("code audit locator differs")
        group, in_group = locations[row_index]
        by_group.setdefault(group, []).append((row_index, in_group))
    result = {}
    selected_by_index = {row["row_index"]: row for row in selected}
    source = parquet.ParquetFile(source_path)
    for group_index, members in sorted(by_group.items()):
        table = source.read_row_group(group_index, columns=["text"], use_threads=False)
        for row_index, row_in_group in members:
            value = table["text"][row_in_group].as_py()
            if not isinstance(value, str):
                raise OpenCoderCodeAuditPopulationError("code audit text differs")
            text = value.strip()
            metadata = selected_by_index[row_index]
            if (
                hashlib.sha256(text.encode()).hexdigest() != metadata["content_sha256"]
                or len(text.encode()) != metadata["full_text_bytes"]
            ):
                raise OpenCoderCodeAuditPopulationError("code audit content differs")
            result[row_index] = text
    if len(result) != len(selected):
        raise OpenCoderCodeAuditPopulationError("code audit text coverage differs")
    return result


def _candidate(
    row: dict[str, Any],
    text: str,
    *,
    ordinal: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    group_index = row["row_group"]
    row_in_group = row["row_in_group"]
    plan = {
        "ordinal": ordinal,
        "source_id": "opencoder_fineweb_code",
        "stratum": f"host:{row['host']}",
        "source_type": "educational_web",
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": SOURCE_LICENSE,
        "access": "public",
        "path": SOURCE_MEMBER,
        "parent_file_bytes": SOURCE_BYTES,
        "parent_file_sha256": SOURCE_SHA256,
        "selection_key": row["selection_key"],
        "text_column": "text",
    }
    acquired = {
        "text": text,
        "locator": {
            "format": "parquet",
            "row_group": group_index,
            "row_in_group": row_in_group,
            "row_index": row["row_index"],
            "url_sha256": row["url_sha256"],
            "file_path_sha256": row["file_path_sha256"],
        },
        "full_file_content_verified": True,
    }
    return _candidate_and_lineage(plan, acquired)


def build_population(
    source_path: Path,
    source_card_path: Path,
    boundary_roots: list[Path],
    output_root: Path,
    *,
    target_rows: int = TARGET_ROWS,
    screen_rows: int = SCREEN_ROWS,
) -> dict[str, Any]:
    """Verify one full shard and freeze a clean, host-diverse code-web audit."""

    source_path = source_path.resolve()
    source_card_path = source_card_path.resolve()
    if (
        output_root.exists()
        or output_root.is_symlink()
        or not source_path.is_file()
        or source_path.stat().st_size != SOURCE_BYTES
        or sha256_file(source_path) != SOURCE_SHA256
        or not source_card_path.is_file()
        or source_card_path.stat().st_size != SOURCE_CARD_BYTES
        or sha256_file(source_card_path) != SOURCE_CARD_SHA256
        or not boundary_roots
    ):
        raise OpenCoderCodeAuditPopulationError("code audit input boundary differs")
    metadata, locations = _scan_metadata(source_path)
    screen = select_screen_rows(metadata, screen_rows=screen_rows)
    texts = _load_selected_texts(source_path, screen, locations)
    words, code, boundary_receipts = binary_boundary_index(boundary_roots)
    clean = []
    rejection_counts: Counter[str] = Counter()
    try:
        word_boundary = words[0] if len(words) == 1 else _Union(words)
        code_boundary = code[0] if len(code) == 1 else _Union(code)
        for row in screen:
            group, in_group = locations[row["row_index"]]
            row = {**row, "row_group": group, "row_in_group": in_group}
            candidate, _lineage = _candidate(row, texts[row["row_index"]], ordinal=0)
            mechanical = mechanical_quality_evidence(candidate["text"])
            normalized = _normalize(candidate["text"])
            word_overlaps = _overlap_count(
                _WORD.findall(normalized),
                DECONTAMINATION_POLICY["word_shingle_tokens"],
                word_boundary,
            )
            code_overlaps = _code_overlap_count(
                _CODE.findall(normalized), code_boundary
            )
            if mechanical["decision"] != "pass_mechanical_gate":
                rejection_counts[mechanical["decision"]] += 1
            elif word_overlaps or code_overlaps:
                rejection_counts["benchmark_contaminated"] += 1
            else:
                clean.append(row)
    finally:
        for member in [*words, *code]:
            member.close()
    final = select_final_rows(clean, target_rows=target_rows)
    candidates = []
    lineage = []
    for ordinal, row in enumerate(final):
        candidate, source = _candidate(row, texts[row["row_index"]], ordinal=ordinal)
        candidates.append(candidate)
        lineage.append(source)
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if (
        len(identities) != target_rows
        or len(identities) != len(set(identities))
        or len({row["source_content_sha256"] for row in candidates}) != target_rows
    ):
        raise OpenCoderCodeAuditPopulationError("code audit identities differ")
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise OpenCoderCodeAuditPopulationError("code audit temporary output differs")
    temporary.mkdir(parents=True)
    try:
        candidates_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        _write_jsonl(candidates_path, candidates)
        _write_jsonl(lineage_path, lineage)
        host_counts = Counter(row["host"] for row in final)
        payload = {
            "schema": SCHEMA,
            "status": "complete",
            "seed": SEED,
            "selection_method": (
                "full_pinned_parquet_replay_content_deduplication_lowest_sha256_"
                "host_diversity_mechanical_gate_and_official_boundary_screen"
            ),
            "statistically_representative": False,
            "source": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "path": SOURCE_MEMBER,
                "bytes": SOURCE_BYTES,
                "sha256": SOURCE_SHA256,
                "rows": SOURCE_ROWS,
                "license": SOURCE_LICENSE,
                "source_card_path": SOURCE_CARD,
                "source_card_bytes": SOURCE_CARD_BYTES,
                "source_card_sha256": SOURCE_CARD_SHA256,
            },
            "selection": {
                "eligible_rows_before_content_deduplication": len(metadata),
                "unique_eligible_contents": len(
                    {row["content_sha256"] for row in metadata}
                ),
                "screen_rows": len(screen),
                "clean_screen_rows": len(clean),
                "selected_rows": len(final),
                "selected_hosts": len(host_counts),
                "maximum_selected_rows_per_host": max(host_counts.values()),
                "rejection_counts": dict(sorted(rejection_counts.items())),
                "mechanical_policy_sha256": MECHANICAL_POLICY_SHA256,
                "decontamination_policy_sha256": canonical_sha256(
                    DECONTAMINATION_POLICY
                ),
            },
            "boundary_indexes": boundary_receipts,
            "boundary_indexes_sha256": canonical_sha256(boundary_receipts),
            "population": {
                "path": candidates_path.name,
                "rows": len(candidates),
                "bytes": candidates_path.stat().st_size,
                "sha256": sha256_file(candidates_path),
                "ordered_identities_sha256": canonical_sha256(identities),
            },
            "lineage": {
                "path": lineage_path.name,
                "rows": len(lineage),
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
                "ordered_rows_sha256": canonical_sha256(lineage),
            },
            "full_source_file_verified": True,
            "mechanical_quality_gate_complete": True,
            "selected_excerpt_official_boundary_screen_complete": True,
            "full_source_population_decontaminated": False,
            "source_rights_provenance_verified": False,
            "independent_semantic_verification_complete": False,
            "hermes_judgments_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _write_jsonl(temporary / "receipt.json", [payload])
        os.replace(temporary, output_root)
        return payload
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-card", type=Path, required=True)
    parser.add_argument("--boundary-index", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-rows", type=int, default=TARGET_ROWS)
    parser.add_argument("--screen-rows", type=int, default=SCREEN_ROWS)
    args = parser.parse_args()
    result = build_population(
        args.source,
        args.source_card,
        args.boundary_index,
        args.output_root,
        target_rows=args.target_rows,
        screen_rows=args.screen_rows,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
