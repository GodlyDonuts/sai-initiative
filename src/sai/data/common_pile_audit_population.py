"""Acquire a storage-bounded audit population across every Common Pile source."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sai.data.frontier_source_audit_population import load_frontier_reservoir
from sai.data.frontier_source_reservoir import COMMON_PILE_FILTERED_SOURCES
from sai.data.reservoir_audit_population import (
    SCHEMA,
    ReservoirAuditError,
    _candidate_and_lineage,
    _write_jsonl,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SEED = 20260824
ROWS_PER_SOURCE = 4
EXPECTED_SOURCES = len(COMMON_PILE_FILTERED_SOURCES)
EXPECTED_ROWS = EXPECTED_SOURCES * ROWS_PER_SOURCE

COMMON_PILE_SOURCE_TYPES = {
    "arxiv_abstracts": "research_paper",
    "arxiv_papers": "research_paper",
    "biodiversity_heritage_library": "research_paper",
    "caselaw_access_project": "reference",
    "cccc": "general_web",
    "data_provenance_initiative": "reference",
    "doab": "textbook",
    "foodista": "general_web",
    "github_archive": "code_repository",
    "library_of_congress": "reference",
    "libretexts": "textbook",
    "news": "general_web",
    "oercommons": "textbook",
    "peS2o": "research_paper",
    "pre_1929_books": "reference",
    "pressbooks": "textbook",
    "project_gutenberg": "reference",
    "public_domain_review": "reference",
    "pubmed": "research_paper",
    "python_enhancement_proposals": "documentation",
    "regulations": "reference",
    "stackexchange": "forum",
    "stackv2_edu": "code_repository",
    "stackv2_html": "code_repository",
    "ubuntu_irc": "forum",
    "uk_hansard": "reference",
    "usgpo": "reference",
    "uspto": "reference",
    "wikimedia": "reference",
    "wikiteam": "reference",
    "youtube": "general_web",
}
if set(COMMON_PILE_SOURCE_TYPES) != {
    name for name, _, _ in COMMON_PILE_FILTERED_SOURCES
}:  # pragma: no cover - frozen source geometry
    raise RuntimeError("Common Pile source-type map differs")


class CommonPileAuditError(RuntimeError):
    """The Common Pile source geometry, bytes, or sample differs."""


def _parent_selection_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{SEED}:{row['source_id']}:{row['repository']}:"
            f"{row['path']}:{row['sha256']}"
        ).encode()
    ).hexdigest()


def build_parent_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the smallest exact parent from every Common Pile component."""

    expected = {f"common_pile_{name}" for name, _, _ in COMMON_PILE_FILTERED_SOURCES}
    plan = []
    for source_id in sorted(expected):
        component = source_id.removeprefix("common_pile_")
        matches = [row for row in rows if row["source_id"] == source_id]
        if not matches:
            raise CommonPileAuditError(f"Common Pile source is absent: {source_id}")
        selected = min(
            matches,
            key=lambda row: (row["physical_bytes"], row["path"], row["sha256"]),
        )
        plan.append(
            {
                "source_id": selected["source_id"],
                "stratum": selected["epistemic_function"],
                "source_type": COMMON_PILE_SOURCE_TYPES[component],
                "repository": selected["repository"],
                "revision": selected["revision"],
                "license": selected["license"],
                "access": selected["access"],
                "path": selected["path"],
                "parent_file_bytes": selected["physical_bytes"],
                "parent_file_sha256": selected["sha256"],
                "text_column": selected["text_column"],
                "parent_selection_key": _parent_selection_key(selected),
            }
        )
    if (
        len(plan) != EXPECTED_SOURCES
        or {row["source_id"] for row in plan} != expected
        or len({(row["repository"], row["path"]) for row in plan}) != len(plan)
    ):
        raise CommonPileAuditError("Common Pile parent geometry differs")
    return plan


def _native_id(row: dict[str, Any]) -> str | None:
    value = row.get("id")
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return canonical_sha256(value)


def _declared_license(row: dict[str, Any], fallback: str) -> str:
    metadata = row.get("metadata")
    candidates = [row.get("license")]
    if isinstance(metadata, dict):
        candidates.insert(0, metadata.get("license"))
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def sample_verified_gzip_parent(
    path: Path,
    parent: dict[str, Any],
    *,
    rows_per_source: int = ROWS_PER_SOURCE,
    excluded_line_numbers: frozenset[int] = frozenset(),
    excluded_text_sha256s: frozenset[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Verify one compressed parent fully and select deterministic bottom-k rows."""

    if (
        rows_per_source <= 0
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != parent["parent_file_bytes"]
        or sha256_file(path) != parent["parent_file_sha256"]
    ):
        raise CommonPileAuditError("Common Pile parent identity differs")

    heap: list[tuple[int, str, int, str, str | None, str, str]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise CommonPileAuditError("Common Pile row is not an object")
                text = row.get("text")
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if len(text.encode("utf-8")) < 200:
                    continue
                text_sha256 = hashlib.sha256(text.encode()).hexdigest()
                if (
                    line_number in excluded_line_numbers
                    or text_sha256 in excluded_text_sha256s
                ):
                    continue
                native_id = _native_id(row)
                key = hashlib.sha256(
                    (
                        f"{parent['parent_selection_key']}:{line_number}:"
                        f"{native_id or ''}:{text_sha256}"
                    ).encode()
                ).hexdigest()
                metadata = row.get("metadata")
                metadata_sha256 = canonical_sha256(
                    metadata if isinstance(metadata, dict) else {}
                )
                item = (
                    -int(key, 16),
                    key,
                    line_number,
                    text,
                    native_id,
                    _declared_license(row, parent["license"]),
                    metadata_sha256,
                )
                if len(heap) < rows_per_source:
                    heapq.heappush(heap, item)
                elif int(key, 16) < -heap[0][0]:
                    heapq.heapreplace(heap, item)
    except CommonPileAuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CommonPileAuditError("Common Pile gzip content differs") from error

    if len(heap) != rows_per_source:
        raise CommonPileAuditError("Common Pile parent has too few usable rows")
    selected = []
    for _, key, line_number, text, native_id, license_name, metadata_sha256 in sorted(
        heap, key=lambda item: item[1]
    ):
        selected.append(
            {
                "text": text,
                "locator": {
                    "format": "json.gz",
                    "line_number": line_number,
                    "native_id": native_id,
                    "metadata_sha256": metadata_sha256,
                },
                "declared_license": license_name,
                "row_selection_key": key,
                "full_file_content_verified": True,
            }
        )
    return selected


def download_and_sample_parent(
    parent: dict[str, Any], token: str
) -> list[dict[str, Any]]:
    """Download only one parent at a time and remove it after full verification."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise CommonPileAuditError("huggingface_hub is required") from error
    with tempfile.TemporaryDirectory(prefix="sai-common-pile-") as temporary:
        try:
            path = Path(
                hf_hub_download(
                    parent["repository"],
                    parent["path"],
                    repo_type="dataset",
                    revision=parent["revision"],
                    token=token,
                    local_dir=temporary,
                )
            )
        except Exception as error:
            raise CommonPileAuditError("Common Pile download failed") from error
        return sample_verified_gzip_parent(path, parent)


def build_population(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    output_root: Path,
    *,
    token: str,
    acquire_function: Callable[
        [dict[str, Any], str], list[dict[str, Any]]
    ] = download_and_sample_parent,
) -> dict[str, Any]:
    """Acquire and seal four fully verified rows per Common Pile component."""

    if not token or output_root.exists() or output_root.is_symlink():
        raise CommonPileAuditError("Common Pile credential or output boundary differs")
    rows = load_frontier_reservoir(manifest_path, reservoir_receipt_path)
    plan = build_parent_plan(rows)
    candidates = []
    lineage = []
    for parent_index, parent in enumerate(plan, start=1):
        acquired_rows = acquire_function(parent, token)
        if len(acquired_rows) != ROWS_PER_SOURCE:
            raise CommonPileAuditError("Common Pile acquired row count differs")
        for acquired in acquired_rows:
            row_plan = {
                **parent,
                "ordinal": len(candidates),
                "license": acquired["declared_license"],
                "selection_key": acquired["row_selection_key"],
            }
            try:
                candidate, source_lineage = _candidate_and_lineage(row_plan, acquired)
            except ReservoirAuditError as error:
                raise CommonPileAuditError("Common Pile candidate differs") from error
            source_lineage["manifest_license"] = parent["license"]
            source_lineage["declared_license"] = acquired["declared_license"]
            source_lineage.pop("lineage_sha256")
            source_lineage["lineage_sha256"] = canonical_sha256(source_lineage)
            candidates.append(candidate)
            lineage.append(source_lineage)
        print(
            json.dumps(
                {
                    "event": "common_pile_audit_acquisition_progress",
                    "parents_acquired": parent_index,
                    "parents_remaining": len(plan) - parent_index,
                    "rows_acquired": len(candidates),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    identities = [row["candidate_identity_sha256"] for row in candidates]
    if len(candidates) != EXPECTED_ROWS or len(identities) != len(set(identities)):
        raise CommonPileAuditError("Common Pile candidate identities differ")
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise CommonPileAuditError("Common Pile temporary output exists")
    temporary.mkdir(parents=True)
    try:
        candidate_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        receipt_path = temporary / "receipt.json"
        _write_jsonl(candidate_path, candidates)
        _write_jsonl(lineage_path, lineage)
        by_source = Counter(row["source_id"] for row in lineage)
        parent_bytes = sum(row["parent_file_bytes"] for row in plan)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "seed": SEED,
            "selection_method": (
                "smallest_exact_parent_per_common_pile_component_then_"
                "deterministic_bottom_k_usable_rows"
            ),
            "statistically_representative": False,
            "reservoir": {
                "manifest_sha256": sha256_file(manifest_path),
                "receipt_sha256": sha256_file(reservoir_receipt_path),
                "selected_files": len(rows),
                "selected_bytes": sum(row["physical_bytes"] for row in rows),
            },
            "population": {
                "path": candidate_path.name,
                "rows": len(candidates),
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
                "ordered_identities_sha256": canonical_sha256(identities),
            },
            "lineage": {
                "path": lineage_path.name,
                "rows": len(lineage),
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
                "ordered_rows_sha256": canonical_sha256(lineage),
            },
            "by_source": dict(sorted(by_source.items())),
            "fully_verified_parent_files": len(plan),
            "fully_verified_compressed_parent_bytes": parent_bytes,
            "maximum_simultaneous_parent_files": 1,
            "raw_source_rows_are_training_ready": False,
            "hermes_judgments_complete": False,
            "training_ready": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        _write_jsonl(receipt_path, [receipt])
        os.replace(temporary, output_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reservoir-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    payload = build_population(
        args.manifest,
        args.reservoir_receipt,
        args.output_root,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
