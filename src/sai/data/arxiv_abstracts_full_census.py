"""Stream a text-free full census of both exact CC0 arXiv parents."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.arxiv_abstracts_audit_population import (
    MAXIMUM_TEXT_BYTES,
    MINIMUM_TEXT_BYTES,
    REPOSITORY,
    REVISION,
    SOURCE_ID,
    SOURCE_MEMORY_BYTES,
    SOURCE_ORIGINAL_BYTES,
    SOURCE_ROWS,
)
from sai.data.arxiv_abstracts_audit_publication import (
    SCHEMA as PUBLICATION_SCHEMA,
)
from sai.data.common_pile_streaming_pilot import download_parent
from sai.data.frontier_source_audit_population import load_frontier_reservoir
from sai.data.reservoir_audit_aggregate import load_population
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-arxiv-abstracts-full-source-census-v1"
EXPECTED_PARENTS = {
    "arxiv-abstracts-dolma-0000.json.gz": {
        "bytes": 892_766_221,
        "sha256": "2aa4dba7c362a977853619239655f073205346978931d6ad070026c7dfa1c4d2",
    },
    "arxiv-abstracts-dolma-0001.json.gz": {
        "bytes": 235_616_002,
        "sha256": "451ff8bbc867360d66ddca2794459377d1cc56f91b2243587857f8ded982af36",
    },
}
EXPECTED_AUDIT_ROWS = 1_060


class ArxivAbstractsFullCensusError(RuntimeError):
    """The source evidence, parent bytes, or text-free census differs."""


def _load_signed(path: Path, schema: str, label: str) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size > 16 << 20
    ):
        raise ArxivAbstractsFullCensusError(f"{label} is missing or unsafe")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ArxivAbstractsFullCensusError(f"{label} cannot be decoded") from error
    if not isinstance(payload, dict):
        raise ArxivAbstractsFullCensusError(f"{label} differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload.get("schema") != schema or payload.get(
        "receipt_sha256"
    ) != canonical_sha256(unsigned):
        raise ArxivAbstractsFullCensusError(f"{label} receipt differs")
    return payload


def validate_census_authorization(publication: dict[str, Any]) -> dict[str, Any]:
    """Authorize a text-free census only, never bulk ingestion."""

    snapshot = publication.get("source_snapshot", {})
    if (
        publication.get("schema") != PUBLICATION_SCHEMA
        or publication.get("status") != "complete_pre_hermes_source_safe_evidence"
        or snapshot.get("repository") != REPOSITORY
        or snapshot.get("revision") != REVISION
        or snapshot.get("rows") != SOURCE_ROWS
        or snapshot.get("original_file_bytes") != SOURCE_ORIGINAL_BYTES
        or snapshot.get("memory_bytes") != SOURCE_MEMORY_BYTES
        or publication.get("input_rows") != 1_024
        or publication.get("clean_rows") != 1_023
        or publication.get("contaminated_rows") != 1
        or publication.get("near_duplicate_pairs") != 0
        or publication.get("recognized_cc0_declaration_rows") != 1_024
        or publication.get("rights_hold_rows") != 0
        or publication.get("hermes_judgments_complete") is not False
        or publication.get("quality_compilation_complete") is not False
        or publication.get("full_source_ingestion_authorized") is not False
        or publication.get("training_ready") is not False
    ):
        raise ArxivAbstractsFullCensusError("full census authorization differs")
    return {
        "decision_scope": "text_free_full_parent_census_only",
        "screen_rows": 1_024,
        "clean_screen_rows": 1_023,
        "recognized_cc0_declaration_rows": 1_024,
        "bulk_ingestion_authorized": False,
        "training_ready": False,
    }


def _source_position(lineage: dict[str, Any]) -> tuple[str, str, str, int]:
    locator = lineage.get("locator")
    repository = lineage.get("repository")
    if not isinstance(locator, dict) or repository != REPOSITORY:
        raise ArxivAbstractsFullCensusError("audit source locator differs")
    path = lineage.get("path")
    line_number = locator.get("line_number")
    if isinstance(path, str) and isinstance(line_number, int) and line_number > 0:
        return repository, path, "physical_line", line_number
    provenance = locator.get("source_provenance")
    if not isinstance(provenance, str) or ":" not in provenance:
        raise ArxivAbstractsFullCensusError("audit source provenance differs")
    path, line_text = provenance.rsplit(":", 1)
    try:
        line_number = int(line_text)
    except ValueError as error:
        raise ArxivAbstractsFullCensusError(
            "audit source provenance differs"
        ) from error
    if Path(path).name != path or line_number <= 0:
        raise ArxivAbstractsFullCensusError("audit source provenance differs")
    return repository, path, "source_provenance", line_number


def load_audit_exclusions(
    roots: list[Path],
) -> tuple[
    dict[tuple[str, str, str, int], str],
    frozenset[str],
    list[dict[str, Any]],
]:
    """Bind prior audit positions and content so validation never enters data."""

    if not roots or len(roots) != len(set(roots)):
        raise ArxivAbstractsFullCensusError("audit exclusion roots differ")
    positions: dict[tuple[str, str, str, int], str] = {}
    content = set()
    receipts = []
    for root in roots:
        candidates, lineage, receipt = load_population(root)
        matched = 0
        for _candidate, source in zip(candidates, lineage, strict=True):
            if source.get("source_id") != SOURCE_ID:
                continue
            position = _source_position(source)
            if position in positions:
                raise ArxivAbstractsFullCensusError("audit source position repeats")
            full_text_sha256 = source.get("full_text_sha256")
            if (
                not isinstance(full_text_sha256, str)
                or len(full_text_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in full_text_sha256
                )
            ):
                raise ArxivAbstractsFullCensusError("audit full-text identity differs")
            positions[position] = full_text_sha256
            content.add(full_text_sha256)
            matched += 1
        receipts.append(
            {
                "root_name": root.name,
                "receipt_sha256": receipt["receipt_sha256"],
                "population_sha256": sha256_file(root / "candidates.jsonl"),
                "lineage_sha256": sha256_file(root / "lineage.jsonl"),
                "matched_source_rows": matched,
            }
        )
    if len(positions) != EXPECTED_AUDIT_ROWS:
        raise ArxivAbstractsFullCensusError("audit exclusion coverage differs")
    return positions, frozenset(content), receipts


def _provenance_line(value: Any, expected_path: str) -> int | None:
    if not isinstance(value, str) or ":" not in value:
        return None
    path, line_text = value.rsplit(":", 1)
    try:
        line_number = int(line_text)
    except ValueError:
        return None
    if path != expected_path or Path(path).name != path or line_number <= 0:
        return None
    return line_number


def _scan_parent(
    compressed_path: Path,
    parent: dict[str, Any],
    excluded_positions: dict[tuple[str, str, str, int], str],
    excluded_content_sha256s: frozenset[str],
    observed_excluded_positions: set[tuple[str, str, str, int]],
    observed_excluded_content_sha256s: set[str],
    seen_content_sha256s: set[bytes],
    seen_native_id_sha256s: set[bytes],
    ordered_digest: Any,
) -> dict[str, Any]:
    """Scan one verified gzip parent while retaining no source text."""

    if (
        not compressed_path.is_file()
        or compressed_path.is_symlink()
        or compressed_path.stat().st_size != parent.get("bytes")
        or sha256_file(compressed_path) != parent.get("sha256")
    ):
        raise ArxivAbstractsFullCensusError("arXiv parent identity differs")
    counters: Counter[str] = Counter()
    collections: Counter[str] = Counter()
    years: Counter[str] = Counter()
    minimum_observed_text_bytes: int | None = None
    maximum_observed_text_bytes = 0
    previous_provenance_line = 0
    try:
        with gzip.open(compressed_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                counters["scanned_rows"] += 1
                row = json.loads(line)
                if not isinstance(row, dict):
                    counters["non_object_rows"] += 1
                    continue
                metadata = row.get("metadata")
                text = row.get("text")
                native_id = row.get("id")
                created = row.get("created")
                if not isinstance(text, str):
                    counters["non_text_rows"] += 1
                    continue
                text = text.strip()
                text_bytes = len(text.encode())
                counters["text_rows"] += 1
                counters["text_bytes"] += text_bytes
                minimum_observed_text_bytes = (
                    text_bytes
                    if minimum_observed_text_bytes is None
                    else min(minimum_observed_text_bytes, text_bytes)
                )
                maximum_observed_text_bytes = max(
                    maximum_observed_text_bytes, text_bytes
                )
                collection = row.get("source")
                collection_label = (
                    collection if isinstance(collection, str) else "<missing>"
                )
                collections[collection_label] += 1
                year = str(created)[:4] if created is not None else "<missing>"
                years[year if len(year) == 4 and year.isdigit() else "<missing>"] += 1
                if not isinstance(metadata, dict):
                    counters["invalid_metadata_rows"] += 1
                    continue
                if not isinstance(native_id, str) or not native_id:
                    counters["invalid_native_id_rows"] += 1
                    continue
                source_line = _provenance_line(
                    metadata.get("provenance"), parent["path"]
                )
                if source_line is None:
                    counters["invalid_provenance_rows"] += 1
                    continue
                if source_line <= previous_provenance_line:
                    counters["non_monotonic_provenance_rows"] += 1
                    continue
                counters["source_provenance_gap_positions"] += (
                    source_line - previous_provenance_line - 1
                )
                previous_provenance_line = source_line
                if source_line != line_number:
                    counters["provenance_physical_line_delta_rows"] += 1
                counters["provenance_valid_rows"] += 1
                content_digest = hashlib.sha256(text.encode()).digest()
                content_hex = content_digest.hex()
                native_digest = hashlib.sha256(native_id.encode()).digest()
                if native_digest in seen_native_id_sha256s:
                    counters["duplicate_native_id_rows"] += 1
                else:
                    seen_native_id_sha256s.add(native_digest)
                positions = (
                    (REPOSITORY, parent["path"], "physical_line", line_number),
                    (
                        REPOSITORY,
                        parent["path"],
                        "source_provenance",
                        source_line,
                    ),
                )
                matched_positions = []
                for position in positions:
                    expected_audit_content = excluded_positions.get(position)
                    if expected_audit_content is None:
                        continue
                    if content_hex != expected_audit_content:
                        raise ArxivAbstractsFullCensusError(
                            "audit position content identity differs"
                        )
                    matched_positions.append(position)
                if matched_positions:
                    observed_excluded_positions.update(matched_positions)
                    observed_excluded_content_sha256s.add(content_hex)
                    counters["audit_excluded_rows"] += 1
                    counters["audit_position_excluded_rows"] += 1
                    counters["audit_position_excluded_identities"] += len(
                        matched_positions
                    )
                    continue
                declared_license = metadata.get("license")
                if not isinstance(
                    declared_license, str
                ) or not declared_license.startswith("Creative Commons Zero"):
                    counters["non_cc0_declaration_rows"] += 1
                    continue
                counters["declared_cc0_rows"] += 1
                counters["validated_source_rows"] += 1
                if content_hex in excluded_content_sha256s:
                    observed_excluded_content_sha256s.add(content_hex)
                    counters["audit_excluded_rows"] += 1
                    counters["audit_content_excluded_rows"] += 1
                    continue
                if text_bytes < MINIMUM_TEXT_BYTES:
                    counters["short_rows"] += 1
                    continue
                if text_bytes > MAXIMUM_TEXT_BYTES:
                    counters["oversized_rows"] += 1
                    continue
                if content_digest in seen_content_sha256s:
                    counters["exact_duplicate_rows"] += 1
                    counters["exact_duplicate_text_bytes"] += text_bytes
                    continue
                seen_content_sha256s.add(content_digest)
                counters["mechanically_eligible_unique_rows"] += 1
                counters["mechanically_eligible_unique_text_bytes"] += text_bytes
                row_identity = canonical_sha256(
                    {
                        "repository": REPOSITORY,
                        "revision": REVISION,
                        "path": parent["path"],
                        "line_number": line_number,
                        "native_id_sha256": native_digest.hex(),
                        "full_text_sha256": content_hex,
                    }
                )
                ordered_digest.update(row_identity.encode() + b"\n")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArxivAbstractsFullCensusError("arXiv parent content differs") from error
    if sha256_file(compressed_path) != parent["sha256"]:
        raise ArxivAbstractsFullCensusError("arXiv parent changed during census")
    return {
        "repository": REPOSITORY,
        "revision": REVISION,
        "path": parent["path"],
        "compressed_bytes": parent["bytes"],
        "compressed_sha256": parent["sha256"],
        **dict(sorted(counters.items())),
        "minimum_observed_text_bytes": minimum_observed_text_bytes,
        "maximum_observed_text_bytes": maximum_observed_text_bytes,
        "maximum_source_provenance_line": previous_provenance_line,
        "by_source_label": dict(sorted(collections.items())),
        "by_created_year": dict(sorted(years.items())),
        "source_text_persisted": False,
    }


def build_census(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    publication_path: Path,
    audit_roots: list[Path],
    output_path: Path,
    *,
    token: str,
    downloader: Callable[[dict[str, Any], str, Path], Path] = download_parent,
) -> dict[str, Any]:
    """Download each parent once, scan all rows, and remove the parent."""

    if output_path.exists() or output_path.is_symlink():
        raise ArxivAbstractsFullCensusError("full census output already exists")
    publication = _load_signed(
        publication_path, PUBLICATION_SCHEMA, "screen publication"
    )
    authorization = validate_census_authorization(publication)
    reservoir = load_frontier_reservoir(manifest_path, reservoir_receipt_path)
    matches = sorted(
        (row for row in reservoir if row.get("source_id") == SOURCE_ID),
        key=lambda row: row["path"],
    )
    if len(matches) != len(EXPECTED_PARENTS) or {row["path"] for row in matches} != set(
        EXPECTED_PARENTS
    ):
        raise ArxivAbstractsFullCensusError("arXiv parent manifest differs")
    parents = []
    for row in matches:
        expected = EXPECTED_PARENTS[row["path"]]
        if (
            row.get("repository") != REPOSITORY
            or row.get("revision") != REVISION
            or row.get("physical_bytes") != expected["bytes"]
            or row.get("sha256") != expected["sha256"]
        ):
            raise ArxivAbstractsFullCensusError("arXiv parent manifest differs")
        parents.append(
            {
                "repository": REPOSITORY,
                "revision": REVISION,
                "path": row["path"],
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
            }
        )
    excluded_positions, excluded_content, audit_receipts = load_audit_exclusions(
        audit_roots
    )
    seen_content: set[bytes] = set()
    seen_native_ids: set[bytes] = set()
    observed_excluded_positions: set[tuple[str, str, str, int]] = set()
    observed_excluded_content: set[str] = set()
    ordered_digest = hashlib.sha256()
    parent_receipts = []
    with tempfile.TemporaryDirectory(prefix="sai-arxiv-full-census-") as temporary:
        temporary_root = Path(temporary)
        for parent in parents:
            parent_root = temporary_root / parent["path"]
            parent_root.mkdir()
            compressed = downloader(parent, token, parent_root)
            parent_receipts.append(
                _scan_parent(
                    compressed,
                    parent,
                    excluded_positions,
                    excluded_content,
                    observed_excluded_positions,
                    observed_excluded_content,
                    seen_content,
                    seen_native_ids,
                    ordered_digest,
                )
            )
    totals: Counter[str] = Counter()
    for parent in parent_receipts:
        for key in (
            "scanned_rows",
            "text_rows",
            "text_bytes",
            "non_object_rows",
            "non_text_rows",
            "invalid_metadata_rows",
            "invalid_native_id_rows",
            "invalid_provenance_rows",
            "non_monotonic_provenance_rows",
            "source_provenance_gap_positions",
            "provenance_physical_line_delta_rows",
            "provenance_valid_rows",
            "non_cc0_declaration_rows",
            "declared_cc0_rows",
            "validated_source_rows",
            "audit_excluded_rows",
            "audit_position_excluded_rows",
            "audit_position_excluded_identities",
            "audit_content_excluded_rows",
            "short_rows",
            "oversized_rows",
            "exact_duplicate_rows",
            "exact_duplicate_text_bytes",
            "duplicate_native_id_rows",
            "mechanically_eligible_unique_rows",
            "mechanically_eligible_unique_text_bytes",
        ):
            totals[key] += parent.get(key, 0)
    coverage_diagnostic = {
        "expected_rows": SOURCE_ROWS,
        "scanned_rows": totals["scanned_rows"],
        "expected_compressed_bytes": SOURCE_ORIGINAL_BYTES,
        "scanned_compressed_bytes": sum(
            parent["compressed_bytes"] for parent in parent_receipts
        ),
        "minimum_audit_excluded_rows": EXPECTED_AUDIT_ROWS,
        "audit_excluded_rows": totals["audit_excluded_rows"],
        "expected_audit_position_excluded_rows": len(excluded_positions),
        "audit_position_excluded_rows": totals["audit_position_excluded_rows"],
        "audit_position_excluded_identities": totals[
            "audit_position_excluded_identities"
        ],
        "audit_content_excluded_rows": totals["audit_content_excluded_rows"],
        "observed_audit_position_identities": len(observed_excluded_positions),
        "expected_audit_content_identities": len(excluded_content),
        "observed_audit_content_identities": len(observed_excluded_content),
        "eligible_unique_rows": totals["mechanically_eligible_unique_rows"],
        "seen_unique_content_identities": len(seen_content),
        "provenance_valid_rows": totals["provenance_valid_rows"],
        "seen_unique_native_ids": len(seen_native_ids),
        "duplicate_native_id_rows": totals["duplicate_native_id_rows"],
    }
    if (
        totals["scanned_rows"] != SOURCE_ROWS
        or coverage_diagnostic["scanned_compressed_bytes"] != SOURCE_ORIGINAL_BYTES
        or observed_excluded_positions != set(excluded_positions)
        or not excluded_content.issubset(observed_excluded_content)
        or totals["audit_position_excluded_identities"] != len(excluded_positions)
        or totals["audit_excluded_rows"] < len(excluded_positions)
        or totals["mechanically_eligible_unique_rows"] != len(seen_content)
        or len(seen_native_ids) + totals["duplicate_native_id_rows"]
        != totals["provenance_valid_rows"]
    ):
        print(
            json.dumps(
                {"event": "arxiv_full_census_coverage_failure", **coverage_diagnostic},
                sort_keys=True,
            ),
            flush=True,
        )
        raise ArxivAbstractsFullCensusError("arXiv full census coverage differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete_text_free_full_parent_census",
        "source_id": SOURCE_ID,
        "source_snapshot": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "parents": len(parent_receipts),
            "compressed_bytes": SOURCE_ORIGINAL_BYTES,
            "reported_memory_bytes": SOURCE_MEMORY_BYTES,
            "rows": SOURCE_ROWS,
        },
        "authorization": authorization,
        "screen_publication": {
            "file_sha256": sha256_file(publication_path),
            "receipt_sha256": publication["receipt_sha256"],
        },
        "reservoir": {
            "manifest_sha256": sha256_file(manifest_path),
            "receipt_file_sha256": sha256_file(reservoir_receipt_path),
        },
        "audit_exclusions": audit_receipts,
        "audit_excluded_positions": len(excluded_positions),
        "audit_excluded_content_identities": len(excluded_content),
        "parents": parent_receipts,
        "totals": dict(sorted(totals.items())),
        "ordered_mechanically_eligible_identity_sha256": ordered_digest.hexdigest(),
        "complete_parent_census": True,
        "maximum_simultaneous_parent_files": 1,
        "parents_removed_after_census": True,
        "source_text_persisted": False,
        "benchmark_contamination_screen_complete": False,
        "near_duplicate_filter_complete": False,
        "hermes_judgments_complete": False,
        "quality_compilation_complete": False,
        "full_source_ingestion_authorized": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reservoir-receipt", type=Path, required=True)
    parser.add_argument("--screen-publication", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    import os

    result = build_census(
        args.manifest,
        args.reservoir_receipt,
        args.screen_publication,
        args.audit_root,
        args.output,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
