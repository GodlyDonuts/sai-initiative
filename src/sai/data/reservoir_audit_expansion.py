"""Build a byte-aware second Hermes audit wave from the 8 TiB reservoir."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from sai.data.reservoir_audit_aggregate import load_population
from sai.data.reservoir_audit_population import (
    SCHEMA,
    _acquire_one,
    _candidate_and_lineage,
    _load_reservoir,
    _write_jsonl,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SEED = 20260824
EXPECTED_ROWS = 1024
MINIMUM_ROWS_PER_SOURCE = 32
AUDIT_SOURCE_IDS = (
    "finepdfs",
    "finemath",
    "dolma3_mix_150b",
    "smollm_corpus",
    "open_web_math",
    "fineweb_edu_fill",
)


class ReservoirAuditExpansionError(RuntimeError):
    """The expansion geometry, identity, or acquisition differs."""


def _source_type(source_id: str, path: str) -> tuple[str, str]:
    if source_id == "finepdfs":
        language = path.split("/", 2)[1] if "/" in path else "unknown"
        return f"language:{language}", "reference"
    if source_id == "finemath":
        return f"subset:{path.split('/', 1)[0]}", "educational_web"
    if source_id == "smollm_corpus":
        subset = path.split("/", 1)[0]
        return (
            f"subset:{subset}",
            "synthetic" if subset == "cosmopedia-v2" else "educational_web",
        )
    if source_id == "open_web_math":
        return "all", "educational_web"
    if source_id == "fineweb_edu_fill":
        crawl = path.split("/", 2)[1] if "/" in path else "unknown"
        return f"crawl:{crawl}", "educational_web"
    if source_id == "dolma3_mix_150b":
        if "stack_edu-" in path:
            return (
                f"code:{path.split('stack_edu-', 1)[1].split('/', 1)[0]}",
                "code_repository",
            )
        if "rpj-proofpile-arxiv" in path:
            return "arxiv", "research_paper"
        if "wiki" in path:
            return "wikipedia", "reference"
        if "olmocr_science_pdfs-" in path:
            return (
                f"pdf:{path.split('olmocr_science_pdfs-', 1)[1].split('/', 1)[0]}",
                "reference",
            )
        if "common_crawl-" in path:
            return (
                f"web:{path.split('common_crawl-', 1)[1].rsplit('-', 1)[0]}",
                "general_web",
            )
        if "finemath" in path:
            return "mathematics", "educational_web"
        return f"component:{path.split('/', 2)[1]}", "reference"
    raise ReservoirAuditExpansionError("expansion source differs")


def allocate_source_quotas(
    rows: list[dict[str, Any]],
    *,
    total_rows: int = EXPECTED_ROWS,
    minimum_rows_per_source: int = MINIMUM_ROWS_PER_SOURCE,
) -> dict[str, int]:
    """Allocate a coverage floor plus byte-proportional remaining rows."""

    if (
        isinstance(total_rows, bool)
        or not isinstance(total_rows, int)
        or isinstance(minimum_rows_per_source, bool)
        or not isinstance(minimum_rows_per_source, int)
        or minimum_rows_per_source <= 0
        or total_rows < minimum_rows_per_source * len(AUDIT_SOURCE_IDS)
    ):
        raise ReservoirAuditExpansionError("expansion quota geometry differs")
    by_source_bytes = Counter()
    by_source_files = Counter()
    for row in rows:
        source_id = row.get("source_id")
        if source_id in AUDIT_SOURCE_IDS:
            size = row.get("bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ReservoirAuditExpansionError("expansion source bytes differ")
            by_source_bytes[source_id] += size
            by_source_files[source_id] += 1
    if set(by_source_bytes) != set(AUDIT_SOURCE_IDS):
        raise ReservoirAuditExpansionError("expansion source coverage differs")
    remaining = total_rows - minimum_rows_per_source * len(AUDIT_SOURCE_IDS)
    total_bytes = sum(by_source_bytes.values())
    exact_additions = {
        source_id: Decimal(remaining) * by_source_bytes[source_id] / total_bytes
        for source_id in AUDIT_SOURCE_IDS
    }
    additions = {source_id: int(value) for source_id, value in exact_additions.items()}
    leftover = remaining - sum(additions.values())
    order = sorted(
        AUDIT_SOURCE_IDS,
        key=lambda source_id: (
            -(exact_additions[source_id] - additions[source_id]),
            source_id,
        ),
    )
    for source_id in order[:leftover]:
        additions[source_id] += 1
    quotas = {
        source_id: minimum_rows_per_source + additions[source_id]
        for source_id in AUDIT_SOURCE_IDS
    }
    if sum(quotas.values()) != total_rows or any(
        quotas[source_id] > by_source_files[source_id] for source_id in AUDIT_SOURCE_IDS
    ):
        raise ReservoirAuditExpansionError("expansion quota capacity differs")
    return dict(sorted(quotas.items()))


def _exponential_race_key(row: dict[str, Any]) -> Decimal:
    digest = hashlib.sha256(
        (
            f"{SEED}:{row['source_id']}:{row['repository']}:"
            f"{row['path']}:{row['sha256']}"
        ).encode()
    ).digest()
    with localcontext() as context:
        context.prec = 80
        uniform = Decimal(int.from_bytes(digest) + 1) / Decimal(2**256 + 1)
        return -uniform.ln() / Decimal(row["bytes"])


def build_weighted_plan(
    rows: list[dict[str, Any]],
    excluded_parent_keys: set[tuple[str, str]],
    *,
    total_rows: int = EXPECTED_ROWS,
    minimum_rows_per_source: int = MINIMUM_ROWS_PER_SOURCE,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select disjoint parent files with deterministic byte-weighted sampling."""

    quotas = allocate_source_quotas(
        rows,
        total_rows=total_rows,
        minimum_rows_per_source=minimum_rows_per_source,
    )
    plan = []
    for source_id in AUDIT_SOURCE_IDS:
        available = [
            row
            for row in rows
            if row["source_id"] == source_id
            and (row["repository"], row["path"]) not in excluded_parent_keys
        ]
        ranked = sorted(
            available,
            key=lambda row: (_exponential_race_key(row), row["path"]),
        )
        if len(ranked) < quotas[source_id]:
            raise ReservoirAuditExpansionError(
                f"expansion source is underfilled: {source_id}"
            )
        for row in ranked[: quotas[source_id]]:
            stratum, source_type = _source_type(source_id, row["path"])
            selection_key = hashlib.sha256(
                (
                    f"{SEED}:{source_id}:{row['repository']}:"
                    f"{row['path']}:{row['sha256']}"
                ).encode()
            ).hexdigest()
            plan.append(
                {
                    "ordinal": len(plan),
                    "source_id": source_id,
                    "stratum": stratum,
                    "source_type": source_type,
                    "repository": row["repository"],
                    "revision": row["revision"],
                    "license": row["license"],
                    "access": row["access"],
                    "path": row["path"],
                    "parent_file_bytes": row["bytes"],
                    "parent_file_sha256": row["sha256"],
                    "selection_key": selection_key,
                }
            )
    parent_keys = [(row["repository"], row["path"]) for row in plan]
    if len(plan) != total_rows or len(parent_keys) != len(set(parent_keys)):
        raise ReservoirAuditExpansionError("expansion plan identity differs")
    return plan, quotas


def build_expansion_population(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    excluded_population_root: Path,
    output_root: Path,
    *,
    token: str,
) -> dict[str, Any]:
    """Acquire and seal the disjoint 1,024-row second audit wave."""

    if not token or output_root.exists() or output_root.is_symlink():
        raise ReservoirAuditExpansionError(
            "expansion credential or output boundary differs"
        )
    rows = _load_reservoir(manifest_path, reservoir_receipt_path)
    _, excluded_lineage, excluded_receipt = load_population(excluded_population_root)
    excluded_parent_keys = {
        (row["repository"], row["path"]) for row in excluded_lineage
    }
    plan, quotas = build_weighted_plan(rows, excluded_parent_keys)
    candidates = []
    lineage = []
    for index, item in enumerate(plan, start=1):
        candidate, source_lineage = _candidate_and_lineage(
            item, _acquire_one(item, token)
        )
        candidates.append(candidate)
        lineage.append(source_lineage)
        if index % 16 == 0 or index == len(plan):
            print(
                json.dumps(
                    {
                        "event": "reservoir_audit_expansion_progress",
                        "acquired": index,
                        "remaining": len(plan) - index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if len(candidates) != EXPECTED_ROWS or len(identities) != len(set(identities)):
        raise ReservoirAuditExpansionError("expansion candidates differ")
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise ReservoirAuditExpansionError("expansion temporary output exists")
    temporary.mkdir(parents=True)
    try:
        candidate_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        receipt_path = temporary / "receipt.json"
        _write_jsonl(candidate_path, candidates)
        _write_jsonl(lineage_path, lineage)
        by_source = Counter(row["source_id"] for row in lineage)
        by_stratum = Counter(f"{row['source_id']}::{row['stratum']}" for row in lineage)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "seed": SEED,
            "selection_method": (
                "per_source_coverage_floor_then_byte_proportional_quota_and_"
                "compressed_parent_file_ppswor_exponential_race"
            ),
            "statistically_representative": False,
            "statistical_scope": (
                "byte_aware_parent_file_diagnostic_not_document_token_or_"
                "acceptance_estimate"
            ),
            "source_quotas": quotas,
            "excluded_population_receipt_sha256": excluded_receipt["receipt_sha256"],
            "excluded_parent_files": len(excluded_parent_keys),
            "reservoir": {
                "manifest_sha256": sha256_file(manifest_path),
                "receipt_sha256": sha256_file(reservoir_receipt_path),
                "selected_files": len(rows),
                "selected_bytes": sum(row["bytes"] for row in rows),
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
            "by_stratum": dict(sorted(by_stratum.items())),
            "range_read_parent_files": sum(
                not row["full_file_content_verified"] for row in lineage
            ),
            "fully_verified_parent_files": sum(
                row["full_file_content_verified"] for row in lineage
            ),
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
    parser.add_argument("--exclude-population-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    receipt = build_expansion_population(
        args.manifest,
        args.reservoir_receipt,
        args.exclude_population_root,
        args.output_root,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
