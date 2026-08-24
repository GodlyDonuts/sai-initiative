"""Freeze source-disjoint PleIAs collection rows for independent confirmation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.independent_review_population import SCHEMA
from sai.data.reservoir_audit_aggregate import (
    _validate_compiler_receipt,
    load_population,
)
from sai.data.token_stream import canonical_sha256, sha256_file


class PleiasCollectionReviewError(RuntimeError):
    """The source population, exclusions, or collection coverage differs."""


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    stage = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with stage.open("x") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, path)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise


def excluded_identities(path: Path | None) -> frozenset[str]:
    """Load identities from an earlier candidate population."""

    if path is None:
        return frozenset()
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise PleiasCollectionReviewError("collection exclusion differs")
    identities = set()
    try:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                identity = row.get("candidate_identity_sha256")
                if not isinstance(identity, str) or len(identity) != 64:
                    raise PleiasCollectionReviewError(
                        "collection exclusion identity differs"
                    )
                identities.add(identity)
    except (OSError, json.JSONDecodeError) as error:
        raise PleiasCollectionReviewError("collection exclusion differs") from error
    return frozenset(identities)


def select_collection_rows(
    candidates: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    collections: list[str],
    per_collection: int,
    excluded: frozenset[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Select the lowest unused exact identities per requested collection."""

    if (
        not collections
        or len(collections) != len(set(collections))
        or any(not isinstance(value, str) or not value for value in collections)
        or isinstance(per_collection, bool)
        or not isinstance(per_collection, int)
        or per_collection <= 0
        or len(candidates) != len(lineage)
    ):
        raise PleiasCollectionReviewError("collection review geometry differs")
    by_collection: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        value: [] for value in collections
    }
    for candidate, source in zip(candidates, lineage, strict=True):
        identity = candidate["candidate_identity_sha256"]
        locator = source.get("locator")
        if identity in excluded or not isinstance(locator, dict):
            continue
        collection = locator.get("collection")
        if collection in by_collection:
            by_collection[collection].append((candidate, source))
    selected = []
    for collection in collections:
        rows = sorted(
            by_collection[collection],
            key=lambda item: item[0]["candidate_identity_sha256"],
        )
        if len(rows) < per_collection:
            raise PleiasCollectionReviewError(
                f"collection {collection} is underfilled"
            )
        selected.extend(rows[:per_collection])
    selected.sort(key=lambda item: item[0]["candidate_identity_sha256"])
    identities = [item[0]["candidate_identity_sha256"] for item in selected]
    if len(identities) != len(set(identities)):
        raise PleiasCollectionReviewError("collection identities overlap")
    return selected


def build_population(
    population_root: Path,
    judgments_root: Path,
    output_root: Path,
    collections: list[str],
    per_collection: int,
    exclude_candidates: Path | None = None,
) -> dict[str, Any]:
    """Build a compatible independent-review population with primary custody."""

    if output_root.exists() or output_root.is_symlink():
        raise PleiasCollectionReviewError("collection output exists")
    candidates, lineage, population = load_population(population_root)
    excluded = excluded_identities(exclude_candidates)
    available_primary = {
        path.name.removesuffix(".compiler.json")
        for path in judgments_root.glob("*.compiler.json")
    }
    unavailable = {
        row["candidate_identity_sha256"]
        for row in candidates
        if row["candidate_identity_sha256"] not in available_primary
    }
    selected = select_collection_rows(
        candidates, lineage, collections, per_collection, excluded | unavailable
    )
    descriptors = []
    selected_candidates = []
    cells = Counter()
    for candidate, source in selected:
        identity = candidate["candidate_identity_sha256"]
        path = judgments_root / f"{identity}.compiler.json"
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise PleiasCollectionReviewError("primary collection receipt missing")
        try:
            receipt = _validate_compiler_receipt(
                json.loads(path.read_bytes()), candidate
            )
        except (OSError, json.JSONDecodeError, RuntimeError) as error:
            raise PleiasCollectionReviewError(
                "primary collection receipt differs"
            ) from error
        collection = source["locator"]["collection"]
        cells[f"pleias::{collection}"] += 1
        descriptors.append(
            {
                "lane": "pleias",
                "stratum": f"collection::{collection}",
                "collection": collection,
                "candidate_identity_sha256": identity,
                "primary_receipt_sha256": receipt["receipt_sha256"],
                "primary_judgment_sha256": receipt["judgment"][
                    "judgment_sha256"
                ],
            }
        )
        selected_candidates.append(candidate)
    output_root.mkdir(parents=True)
    try:
        candidate_path = output_root / "candidates.jsonl"
        _atomic_jsonl(candidate_path, selected_candidates)
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_independent_review_population",
            "per_source_stratum_limit": per_collection,
            "source_snapshots": [
                {
                    "lane": "pleias",
                    "population_receipt_sha256": population["receipt_sha256"],
                    "candidate_file_sha256": sha256_file(
                        population_root / "candidates.jsonl"
                    ),
                    "completed_receipts_at_freeze": len(available_primary),
                    "selected_completed_receipts": len(selected),
                    "ordered_completed_receipts_sha256": canonical_sha256(
                        [row["primary_receipt_sha256"] for row in descriptors]
                    ),
                }
            ],
            "requested_collections": collections,
            "excluded_population": {
                "candidate_file_name": (
                    exclude_candidates.name if exclude_candidates else None
                ),
                "candidate_file_sha256": (
                    sha256_file(exclude_candidates) if exclude_candidates else None
                ),
                "excluded_identities": len(excluded),
            },
            "population": {
                "path": candidate_path.name,
                "rows": len(selected_candidates),
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
                "ordered_identities_sha256": canonical_sha256(
                    [row["candidate_identity_sha256"] for row in selected_candidates]
                ),
            },
            "selected_cells": dict(sorted(cells.items())),
            "selected_descriptors": descriptors,
            "source_text_persisted_in_receipt": False,
            "selection_is_training_admission": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--collection", action="append", required=True)
    parser.add_argument("--per-collection", type=int, default=8)
    parser.add_argument("--exclude-candidates", type=Path)
    args = parser.parse_args()
    result = build_population(
        args.population_root,
        args.judgments_root,
        args.output_root,
        args.collection,
        args.per_collection,
        args.exclude_candidates,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "population": result["population"],
                "selected_cells": result["selected_cells"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
