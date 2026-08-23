"""Acquire a deterministic temporal screen of exact CC0 arXiv abstracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import CANDIDATE_SCHEMA, normalize_candidate
from sai.data.common_pile_streaming_pilot import audit_exclusions
from sai.data.reservoir_audit_population import (
    LINEAGE_SCHEMA,
    SCHEMA,
    _excerpt,
    _write_jsonl,
)
from sai.data.token_stream import canonical_sha256, sha256_file

REPOSITORY = "common-pile/arxiv_abstracts_filtered"
REVISION = "dc1ceab4755eb037ec61e49cf1350dab7ceee6e7"
SOURCE_ID = "common_pile_arxiv_abstracts"
LICENSE = "CC0-1.0"
SOURCE_TYPE = "research_paper"
DATASET_SERVER = "https://datasets-server.huggingface.co"
CONFIG = "default"
SPLIT = "train"
SEED = 20260825
SOURCE_ROWS = 2_504_679
SOURCE_ORIGINAL_BYTES = 1_128_382_223
SOURCE_MEMORY_BYTES = 3_473_188_609
TEMPORAL_STRATA = 32
FETCH_ROWS_PER_STRATUM = 64
SELECT_ROWS_PER_STRATUM = 32
EXPECTED_ROWS = TEMPORAL_STRATA * SELECT_ROWS_PER_STRATUM
MINIMUM_TEXT_BYTES = 200
MAXIMUM_TEXT_BYTES = 128 * 1024


class ArxivAbstractsAuditError(RuntimeError):
    """The exact arXiv source, audit geometry, or response custody differs."""


def _resolve_revision() -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise ArxivAbstractsAuditError("huggingface_hub is required") from error
    return HfApi().dataset_info(REPOSITORY, revision=REVISION).sha


def build_batch_plan() -> list[dict[str, Any]]:
    """Choose one deterministic 64-row window within each temporal stratum."""

    plan = []
    for stratum_index in range(TEMPORAL_STRATA):
        start = SOURCE_ROWS * stratum_index // TEMPORAL_STRATA
        end = SOURCE_ROWS * (stratum_index + 1) // TEMPORAL_STRATA
        maximum_offset = end - FETCH_ROWS_PER_STRATUM
        if maximum_offset < start:
            raise ArxivAbstractsAuditError("arXiv temporal stratum is too small")
        key = hashlib.sha256(
            f"{SEED}:{REPOSITORY}:{REVISION}:{stratum_index}".encode()
        ).hexdigest()
        offset = start + int(key, 16) % (maximum_offset - start + 1)
        plan.append(
            {
                "ordinal": stratum_index,
                "stratum_index": stratum_index,
                "stratum": f"temporal_{stratum_index:02d}",
                "stratum_start": start,
                "stratum_end": end,
                "offset": offset,
                "length": FETCH_ROWS_PER_STRATUM,
                "selection_key": key,
            }
        )
    return plan


def _fetch_batch(plan: dict[str, Any]) -> dict[str, Any]:
    try:
        import requests
    except ImportError as error:
        raise ArxivAbstractsAuditError("requests is required") from error
    params = {
        "dataset": REPOSITORY,
        "config": CONFIG,
        "split": SPLIT,
        "offset": plan["offset"],
        "length": plan["length"],
    }
    try:
        response = requests.get(
            f"{DATASET_SERVER}/rows", params=params, timeout=60
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, json.JSONDecodeError) as error:
        raise ArxivAbstractsAuditError(
            "arXiv dataset-server request failed"
        ) from error
    return {
        "x_revision": response.headers.get("x-revision"),
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
        "request_url": response.url,
        "payload": payload,
    }


def _validate_and_select_batch(
    plan: dict[str, Any],
    result: dict[str, Any],
    excluded_content_sha256s: frozenset[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = result.get("payload")
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if (
        result.get("x_revision") != REVISION
        or not isinstance(result.get("response_sha256"), str)
        or len(result["response_sha256"]) != 64
        or not isinstance(result.get("request_url"), str)
        or not isinstance(rows, list)
        or len(rows) != FETCH_ROWS_PER_STRATUM
        or payload.get("num_rows_total") != SOURCE_ROWS
    ):
        raise ArxivAbstractsAuditError("arXiv response geometry differs")
    eligible = []
    counters: Counter[str] = Counter()
    for local_index, item in enumerate(rows):
        expected_index = plan["offset"] + local_index
        row = item.get("row") if isinstance(item, dict) else None
        metadata = row.get("metadata") if isinstance(row, dict) else None
        text = row.get("text") if isinstance(row, dict) else None
        native_id = row.get("id") if isinstance(row, dict) else None
        declared_license = (
            metadata.get("license") if isinstance(metadata, dict) else None
        )
        if (
            item.get("row_idx") != expected_index
            or not isinstance(text, str)
            or not isinstance(native_id, str)
            or not native_id
            or not isinstance(metadata, dict)
            or not isinstance(declared_license, str)
            or not declared_license.startswith("Creative Commons Zero")
        ):
            raise ArxivAbstractsAuditError("arXiv source row differs")
        text = text.strip()
        text_bytes = len(text.encode())
        if text_bytes < MINIMUM_TEXT_BYTES:
            counters["short_rows"] += 1
            continue
        if text_bytes > MAXIMUM_TEXT_BYTES:
            counters["oversized_rows"] += 1
            continue
        excerpt, _ = _excerpt(text)
        excerpt_sha256 = hashlib.sha256(excerpt.encode()).hexdigest()
        if excerpt_sha256 in excluded_content_sha256s:
            counters["audit_excluded_rows"] += 1
            continue
        selection_key = canonical_sha256(
            {
                "seed": SEED,
                "stratum_selection_key": plan["selection_key"],
                "row_index": expected_index,
                "native_id": native_id,
                "full_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            }
        )
        eligible.append(
            {
                "selection_key": selection_key,
                "row_index": expected_index,
                "native_id": native_id,
                "text": text,
                "metadata": metadata,
            }
        )
        counters["eligible_rows"] += 1
    selected = sorted(eligible, key=lambda row: row["selection_key"])[
        :SELECT_ROWS_PER_STRATUM
    ]
    if len(selected) != SELECT_ROWS_PER_STRATUM:
        raise ArxivAbstractsAuditError("arXiv stratum lacks eligible source rows")
    receipt = {
        "schema": "sai-arxiv-abstracts-dataset-server-batch-v1",
        "status": "complete",
        "repository": REPOSITORY,
        "revision": REVISION,
        "config": CONFIG,
        "split": SPLIT,
        "stratum_index": plan["stratum_index"],
        "stratum_start": plan["stratum_start"],
        "stratum_end": plan["stratum_end"],
        "offset": plan["offset"],
        "fetched_rows": FETCH_ROWS_PER_STRATUM,
        "eligible_rows": counters["eligible_rows"],
        "selected_rows": len(selected),
        "short_rows": counters["short_rows"],
        "oversized_rows": counters["oversized_rows"],
        "audit_excluded_rows": counters["audit_excluded_rows"],
        "selection_key": plan["selection_key"],
        "request_url": result["request_url"],
        "response_sha256": result["response_sha256"],
        "x_revision": result["x_revision"],
        "selected_row_indexes": sorted(row["row_index"] for row in selected),
        "source_text_persisted": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return sorted(selected, key=lambda row: row["row_index"]), receipt


def _candidate_and_lineage(
    plan: dict[str, Any], row: dict[str, Any], ordinal: int, response_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_text = row["text"]
    metadata = row["metadata"]
    excerpt, excerpt_method = _excerpt(full_text)
    full_text_sha256 = hashlib.sha256(full_text.encode()).hexdigest()
    excerpt_sha256 = hashlib.sha256(excerpt.encode()).hexdigest()
    locator = {
        "dataset_server": DATASET_SERVER,
        "config": CONFIG,
        "split": SPLIT,
        "row_index": row["row_index"],
        "native_id": row["native_id"],
        "source_provenance": metadata.get("provenance"),
        "source_url": metadata.get("url"),
        "authors": metadata.get("authors"),
        "created": metadata.get("created"),
        "declared_license": metadata["license"],
        "full_text_license": metadata.get("full_text_license"),
        "response_sha256": response_sha256,
    }
    row_id = canonical_sha256(
        {
            "repository": REPOSITORY,
            "revision": REVISION,
            "config": CONFIG,
            "split": SPLIT,
            "row_index": row["row_index"],
            "native_id": row["native_id"],
        }
    )
    provenance_sha256 = canonical_sha256(
        {
            "locator": locator,
            "full_text_sha256": full_text_sha256,
            "excerpt_method": excerpt_method,
            "excerpt_sha256": excerpt_sha256,
        }
    )
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "text": excerpt,
        "source": {
            "dataset": REPOSITORY,
            "revision": REVISION,
            "row_id": row_id,
            "license": LICENSE,
            "source_type": SOURCE_TYPE,
        },
        "source_content_sha256": excerpt_sha256,
        "provenance_sha256": provenance_sha256,
    }
    candidate["candidate_identity_sha256"] = canonical_sha256(candidate)
    candidate = normalize_candidate(candidate)
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "ordinal": ordinal,
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "source_id": SOURCE_ID,
        "stratum": plan["stratum"],
        "selection_key": row["selection_key"],
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": LICENSE,
        "declared_license": metadata["license"],
        "access": "public",
        "locator": locator,
        "full_file_content_verified": False,
        "full_text_bytes": len(full_text.encode()),
        "full_text_sha256": full_text_sha256,
        "excerpt_method": excerpt_method,
        "excerpt_bytes": len(excerpt.encode()),
        "excerpt_sha256": excerpt_sha256,
        "source_declared_cc0": True,
        "raw_source_is_training_ready": False,
    }
    lineage["lineage_sha256"] = canonical_sha256(lineage)
    return candidate, lineage


def _load_audit_exclusions(
    roots: list[Path],
) -> tuple[frozenset[str], list[dict[str, Any]]]:
    try:
        _lines, content, receipts = audit_exclusions(roots, SOURCE_ID)
    except Exception as error:  # noqa: BLE001 - normalize shared audit errors
        raise ArxivAbstractsAuditError("arXiv audit exclusions differ") from error
    return content, receipts


def build_population(
    output_root: Path,
    audit_roots: list[Path],
    *,
    revision_resolver: Callable[[], str] = _resolve_revision,
    batch_fetcher: Callable[[dict[str, Any]], dict[str, Any]] = _fetch_batch,
    exclusion_loader: Callable[
        [list[Path]], tuple[frozenset[str], list[dict[str, Any]]]
    ] = _load_audit_exclusions,
) -> dict[str, Any]:
    """Seal 1,024 source-disjoint rows without admitting them for training."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not audit_roots
        or len(audit_roots) != len(set(audit_roots))
        or revision_resolver() != REVISION
    ):
        raise ArxivAbstractsAuditError("arXiv audit output boundary differs")
    excluded_content, audit_receipts = exclusion_loader(audit_roots)
    candidates: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    batch_receipts: list[dict[str, Any]] = []
    for plan in build_batch_plan():
        selected, batch_receipt = _validate_and_select_batch(
            plan, batch_fetcher(plan), excluded_content
        )
        batch_receipts.append(batch_receipt)
        for row in selected:
            candidate, source_lineage = _candidate_and_lineage(
                plan,
                row,
                len(candidates),
                batch_receipt["response_sha256"],
            )
            candidates.append(candidate)
            lineage.append(source_lineage)
        print(
            json.dumps(
                {
                    "event": "arxiv_abstracts_audit_progress",
                    "completed_strata": len(batch_receipts),
                    "rows": len(candidates),
                    "remaining_rows": EXPECTED_ROWS - len(candidates),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if revision_resolver() != REVISION:
        raise ArxivAbstractsAuditError("arXiv source revision changed")
    identities = [row["candidate_identity_sha256"] for row in candidates]
    content_identities = [row["source_content_sha256"] for row in candidates]
    if (
        len(candidates) != EXPECTED_ROWS
        or len(identities) != len(set(identities))
        or set(content_identities) & excluded_content
    ):
        raise ArxivAbstractsAuditError("arXiv audit identities differ")

    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise ArxivAbstractsAuditError("arXiv audit temporary output exists")
    temporary.mkdir(parents=True)
    try:
        candidates_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        batches_path = temporary / "batch_receipts.jsonl"
        receipt_path = temporary / "receipt.json"
        _write_jsonl(candidates_path, candidates)
        _write_jsonl(lineage_path, lineage)
        _write_jsonl(batches_path, batch_receipts)
        by_stratum = Counter(row["stratum"] for row in lineage)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "seed": SEED,
            "selection_method": (
                "32_equal_temporal_strata_one_sha256_window_and_bottom32_per_stratum"
            ),
            "statistically_representative": False,
            "screen_only": True,
            "source_snapshot": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "config": CONFIG,
                "split": SPLIT,
                "rows": SOURCE_ROWS,
                "original_file_bytes": SOURCE_ORIGINAL_BYTES,
                "memory_bytes": SOURCE_MEMORY_BYTES,
                "declared_license": LICENSE,
            },
            "audit_exclusions": audit_receipts,
            "audit_excluded_content_identities": len(excluded_content),
            "source_disjoint_from_audit_populations": True,
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
            "batch_receipts": {
                "path": batches_path.name,
                "rows": len(batch_receipts),
                "bytes": batches_path.stat().st_size,
                "sha256": sha256_file(batches_path),
                "ordered_rows_sha256": canonical_sha256(batch_receipts),
            },
            "by_source": {SOURCE_ID: len(candidates)},
            "by_stratum": dict(sorted(by_stratum.items())),
            "dataset_server_batches": len(batch_receipts),
            "source_declared_cc0": True,
            "rights_verification_complete": False,
            "raw_source_text_redistributable_by_sai": False,
            "benchmark_contamination_screen_complete": False,
            "hermes_judgments_complete": False,
            "quality_compilation_complete": False,
            "full_source_ingestion_authorized": False,
            "training_ready": False,
            "four_b_training_authorized": False,
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
    parser.add_argument("--audit-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_population(args.output_root, args.audit_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
