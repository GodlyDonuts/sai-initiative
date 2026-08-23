"""Acquire a deterministic index-stratified screen of licensed PubMed full text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
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

REPOSITORY = "common-pile/pubmed_filtered"
REVISION = "c156f0569a92d8f2edc33cebe1f72f7d3e1cae84"
SOURCE_ID = "common_pile_pubmed"
SOURCE_TYPE = "research_paper"
DATASET_SERVER = "https://datasets-server.huggingface.co"
CONFIG = "default"
SPLIT = "train"
SEED = 20260825
SOURCE_ROWS = 163_259
SOURCE_ORIGINAL_BYTES = 43_555_182_605
SOURCE_MEMORY_BYTES = 5_006_210_897
SOURCE_FILES = 17
INDEX_STRATA = 32
FETCH_ROWS_PER_STRATUM = 64
WINDOWS_PER_STRATUM = 4
SELECT_ROWS_PER_STRATUM = 32
EXPECTED_ROWS = INDEX_STRATA * SELECT_ROWS_PER_STRATUM
EXPECTED_AUDIT_EXCLUSIONS = 4
MINIMUM_TEXT_BYTES = 200
MAXIMUM_TEXT_BYTES = 2 * 1024 * 1024
REQUEST_MAXIMUM_ATTEMPTS = 8
REQUEST_BACKOFF_SECONDS = 2
CC_BY_DECLARATION = (
    "Creative Commons - Attribution - https://creativecommons.org/licenses/by/4.0/"
)
CC0_DECLARATION = (
    "Creative Commons Zero - Public Domain - "
    "https://creativecommons.org/publicdomain/zero/1.0/"
)
LICENSE_MAP = {
    CC_BY_DECLARATION: "CC-BY-4.0",
    CC0_DECLARATION: "CC0-1.0",
}


class PubmedFulltextAuditError(RuntimeError):
    """The exact PubMed source, audit geometry, or response custody differs."""


def _resolve_revision() -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise PubmedFulltextAuditError("huggingface_hub is required") from error
    return HfApi().dataset_info(REPOSITORY, revision=REVISION).sha


def build_batch_plan() -> list[dict[str, Any]]:
    """Choose four fixed, nonoverlapping 64-row windows per index stratum."""

    plan = []
    for stratum_index in range(INDEX_STRATA):
        start = SOURCE_ROWS * stratum_index // INDEX_STRATA
        end = SOURCE_ROWS * (stratum_index + 1) // INDEX_STRATA
        stratum_key = hashlib.sha256(
            f"{SEED}:{REPOSITORY}:{REVISION}:{stratum_index}".encode()
        ).hexdigest()
        for window_index in range(WINDOWS_PER_STRATUM):
            window_start = start + (end - start) * window_index // WINDOWS_PER_STRATUM
            window_end = (
                start + (end - start) * (window_index + 1) // WINDOWS_PER_STRATUM
            )
            maximum_offset = window_end - FETCH_ROWS_PER_STRATUM
            if maximum_offset < window_start:
                raise PubmedFulltextAuditError("PubMed index window is too small")
            window_key = hashlib.sha256(
                (
                    f"{SEED}:{REPOSITORY}:{REVISION}:{stratum_index}:{window_index}"
                ).encode()
            ).hexdigest()
            offset = window_start + int(window_key, 16) % (
                maximum_offset - window_start + 1
            )
            plan.append(
                {
                    "ordinal": len(plan),
                    "stratum_index": stratum_index,
                    "stratum": f"index_{stratum_index:02d}",
                    "stratum_start": start,
                    "stratum_end": end,
                    "window_index": window_index,
                    "window_start": window_start,
                    "window_end": window_end,
                    "offset": offset,
                    "length": FETCH_ROWS_PER_STRATUM,
                    "stratum_selection_key": stratum_key,
                    "window_selection_key": window_key,
                }
            )
    return plan


def _fetch_batch(plan: dict[str, Any]) -> dict[str, Any]:
    try:
        import requests
    except ImportError as error:
        raise PubmedFulltextAuditError("requests is required") from error
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    retry_statuses: list[int | str] = []
    response = None
    payload = None
    last_error: Exception | None = None
    for attempt in range(1, REQUEST_MAXIMUM_ATTEMPTS + 1):
        try:
            response = requests.get(
                f"{DATASET_SERVER}/rows",
                params={
                    "dataset": REPOSITORY,
                    "config": CONFIG,
                    "split": SPLIT,
                    "offset": plan["offset"],
                    "length": plan["length"],
                },
                headers=headers,
                timeout=120,
            )
            if response.status_code == 429 or response.status_code >= 500:
                retry_statuses.append(response.status_code)
                if attempt == REQUEST_MAXIMUM_ATTEMPTS:
                    response.raise_for_status()
                retry_after = response.headers.get("retry-after")
                try:
                    retry_after_seconds = int(retry_after) if retry_after else 0
                except ValueError:
                    retry_after_seconds = 0
                time.sleep(
                    min(
                        30,
                        max(
                            retry_after_seconds,
                            REQUEST_BACKOFF_SECONDS * 2 ** (attempt - 1),
                        ),
                    )
                )
                continue
            response.raise_for_status()
            payload = response.json()
            break
        except (requests.RequestException, json.JSONDecodeError) as error:
            last_error = error
            retry_statuses.append(type(error).__name__)
            if attempt == REQUEST_MAXIMUM_ATTEMPTS:
                raise PubmedFulltextAuditError(
                    "PubMed dataset-server request failed"
                ) from error
            time.sleep(min(30, REQUEST_BACKOFF_SECONDS * 2 ** (attempt - 1)))
    if response is None or payload is None:
        raise PubmedFulltextAuditError(
            "PubMed dataset-server request failed"
        ) from last_error
    return {
        "x_revision": response.headers.get("x-revision"),
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
        "request_url": response.url,
        "request_attempts": len(retry_statuses) + 1,
        "retry_statuses": retry_statuses,
        "payload": payload,
    }


def _validate_batch(
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
        raise PubmedFulltextAuditError("PubMed response geometry differs")
    eligible = []
    counters: Counter[str] = Counter()
    declared_licenses: Counter[str] = Counter()
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
            or declared_license not in LICENSE_MAP
        ):
            raise PubmedFulltextAuditError("PubMed source row differs")
        declared_licenses[declared_license] += 1
        text = text.strip()
        text_bytes = len(text.encode())
        if text_bytes < MINIMUM_TEXT_BYTES:
            counters["short_rows"] += 1
            continue
        if text_bytes > MAXIMUM_TEXT_BYTES:
            counters["oversized_rows"] += 1
            continue
        excerpt, _method = _excerpt(text)
        excerpt_sha256 = hashlib.sha256(excerpt.encode()).hexdigest()
        if excerpt_sha256 in excluded_content_sha256s:
            counters["audit_excluded_rows"] += 1
            continue
        selection_key = canonical_sha256(
            {
                "seed": SEED,
                "stratum_selection_key": plan["stratum_selection_key"],
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
                "canonical_license": LICENSE_MAP[declared_license],
                "response_sha256": result["response_sha256"],
            }
        )
        counters["eligible_rows"] += 1
    receipt = {
        "schema": "sai-pubmed-fulltext-dataset-server-batch-v2",
        "status": "complete",
        "repository": REPOSITORY,
        "revision": REVISION,
        "config": CONFIG,
        "split": SPLIT,
        "stratum_index": plan["stratum_index"],
        "stratum_start": plan["stratum_start"],
        "stratum_end": plan["stratum_end"],
        "window_index": plan["window_index"],
        "window_start": plan["window_start"],
        "window_end": plan["window_end"],
        "offset": plan["offset"],
        "fetched_rows": FETCH_ROWS_PER_STRATUM,
        "eligible_rows": counters["eligible_rows"],
        "short_rows": counters["short_rows"],
        "oversized_rows": counters["oversized_rows"],
        "audit_excluded_rows": counters["audit_excluded_rows"],
        "declared_license_counts": dict(sorted(declared_licenses.items())),
        "stratum_selection_key": plan["stratum_selection_key"],
        "window_selection_key": plan["window_selection_key"],
        "request_url": result["request_url"],
        "request_attempts": result.get("request_attempts", 1),
        "retry_statuses": result.get("retry_statuses", []),
        "response_sha256": result["response_sha256"],
        "x_revision": result["x_revision"],
        "source_text_persisted": False,
    }
    return eligible, receipt


def _select_stratum(
    eligible: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select the fixed bottom-32 identities from one four-window pool."""

    row_indexes = [row["row_index"] for row in eligible]
    if len(row_indexes) != len(set(row_indexes)):
        raise PubmedFulltextAuditError("PubMed stratum windows overlap")
    selected = sorted(eligible, key=lambda row: row["selection_key"])[
        :SELECT_ROWS_PER_STRATUM
    ]
    if len(selected) != SELECT_ROWS_PER_STRATUM:
        raise PubmedFulltextAuditError("PubMed stratum lacks eligible source rows")
    return sorted(selected, key=lambda row: row["row_index"])


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
        "journal": metadata.get("journal"),
        "created": metadata.get("created"),
        "declared_license": metadata["license"],
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
            "license": row["canonical_license"],
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
        "license": row["canonical_license"],
        "declared_license": metadata["license"],
        "access": "public",
        "locator": locator,
        "full_file_content_verified": False,
        "full_text_bytes": len(full_text.encode()),
        "full_text_sha256": full_text_sha256,
        "excerpt_method": excerpt_method,
        "excerpt_bytes": len(excerpt.encode()),
        "excerpt_sha256": excerpt_sha256,
        "source_declared_reusable_license": True,
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
        raise PubmedFulltextAuditError("PubMed audit exclusions differ") from error
    if len(content) != EXPECTED_AUDIT_EXCLUSIONS:
        raise PubmedFulltextAuditError("PubMed audit exclusion coverage differs")
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
        raise PubmedFulltextAuditError("PubMed audit output boundary differs")
    excluded_content, audit_receipts = exclusion_loader(audit_roots)
    candidates: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    batch_receipts: list[dict[str, Any]] = []
    plans = build_batch_plan()
    for stratum_index in range(INDEX_STRATA):
        stratum_plans = [
            plan for plan in plans if plan["stratum_index"] == stratum_index
        ]
        if len(stratum_plans) != WINDOWS_PER_STRATUM:
            raise PubmedFulltextAuditError("PubMed stratum plan differs")
        eligible: list[dict[str, Any]] = []
        stratum_receipts: list[dict[str, Any]] = []
        for plan in stratum_plans:
            batch_eligible, batch_receipt = _validate_batch(
                plan, batch_fetcher(plan), excluded_content
            )
            eligible.extend(batch_eligible)
            stratum_receipts.append(batch_receipt)
        selected = _select_stratum(eligible)
        selected_row_indexes = {row["row_index"] for row in selected}
        for batch_receipt in stratum_receipts:
            selected_in_batch = sorted(
                row_index
                for row_index in selected_row_indexes
                if batch_receipt["offset"]
                <= row_index
                < batch_receipt["offset"] + batch_receipt["fetched_rows"]
            )
            batch_receipt["selected_rows"] = len(selected_in_batch)
            batch_receipt["selected_row_indexes"] = selected_in_batch
            batch_receipt["receipt_sha256"] = canonical_sha256(batch_receipt)
        batch_receipts.extend(stratum_receipts)
        stratum_plan = stratum_plans[0]
        for row in selected:
            candidate, source_lineage = _candidate_and_lineage(
                stratum_plan,
                row,
                len(candidates),
                row["response_sha256"],
            )
            candidates.append(candidate)
            lineage.append(source_lineage)
        print(
            json.dumps(
                {
                    "event": "pubmed_fulltext_audit_progress",
                    "completed_strata": stratum_index + 1,
                    "dataset_server_batches": len(batch_receipts),
                    "rows": len(candidates),
                    "remaining_rows": EXPECTED_ROWS - len(candidates),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if revision_resolver() != REVISION:
        raise PubmedFulltextAuditError("PubMed source revision changed")
    identities = [row["candidate_identity_sha256"] for row in candidates]
    content_identities = [row["source_content_sha256"] for row in candidates]
    if (
        len(candidates) != EXPECTED_ROWS
        or len(identities) != len(set(identities))
        or set(content_identities) & excluded_content
    ):
        raise PubmedFulltextAuditError("PubMed audit identities differ")

    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise PubmedFulltextAuditError("PubMed audit temporary output exists")
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
        by_license = Counter(row["license"] for row in lineage)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "seed": SEED,
            "selection_method": (
                "32_equal_index_strata_four_fixed_nonoverlapping_sha256_windows_"
                "and_bottom32_per_stratum"
            ),
            "statistically_representative": False,
            "screen_only": True,
            "source_snapshot": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "config": CONFIG,
                "split": SPLIT,
                "rows": SOURCE_ROWS,
                "source_files": SOURCE_FILES,
                "original_file_bytes": SOURCE_ORIGINAL_BYTES,
                "memory_bytes": SOURCE_MEMORY_BYTES,
                "recognized_licenses": sorted(set(LICENSE_MAP.values())),
            },
            "window_geometry": {
                "index_strata": INDEX_STRATA,
                "windows_per_stratum": WINDOWS_PER_STRATUM,
                "rows_per_window": FETCH_ROWS_PER_STRATUM,
                "fetched_rows_per_stratum": (
                    WINDOWS_PER_STRATUM * FETCH_ROWS_PER_STRATUM
                ),
                "selected_rows_per_stratum": SELECT_ROWS_PER_STRATUM,
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
            "by_license": dict(sorted(by_license.items())),
            "dataset_server_batches": len(batch_receipts),
            "rights_verification_complete": False,
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
