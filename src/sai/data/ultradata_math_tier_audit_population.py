"""Acquire a deterministic audit of UltraData Math L2 and L3 data tiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import CANDIDATE_SCHEMA, normalize_candidate
from sai.data.reservoir_audit_population import (
    LINEAGE_SCHEMA,
    SCHEMA,
    _excerpt,
    _write_jsonl,
)
from sai.data.token_stream import canonical_sha256, sha256_file

REPOSITORY = "openbmb/UltraData-Math"
REVISION = "fe10db8efd35597fd7fcff8ff576b5ec4ea5ff87"
LICENSE = "apache-2.0_project_upstream_source_terms_apply"
DATASET_SERVER = "https://datasets-server.huggingface.co"
SEED = 20260825
ROWS_PER_STRATUM = 32
BATCH_ROWS = 8
SOURCE_ID = "ultradata_math_l2_l3"


@dataclass(frozen=True)
class TierSpec:
    config: str
    stratum: str
    source_type: str
    expected_rows: int
    published_tokens: int


TIER_SPECS = (
    TierSpec(
        "UltraData-Math-L2-preview",
        "l2_quality_selected_web_math",
        "educational_web",
        13_707_851,
        33_700_000_000,
    ),
    TierSpec(
        "UltraData-Math-L3-Conversation-Synthetic",
        "l3_conversation_synthetic",
        "synthetic",
        16_881_040,
        22_000_000_000,
    ),
    TierSpec(
        "UltraData-Math-L3-Multi-Style-Synthetic",
        "l3_multi_style_synthetic",
        "synthetic",
        11_445_446,
        22_000_000_000,
    ),
    TierSpec(
        "UltraData-Math-L3-QA-Synthetic",
        "l3_qa_synthetic",
        "synthetic",
        27_113_894,
        22_000_000_000,
    ),
    TierSpec(
        "UltraData-Math-L3-Textbook-Exercise-Synthetic",
        "l3_textbook_exercise_synthetic",
        "synthetic",
        26_005_670,
        22_000_000_000,
    ),
)
EXPECTED_ROWS = len(TIER_SPECS) * ROWS_PER_STRATUM
EXPECTED_BATCHES = EXPECTED_ROWS // BATCH_ROWS
if EXPECTED_ROWS != 160 or EXPECTED_BATCHES != 20:  # pragma: no cover
    raise RuntimeError("UltraData Math tier audit geometry differs")


class UltraDataMathTierAuditError(RuntimeError):
    """The source revision, row geometry, or response custody differs."""


def _selection_key(config: str, batch_index: int) -> str:
    return hashlib.sha256(f"{SEED}:{config}:{batch_index}".encode()).hexdigest()


def build_batch_plan() -> list[dict[str, Any]]:
    """Freeze four non-overlapping eight-row windows per tier."""

    plan: list[dict[str, Any]] = []
    for spec in TIER_SPECS:
        occupied: list[tuple[int, int]] = []
        for batch_index in range(ROWS_PER_STRATUM // BATCH_ROWS):
            nonce = 0
            while True:
                key = hashlib.sha256(
                    f"{_selection_key(spec.config, batch_index)}:{nonce}".encode()
                ).hexdigest()
                offset = int(key, 16) % (spec.expected_rows - BATCH_ROWS + 1)
                interval = (offset, offset + BATCH_ROWS)
                if all(
                    interval[1] <= prior[0] or interval[0] >= prior[1]
                    for prior in occupied
                ):
                    break
                nonce += 1
            occupied.append(interval)
            plan.append(
                {
                    "ordinal": len(plan),
                    "config": spec.config,
                    "stratum": spec.stratum,
                    "source_type": spec.source_type,
                    "expected_rows": spec.expected_rows,
                    "published_tokens": spec.published_tokens,
                    "batch_index": batch_index,
                    "offset": offset,
                    "length": BATCH_ROWS,
                    "selection_key": key,
                }
            )
    if len(plan) != EXPECTED_BATCHES:
        raise UltraDataMathTierAuditError("tier audit batch geometry differs")
    return plan


def _resolve_revision() -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise UltraDataMathTierAuditError("huggingface_hub is required") from error
    return HfApi().dataset_info(REPOSITORY, revision=REVISION).sha


def _fetch_batch(plan: dict[str, Any]) -> dict[str, Any]:
    try:
        import requests
    except ImportError as error:
        raise UltraDataMathTierAuditError("requests is required") from error
    params = {
        "dataset": REPOSITORY,
        "config": plan["config"],
        "split": "train",
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
        raise UltraDataMathTierAuditError(
            "UltraData Math dataset-server request failed"
        ) from error
    return {
        "x_revision": response.headers.get("x-revision"),
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
        "payload": payload,
        "request_url": response.url,
    }


def _validate_batch(
    plan: dict[str, Any], result: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = result.get("payload")
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if (
        result.get("x_revision") != REVISION
        or not isinstance(result.get("response_sha256"), str)
        or len(result["response_sha256"]) != 64
        or not isinstance(result.get("request_url"), str)
        or not isinstance(rows, list)
        or len(rows) != plan["length"]
        or payload.get("num_rows_total") != plan["expected_rows"]
    ):
        raise UltraDataMathTierAuditError("tier audit response geometry differs")
    validated = []
    for local_index, item in enumerate(rows):
        expected_index = plan["offset"] + local_index
        row = item.get("row") if isinstance(item, dict) else None
        content = row.get("content") if isinstance(row, dict) else None
        uid = row.get("uid") if isinstance(row, dict) else None
        quality_label = row.get("quality_label") if isinstance(row, dict) else None
        if (
            item.get("row_idx") != expected_index
            or not isinstance(content, str)
            or len(content.strip().encode()) < 200
            or (uid is not None and not isinstance(uid, str))
            or (quality_label is not None and not isinstance(quality_label, int))
        ):
            raise UltraDataMathTierAuditError("tier audit source row differs")
        validated.append(
            {
                "row_index": expected_index,
                "content": content.strip(),
                "native_id": uid,
                "quality_label": quality_label,
            }
        )
    receipt = {
        "schema": "sai-ultradata-math-dataset-server-batch-v1",
        "status": "complete",
        "repository": REPOSITORY,
        "revision": REVISION,
        "config": plan["config"],
        "split": "train",
        "offset": plan["offset"],
        "length": plan["length"],
        "selection_key": plan["selection_key"],
        "request_url": result["request_url"],
        "response_sha256": result["response_sha256"],
        "x_revision": result["x_revision"],
        "row_indexes": [row["row_index"] for row in validated],
        "source_text_persisted": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return validated, receipt


def _candidate_and_lineage(
    plan: dict[str, Any], row: dict[str, Any], ordinal: int, response_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    full_text = row["content"]
    excerpt, excerpt_method = _excerpt(full_text)
    full_text_sha256 = hashlib.sha256(full_text.encode()).hexdigest()
    locator = {
        "dataset_server": DATASET_SERVER,
        "config": plan["config"],
        "split": "train",
        "row_index": row["row_index"],
        "native_id": row["native_id"],
        "quality_label": row["quality_label"],
        "response_sha256": response_sha256,
    }
    row_id = canonical_sha256(
        {
            "repository": REPOSITORY,
            "revision": REVISION,
            "config": plan["config"],
            "split": "train",
            "row_index": row["row_index"],
            "native_id": row["native_id"],
        }
    )
    provenance = canonical_sha256(
        {
            "locator": locator,
            "full_text_sha256": full_text_sha256,
            "excerpt_method": excerpt_method,
            "excerpt_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
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
            "source_type": plan["source_type"],
        },
        "source_content_sha256": hashlib.sha256(excerpt.encode()).hexdigest(),
        "provenance_sha256": provenance,
    }
    candidate["candidate_identity_sha256"] = canonical_sha256(candidate)
    candidate = normalize_candidate(candidate)
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "ordinal": ordinal,
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "source_id": SOURCE_ID,
        "stratum": plan["stratum"],
        "selection_key": canonical_sha256(
            {"batch": plan["selection_key"], "row_index": row["row_index"]}
        ),
        "repository": REPOSITORY,
        "revision": REVISION,
        "license": LICENSE,
        "access": "public",
        "locator": locator,
        "full_file_content_verified": False,
        "full_text_bytes": len(full_text.encode()),
        "full_text_sha256": full_text_sha256,
        "excerpt_method": excerpt_method,
        "excerpt_bytes": len(excerpt.encode()),
        "excerpt_sha256": candidate["source_content_sha256"],
        "raw_source_is_training_ready": False,
    }
    lineage["lineage_sha256"] = canonical_sha256(lineage)
    return candidate, lineage


def build_population(
    output_root: Path,
    *,
    revision_resolver: Callable[[], str] = _resolve_revision,
    batch_fetcher: Callable[[dict[str, Any]], dict[str, Any]] = _fetch_batch,
) -> dict[str, Any]:
    """Acquire and seal the 160-row tier audit without admitting any row."""

    if output_root.exists() or output_root.is_symlink():
        raise UltraDataMathTierAuditError("tier audit output boundary differs")
    if revision_resolver() != REVISION:
        raise UltraDataMathTierAuditError("UltraData Math revision differs")
    candidates: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    batch_receipts: list[dict[str, Any]] = []
    for batch_number, plan in enumerate(build_batch_plan(), start=1):
        rows, batch_receipt = _validate_batch(plan, batch_fetcher(plan))
        batch_receipts.append(batch_receipt)
        for row in rows:
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
                    "event": "ultradata_math_tier_audit_progress",
                    "batches": batch_number,
                    "rows": len(candidates),
                    "remaining_rows": EXPECTED_ROWS - len(candidates),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if revision_resolver() != REVISION:
        raise UltraDataMathTierAuditError("UltraData Math revision changed")
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if len(candidates) != EXPECTED_ROWS or len(identities) != len(set(identities)):
        raise UltraDataMathTierAuditError("tier audit identities differ")

    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise UltraDataMathTierAuditError("tier audit temporary output exists")
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
                "five_frozen_l2_l3_strata_four_sha256_windows_per_stratum"
            ),
            "statistically_representative": False,
            "screen_only": True,
            "source_snapshot": {
                "repository": REPOSITORY,
                "revision": REVISION,
                "license": LICENSE,
                "published_l2_tokens": 33_700_000_000,
                "published_l3_tokens": 88_000_000_000,
                "published_tokens_total": 121_700_000_000,
                "published_tokens_are_not_training_ready_tokens": True,
            },
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
            "raw_source_text_redistributable_by_sai": False,
            "benchmark_contamination_screen_complete": False,
            "hermes_judgments_complete": False,
            "quality_compilation_complete": False,
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
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_population(args.output_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
