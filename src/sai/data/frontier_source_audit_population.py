"""Acquire a disjoint audit population from modern frontier source candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.frontier_source_reservoir import (
    MANIFEST_SCHEMA,
    RECEIPT_SCHEMA,
    SOURCE_SPECS,
)
from sai.data.reservoir_audit_population import (
    SCHEMA,
    _acquire_one,
    _candidate_and_lineage,
    _write_jsonl,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SEED = 20260824


@dataclass(frozen=True)
class AuditStratum:
    source_id: str
    stratum: str
    path_prefix: str
    quota: int
    source_type: str


FINEWEB2_LANGUAGES = (
    "rus_Cyrl",
    "cmn_Hani",
    "deu_Latn",
    "spa_Latn",
    "jpn_Jpan",
    "fra_Latn",
    "ita_Latn",
    "por_Latn",
    "pol_Latn",
    "nld_Latn",
    "ind_Latn",
    "tur_Latn",
    "ces_Latn",
    "arb_Arab",
    "fas_Arab",
    "hun_Latn",
    "swe_Latn",
    "ell_Grek",
    "dan_Latn",
    "vie_Latn",
)

AUDIT_STRATA = (
    AuditStratum(
        "ultrafineweb_l2_en_20260820",
        "current_l2_english",
        "data/ultrafineweb_l1_en_hq/",
        120,
        "educational_web",
    ),
    AuditStratum(
        "ultrafineweb_l2_en_2025",
        "benchmark_validated_l2_english",
        "data/ultrafineweb_en/",
        80,
        "educational_web",
    ),
    *(
        AuditStratum(
            "fineweb2_hq_multilingual",
            f"translation_discovery:{language}",
            f"{language}/",
            8,
            "general_web",
        )
        for language in FINEWEB2_LANGUAGES
    ),
    AuditStratum(
        "nemotron_specialized_reasoning",
        "grounded_rqa",
        "Nemotron-Pretraining-RQA/",
        52,
        "synthetic",
    ),
    AuditStratum(
        "nemotron_specialized_reasoning",
        "cross_domain_infinibyte",
        "Nemotron-Pretraining-InfiniByte-Reasoning/",
        30,
        "synthetic",
    ),
    AuditStratum(
        "nemotron_specialized_reasoning",
        "math_textbooks",
        "Nemotron-Pretraining-Math-Textbooks/",
        13,
        "synthetic",
    ),
    AuditStratum(
        "nemotron_specialized_reasoning",
        "scientific_coding",
        "Nemotron-Pretraining-Scientific-Coding/",
        1,
        "synthetic",
    ),
    AuditStratum(
        "ultradata_math_l1",
        "filtered_math_l1",
        "data/UltraData-Math-L1/",
        56,
        "educational_web",
    ),
)

EXPECTED_ROWS = sum(stratum.quota for stratum in AUDIT_STRATA)
if EXPECTED_ROWS != 512:  # pragma: no cover - frozen contract
    raise RuntimeError("frontier audit geometry differs")


class FrontierSourceAuditError(RuntimeError):
    """Frontier reservoir identity, audit geometry, or acquisition differs."""


def load_frontier_reservoir(
    manifest_path: Path, receipt_path: Path
) -> list[dict[str, Any]]:
    """Replay the exact source-candidate manifest without admitting it."""

    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
    ):
        raise FrontierSourceAuditError("frontier reservoir is missing or unsafe")
    try:
        rows = [json.loads(line) for line in manifest_path.open()]
        receipt_rows = [json.loads(line) for line in receipt_path.open()]
    except (OSError, json.JSONDecodeError) as error:
        raise FrontierSourceAuditError(
            "frontier reservoir cannot be decoded"
        ) from error
    if not rows or len(receipt_rows) != 1:
        raise FrontierSourceAuditError("frontier reservoir is empty or duplicated")
    receipt = receipt_rows[0]
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    specs = {spec.source_id: spec for spec in SOURCE_SPECS}
    identities = set()
    for ordinal, row in enumerate(rows):
        spec = specs.get(row.get("source_id"))
        identity = (row.get("repository"), row.get("path"))
        if (
            spec is None
            or row.get("schema") != MANIFEST_SCHEMA
            or row.get("repository") != spec.repository
            or row.get("revision") != spec.revision
            or row.get("text_column") != spec.text_column
            or row.get("license") != spec.license
            or row.get("access") != spec.access
            or row.get("epistemic_function") != spec.epistemic_function
            or row.get("ordinal") != ordinal
            or row.get("physical_bytes_are_text_payload_bytes") is not False
            or row.get("source_candidate_is_training_ready") is not False
            or identity in identities
        ):
            raise FrontierSourceAuditError("frontier manifest row differs")
        identities.add(identity)
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "complete"
        or receipt.get("selected_files") != len(rows)
        or receipt.get("selected_physical_bytes")
        != sum(row["physical_bytes"] for row in rows)
        or receipt.get("manifest", {}).get("sha256") != sha256_file(manifest_path)
        or receipt.get("manifest", {}).get("ordered_rows_sha256")
        != canonical_sha256(rows)
        or receipt.get("training_ready") is not False
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise FrontierSourceAuditError("frontier receipt differs")
    return rows


def _selection_key(stratum: AuditStratum, row: dict[str, Any]) -> str:
    return hashlib.sha256(
        (
            f"{SEED}:{stratum.source_id}:{stratum.stratum}:"
            f"{row['repository']}:{row['path']}:{row['sha256']}"
        ).encode()
    ).hexdigest()


def build_plan(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select exact parent files across every frozen modern-source stratum."""

    plan = []
    for stratum in AUDIT_STRATA:
        matches = [
            row
            for row in rows
            if row["source_id"] == stratum.source_id
            and row["path"].startswith(stratum.path_prefix)
        ]
        ranked = sorted(
            matches, key=lambda row: (_selection_key(stratum, row), row["path"])
        )
        if len(ranked) < stratum.quota:
            raise FrontierSourceAuditError(
                f"frontier audit stratum is underfilled: {stratum.stratum}"
            )
        for row in ranked[: stratum.quota]:
            plan.append(
                {
                    "ordinal": len(plan),
                    "source_id": row["source_id"],
                    "stratum": stratum.stratum,
                    "source_type": stratum.source_type,
                    "repository": row["repository"],
                    "revision": row["revision"],
                    "license": row["license"],
                    "access": row["access"],
                    "path": row["path"],
                    "parent_file_bytes": row["physical_bytes"],
                    "parent_file_sha256": row["sha256"],
                    "text_column": row["text_column"],
                    "selection_key": _selection_key(stratum, row),
                }
            )
    parents = [(row["repository"], row["path"]) for row in plan]
    if len(plan) != EXPECTED_ROWS or len(parents) != len(set(parents)):
        raise FrontierSourceAuditError("frontier audit plan identity differs")
    return plan


def build_population(
    manifest_path: Path,
    reservoir_receipt_path: Path,
    output_root: Path,
    *,
    token: str,
) -> dict[str, Any]:
    """Acquire and seal the 512-row modern-source comparison population."""

    if not token or output_root.exists() or output_root.is_symlink():
        raise FrontierSourceAuditError(
            "frontier audit credential or output boundary differs"
        )
    rows = load_frontier_reservoir(manifest_path, reservoir_receipt_path)
    plan = build_plan(rows)
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
                        "event": "frontier_source_audit_acquisition_progress",
                        "acquired": index,
                        "remaining": len(plan) - index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    identities = [row["candidate_identity_sha256"] for row in candidates]
    if len(candidates) != EXPECTED_ROWS or len(identities) != len(set(identities)):
        raise FrontierSourceAuditError("frontier audit candidates differ")
    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise FrontierSourceAuditError("frontier audit temporary output exists")
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
                "frozen_modern_source_strata_then_lowest_sha256_parent_files_"
                "and_deterministic_source_rows"
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
            "by_stratum": dict(sorted(by_stratum.items())),
            "range_read_parent_files": len(lineage),
            "fully_verified_parent_files": 0,
            "cross_reservoir_overlap_resolved": False,
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
