"""Freeze a deterministic cross-source population for independent model review."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.nous_label_worker import _load_jsonl
from sai.data.reservoir_audit_aggregate import _validate_compiler_receipt
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-independent-review-population-receipt-v1"
STRATA = (
    "nonretain",
    "severe_risk_retain",
    "cleanup_risk_retain",
    "clean_retain",
)
SEVERE_RISKS = frozenset(
    {
        "seo_or_content_farm",
        "incoherent_or_corrupted",
        "answer_farm_without_teaching",
        "personal_or_secret_data",
        "license_or_provenance_unclear",
    }
)
CLEANUP_RISKS = frozenset({"ocr_or_extraction_damage", "duplicated_boilerplate"})


class IndependentReviewPopulationError(RuntimeError):
    """The source receipts, selection strata, or output custody differs."""


@dataclass(frozen=True)
class Lane:
    name: str
    candidates: Path
    judgments: Path


def classify(judgment: dict[str, Any]) -> str | None:
    """Assign one mutually exclusive review stratum."""

    verdict = judgment.get("verdict")
    risks = judgment.get("risks")
    if verdict not in {"retain", "review", "reject"} or not isinstance(risks, dict):
        raise IndependentReviewPopulationError("review judgment differs")
    active = {key for key, value in risks.items() if value is True}
    if verdict != "retain":
        return "nonretain"
    if active & SEVERE_RISKS:
        return "severe_risk_retain"
    if active & CLEANUP_RISKS:
        return "cleanup_risk_retain"
    if not active:
        return "clean_retain"
    return None


def select_rows(
    rows: list[tuple[str, dict[str, Any], dict[str, Any]]], per_stratum: int
) -> list[tuple[str, str, dict[str, Any], dict[str, Any]]]:
    """Select the lowest exact identities in each source×stratum cell."""

    if (
        isinstance(per_stratum, bool)
        or not isinstance(per_stratum, int)
        or per_stratum <= 0
    ):
        raise IndependentReviewPopulationError("review geometry differs")
    cells: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for lane, candidate, receipt in rows:
        stratum = classify(receipt["judgment"])
        if stratum is None:
            continue
        cells.setdefault((lane, stratum), []).append((candidate, receipt))
    selected = []
    for (lane, stratum), members in sorted(cells.items()):
        members.sort(key=lambda item: item[0]["candidate_identity_sha256"])
        for candidate, receipt in members[:per_stratum]:
            selected.append((lane, stratum, candidate, receipt))
    selected.sort(key=lambda item: item[2]["candidate_identity_sha256"])
    identities = [item[2]["candidate_identity_sha256"] for item in selected]
    if not selected or len(identities) != len(set(identities)):
        raise IndependentReviewPopulationError("review identity coverage differs")
    return selected


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    stage = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with stage.open("x") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, path)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise


def build_population(
    lanes: list[Lane], output_root: Path, per_stratum: int
) -> dict[str, Any]:
    """Build a local text-bearing review population and source-safe receipt."""

    if (
        not lanes
        or len({lane.name for lane in lanes}) != len(lanes)
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise IndependentReviewPopulationError("review population inputs differ")
    all_rows = []
    snapshots = []
    for lane in lanes:
        candidates = _load_jsonl(lane.candidates)
        by_identity = {row["candidate_identity_sha256"]: row for row in candidates}
        if len(candidates) != len(by_identity):
            raise IndependentReviewPopulationError("review candidates differ")
        validated = []
        for path in sorted(lane.judgments.glob("*.compiler.json")):
            identity = path.name.removesuffix(".compiler.json")
            candidate = by_identity.get(identity)
            if candidate is None:
                raise IndependentReviewPopulationError(
                    "review receipt identity differs"
                )
            try:
                receipt = _validate_compiler_receipt(
                    json.loads(path.read_bytes()), candidate
                )
            except (OSError, json.JSONDecodeError, RuntimeError) as error:
                raise IndependentReviewPopulationError(
                    "review receipt custody differs"
                ) from error
            validated.append((candidate, receipt))
            all_rows.append((lane.name, candidate, receipt))
        if not validated:
            raise IndependentReviewPopulationError("review lane is empty")
        snapshots.append(
            {
                "lane": lane.name,
                "candidate_file_sha256": sha256_file(lane.candidates),
                "completed_receipts": len(validated),
                "ordered_completed_receipts_sha256": canonical_sha256(
                    [receipt["receipt_sha256"] for _, receipt in validated]
                ),
            }
        )
    selected = select_rows(all_rows, per_stratum)
    output_root.mkdir(parents=True)
    try:
        candidate_path = output_root / "candidates.jsonl"
        _atomic_jsonl(candidate_path, [item[2] for item in selected])
        cells: Counter[str] = Counter(
            f"{lane}::{stratum}" for lane, stratum, _, _ in selected
        )
        selected_descriptors = [
            {
                "lane": lane,
                "stratum": stratum,
                "candidate_identity_sha256": candidate["candidate_identity_sha256"],
                "primary_receipt_sha256": receipt["receipt_sha256"],
                "primary_judgment_sha256": receipt["judgment"]["judgment_sha256"],
            }
            for lane, stratum, candidate, receipt in selected
        ]
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_independent_review_population",
            "per_source_stratum_limit": per_stratum,
            "source_snapshots": snapshots,
            "population": {
                "path": candidate_path.name,
                "rows": len(selected),
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
                "ordered_identities_sha256": canonical_sha256(
                    [item[2]["candidate_identity_sha256"] for item in selected]
                ),
            },
            "selected_cells": dict(sorted(cells.items())),
            "selected_descriptors": selected_descriptors,
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


def _lane(value: str) -> Lane:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0]:
        raise argparse.ArgumentTypeError("lane must be NAME=CANDIDATES,JUDGMENTS")
    paths = parts[1].split(",", 1)
    if len(paths) != 2:
        raise argparse.ArgumentTypeError("lane must be NAME=CANDIDATES,JUDGMENTS")
    return Lane(parts[0], Path(paths[0]), Path(paths[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", action="append", type=_lane, required=True)
    parser.add_argument("--per-stratum", type=int, default=4)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_population(args.lane, args.output_root, args.per_stratum)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
