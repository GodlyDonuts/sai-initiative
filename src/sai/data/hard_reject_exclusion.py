"""Exclude exact source rows supported by sealed hard-reject judgments."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-hard-reject-exclusion-receipt-v1"
MANIFEST_SCHEMA = "sai-hard-reject-exclusion-record-v1"


class HardRejectExclusionError(RuntimeError):
    """The evidence, exact row mapping, or survivor custody differs."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise HardRejectExclusionError("hard-reject evidence is missing or unsafe")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise HardRejectExclusionError("hard-reject evidence is invalid") from error
    if not isinstance(value, dict):
        raise HardRejectExclusionError("hard-reject evidence is invalid")
    return value


def _verify_self_hash(value: dict[str, Any], field: str) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise HardRejectExclusionError("hard-reject evidence hash is missing")
    replay = dict(value)
    replay.pop(field)
    if canonical_sha256(replay) != claimed:
        raise HardRejectExclusionError("hard-reject evidence self-hash differs")
    return claimed


def _descriptor(path: Path, rows: int | None = None) -> dict[str, Any]:
    value = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _load_rejections(
    pilot_root: Path, judgments_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipt_path = pilot_root / "receipt.json"
    candidates_path = pilot_root / "candidates.jsonl"
    lineage_path = pilot_root / "lineage.jsonl"
    receipt = _load_json(receipt_path)
    receipt_sha256 = _verify_self_hash(receipt, "receipt_sha256")
    population = receipt.get("population")
    lineage_descriptor = receipt.get("lineage")
    if (
        receipt.get("status") != "complete_nontraining_compiler_population"
        or not isinstance(population, dict)
        or not isinstance(lineage_descriptor, dict)
        or population.get("sha256") != sha256_file(candidates_path)
        or population.get("bytes") != candidates_path.stat().st_size
        or lineage_descriptor.get("sha256") != sha256_file(lineage_path)
        or lineage_descriptor.get("bytes") != lineage_path.stat().st_size
    ):
        raise HardRejectExclusionError("sealed pilot population differs")

    candidate_rows: dict[str, dict[str, Any]] = {}
    with candidates_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            identity = value.get("candidate_identity_sha256")
            source = value.get("source")
            if (
                not isinstance(identity, str)
                or identity in candidate_rows
                or not isinstance(source, dict)
                or not isinstance(source.get("row_id"), str)
                or not isinstance(value.get("source_content_sha256"), str)
            ):
                raise HardRejectExclusionError("pilot candidate identity differs")
            candidate_rows[identity] = value

    lineage_rows: dict[str, dict[str, Any]] = {}
    with lineage_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            identity = value.get("candidate_identity_sha256")
            if not isinstance(identity, str) or identity in lineage_rows:
                raise HardRejectExclusionError("pilot lineage identity differs")
            lineage_rows[identity] = value
    if set(candidate_rows) != set(lineage_rows):
        raise HardRejectExclusionError("pilot candidate and lineage custody differs")

    rejections: dict[str, dict[str, Any]] = {}
    judgment_files = sorted(judgments_root.glob("*.compiler.json"))
    if not judgment_files:
        raise HardRejectExclusionError("hard-reject judgments are missing")
    for path in judgment_files:
        value = _load_json(path)
        judgment_receipt = _verify_self_hash(value, "receipt_sha256")
        identity = value.get("candidate_identity_sha256")
        judgment = value.get("judgment")
        if (
            value.get("status") != "complete"
            or not isinstance(identity, str)
            or identity not in candidate_rows
            or not isinstance(judgment, dict)
            or judgment.get("candidate_identity_sha256") != identity
        ):
            raise HardRejectExclusionError("compiler judgment custody differs")
        if judgment.get("verdict") != "reject":
            continue
        if (
            judgment.get("curriculum_phase") != "reject"
            or judgment.get("preservation_policy") != "reject"
        ):
            raise HardRejectExclusionError("hard-reject judgment is inconsistent")
        candidate = candidate_rows[identity]
        lineage = lineage_rows[identity]
        row_id = candidate["source"]["row_id"]
        if (
            lineage.get("row_id") != row_id
            or lineage.get("source_content_sha256")
            != candidate["source_content_sha256"]
            or row_id in rejections
        ):
            raise HardRejectExclusionError("hard-reject source-row mapping differs")
        record = {
            "schema": MANIFEST_SCHEMA,
            "row_id": row_id,
            "source_id": lineage.get("source_id"),
            "source_content_sha256": candidate["source_content_sha256"],
            "pilot_candidate_identity_sha256": identity,
            "judgment_file_sha256": sha256_file(path),
            "judgment_receipt_sha256": judgment_receipt,
            "verdict": "reject",
            "source_text_persisted": False,
        }
        record["record_sha256"] = canonical_sha256(record)
        rejections[row_id] = record
    if not rejections:
        raise HardRejectExclusionError("sealed pilot contains no hard rejects")
    evidence = {
        "pilot_root_name": pilot_root.name,
        "pilot_receipt_file_sha256": sha256_file(receipt_path),
        "pilot_receipt_sha256": receipt_sha256,
        "pilot_population_sha256": population["sha256"],
        "pilot_lineage_sha256": lineage_descriptor["sha256"],
        "judgments_root_name": judgments_root.name,
        "judgment_files": len(judgment_files),
        "hard_reject_rows": len(rejections),
    }
    return rejections, evidence


def _staged_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise HardRejectExclusionError("hard-reject output already exists")
    return path.with_name(f".{path.name}.tmp.{os.getpid()}")


def build_hard_reject_exclusion(
    candidate_paths: list[Path],
    attribution_paths: list[Path],
    pilot_root: Path,
    judgments_root: Path,
    output_candidates: Path,
    output_attribution: Path,
    exclusion_manifest: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Create byte-preserving survivors while excluding exact rejected row IDs."""

    if (
        not candidate_paths
        or not attribution_paths
        or len(candidate_paths) != len(set(candidate_paths))
        or len(attribution_paths) != len(set(attribution_paths))
    ):
        raise HardRejectExclusionError("hard-reject input set differs")
    for path in [*candidate_paths, *attribution_paths]:
        if not path.is_file() or path.is_symlink():
            raise HardRejectExclusionError("hard-reject input is missing or unsafe")
    candidate_inputs = [_descriptor(path) for path in candidate_paths]
    attribution_inputs = [_descriptor(path) for path in attribution_paths]
    stages = {
        output_candidates: _staged_path(output_candidates),
        output_attribution: _staged_path(output_attribution),
        exclusion_manifest: _staged_path(exclusion_manifest),
    }
    if receipt_path.exists() or receipt_path.is_symlink():
        raise HardRejectExclusionError("hard-reject output already exists")
    rejections, evidence = _load_rejections(pilot_root, judgments_root)
    counts: Counter[str] = Counter()
    for key in (
        "input_candidate_rows",
        "excluded_candidate_rows",
        "output_candidate_rows",
        "input_attribution_rows",
        "excluded_attribution_rows",
        "upstream_absent_attribution_rows",
        "output_attribution_rows",
    ):
        counts[key] = 0
    matched_candidates: Counter[str] = Counter()
    matched_attribution: Counter[str] = Counter()
    survivor_row_ids: set[str] = set()
    survivor_attribution: Counter[str] = Counter()
    try:
        with stages[output_candidates].open("x") as output:
            for path in candidate_paths:
                with path.open() as source:
                    for line in source:
                        if not line.strip():
                            continue
                        document = normalize_document(json.loads(line))
                        row_id = document["source"].get("row_id")
                        if not isinstance(row_id, str):
                            raise HardRejectExclusionError(
                                "candidate source row identity differs"
                            )
                        counts["input_candidate_rows"] += 1
                        if row_id in rejections:
                            matched_candidates[row_id] += 1
                            counts["excluded_candidate_rows"] += 1
                            continue
                        if row_id in survivor_row_ids:
                            raise HardRejectExclusionError(
                                "candidate source row identity is duplicated"
                            )
                        survivor_row_ids.add(row_id)
                        output.write(line if line.endswith("\n") else line + "\n")
                        counts["output_candidate_rows"] += 1

        with stages[output_attribution].open("x") as output:
            for path in attribution_paths:
                with path.open() as source:
                    for line in source:
                        if not line.strip():
                            continue
                        value = json.loads(line)
                        claimed = value.get("record_sha256")
                        replay = dict(value)
                        replay.pop("record_sha256", None)
                        row_id = value.get("row_id")
                        if (
                            not isinstance(row_id, str)
                            or not isinstance(claimed, str)
                            or canonical_sha256(replay) != claimed
                        ):
                            raise HardRejectExclusionError(
                                "attribution record identity differs"
                            )
                        counts["input_attribution_rows"] += 1
                        if row_id in rejections:
                            matched_attribution[row_id] += 1
                            counts["excluded_attribution_rows"] += 1
                            continue
                        if row_id not in survivor_row_ids:
                            counts["upstream_absent_attribution_rows"] += 1
                            continue
                        survivor_attribution[row_id] += 1
                        output.write(line if line.endswith("\n") else line + "\n")
                        counts["output_attribution_rows"] += 1

        if (
            not set(matched_candidates).issubset(rejections)
            or set(matched_attribution) != set(rejections)
            or any(value != 1 for value in matched_candidates.values())
            or any(value != 1 for value in matched_attribution.values())
            or set(survivor_attribution) != survivor_row_ids
            or any(value != 1 for value in survivor_attribution.values())
        ):
            raise HardRejectExclusionError("hard-reject row coverage differs")
        with stages[exclusion_manifest].open("x") as output:
            for row_id in sorted(rejections):
                record = dict(rejections[row_id])
                record.pop("record_sha256")
                record["candidate_input_occurrences"] = matched_candidates[row_id]
                record["attribution_input_occurrences"] = matched_attribution[row_id]
                record["record_sha256"] = canonical_sha256(record)
                output.write(json.dumps(record, sort_keys=True) + "\n")
        if any(
            path.stat().st_size != descriptor["bytes"]
            or sha256_file(path) != descriptor["sha256"]
            for path, descriptor in zip(
                [*candidate_paths, *attribution_paths],
                [*candidate_inputs, *attribution_inputs],
                strict=True,
            )
        ):
            raise HardRejectExclusionError("hard-reject input changed during replay")
        for destination, stage in stages.items():
            os.replace(stage, destination)
    except BaseException:
        for stage in stages.values():
            stage.unlink(missing_ok=True)
        for destination in stages:
            destination.unlink(missing_ok=True)
        raise

    payload = {
        "schema": SCHEMA,
        "status": "complete_exact_hard_reject_exclusion",
        "evidence": evidence,
        "inputs": {
            "candidates": candidate_inputs,
            "attribution": attribution_inputs,
        },
        "counts": dict(sorted(counts.items())),
        "outputs": {
            "candidates": _descriptor(
                output_candidates, counts["output_candidate_rows"]
            ),
            "attribution": _descriptor(
                output_attribution, counts["output_attribution_rows"]
            ),
            "exclusion_manifest": _descriptor(
                exclusion_manifest, len(rejections)
            )
            | {"contains_source_text": False},
        },
        "exact_rejected_source_rows_removed": True,
        "candidate_attribution_survivor_sets_identical": True,
        "source_text_persisted_in_exclusion_manifest": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    try:
        _atomic_create(receipt_path, payload)
    except BaseException:
        for path in stages:
            path.unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--attribution", type=Path, action="append", required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output-candidates", type=Path, required=True)
    parser.add_argument("--output-attribution", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build_hard_reject_exclusion(
        args.candidate,
        args.attribution,
        args.pilot_root,
        args.judgments_root,
        args.output_candidates,
        args.output_attribution,
        args.exclusion_manifest,
        args.receipt,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
