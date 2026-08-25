"""Seal exact quarantine identities from one complete Hermès source audit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.reservoir_audit_aggregate import (
    SCHEMA as AGGREGATE_SCHEMA,
)
from sai.data.reservoir_audit_aggregate import (
    _triage_route,
    _validate_compiler_receipt,
    load_population,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-audit-quarantine-manifest-receipt-v1"
RECORD_SCHEMA = "sai-audit-quarantine-exclusion-v1"
OUTPUT_SUFFIX = "compiler"


class AuditQuarantineManifestError(RuntimeError):
    """The aggregate, source identities, judgments, or output custody differs."""


def _load_aggregate(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise AuditQuarantineManifestError("audit aggregate is missing or unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise AuditQuarantineManifestError("audit aggregate is invalid") from error
    if not isinstance(value, dict):
        raise AuditQuarantineManifestError("audit aggregate is invalid")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    summary = value.get("summary")
    if (
        value.get("schema") != AGGREGATE_SCHEMA
        or value.get("status") != "complete"
        or value.get("training_ready") is not False
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or not isinstance(summary, dict)
        or summary.get("model_judgments_are_verified_admissions") is not False
        or summary.get("representation_verification_is_training_admission") is not False
    ):
        raise AuditQuarantineManifestError("audit aggregate receipt differs")
    return value


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


def build_quarantine_manifest(
    population_root: Path,
    judgments_root: Path,
    aggregate_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Emit no text, only exact identities barred from dataset materialization."""

    if output_root.exists() or output_root.is_symlink():
        raise AuditQuarantineManifestError("quarantine output already exists")
    aggregate = _load_aggregate(aggregate_path)
    candidates, lineage, population = load_population(population_root)
    expected_paths = {
        judgments_root
        / f"{candidate['candidate_identity_sha256']}.{OUTPUT_SUFFIX}.json"
        for candidate in candidates
    }
    if set(judgments_root.glob(f"*.{OUTPUT_SUFFIX}.json")) != expected_paths:
        raise AuditQuarantineManifestError("compiler judgment population differs")
    if aggregate.get("summary", {}).get("rows") != len(candidates):
        raise AuditQuarantineManifestError("aggregate population coverage differs")

    records = []
    for candidate, source in zip(candidates, lineage, strict=True):
        identity = candidate["candidate_identity_sha256"]
        path = judgments_root / f"{identity}.{OUTPUT_SUFFIX}.json"
        try:
            receipt = json.loads(path.read_bytes())
        except (OSError, json.JSONDecodeError) as error:
            raise AuditQuarantineManifestError(
                "compiler judgment is invalid"
            ) from error
        try:
            receipt = _validate_compiler_receipt(receipt, candidate)
        except RuntimeError as error:
            raise AuditQuarantineManifestError(
                "compiler judgment custody differs"
            ) from error
        judgment = receipt["judgment"]
        if _triage_route(judgment) != "quarantine":
            continue
        source_details = candidate.get("source")
        if not isinstance(source_details, dict) or not isinstance(
            source_details.get("row_id"), str
        ):
            raise AuditQuarantineManifestError("source row identity differs")
        active_risks = sorted(
            key for key, enabled in judgment["risks"].items() if enabled is True
        )
        record = {
            "schema": RECORD_SCHEMA,
            "candidate_identity_sha256": identity,
            "source_content_sha256": candidate["source_content_sha256"],
            "source_id": source["source_id"],
            "source_row_id": source_details["row_id"],
            "judgment_receipt_sha256": receipt["receipt_sha256"],
            "judgment_sha256": judgment["judgment_sha256"],
            "verdict": judgment["verdict"],
            "active_risks": active_risks,
            "route": "quarantine",
            "dataset_materialization_allowed": False,
            "source_text_persisted": False,
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)

    expected_quarantine = aggregate.get("summary", {}).get(
        "conservative_triage_routes", {}
    ).get("quarantine")
    if (
        isinstance(expected_quarantine, bool)
        or not isinstance(expected_quarantine, int)
        or expected_quarantine < 0
        or expected_quarantine != len(records)
    ):
        raise AuditQuarantineManifestError("quarantine row coverage differs")

    output_root.mkdir(parents=True)
    try:
        manifest_path = output_root / "quarantine_exclusions.jsonl"
        _atomic_jsonl(manifest_path, records)
        payload = {
            "schema": SCHEMA,
            "status": "complete_audit_quarantine_exclusion_manifest",
            "aggregate": {
                "path": aggregate_path.name,
                "bytes": aggregate_path.stat().st_size,
                "sha256": sha256_file(aggregate_path),
                "receipt_sha256": aggregate["receipt_sha256"],
            },
            "population": {
                "root_name": population_root.name,
                "receipt_sha256": population["receipt_sha256"],
                "rows": len(candidates),
            },
            "judgment_rows": len(candidates),
            "quarantine_rows": len(records),
            "manifest": {
                "path": manifest_path.name,
                "rows": len(records),
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
                "ordered_records_sha256": canonical_sha256(
                    [record["record_sha256"] for record in records]
                ),
            },
            "quarantine_identities_are_dataset_admissions": False,
            "source_text_persisted": False,
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
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_quarantine_manifest(
        args.population_root,
        args.judgments_root,
        args.aggregate,
        args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
