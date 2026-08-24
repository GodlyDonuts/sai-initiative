"""Seal exact quarantine identities from a complete Institutional Books audit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books_compiler_aggregate import (
    SCHEMA as AGGREGATE_SCHEMA,
    _load_json,
    _validate_population,
    _validate_receipt,
    triage_route,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-institutional-books-quarantine-manifest-receipt-v1"
RECORD_SCHEMA = "sai-institutional-books-quarantine-exclusion-v1"


class InstitutionalBooksQuarantineManifestError(RuntimeError):
    """The book aggregate, population, judgments, or output custody differs."""


def _load_aggregate(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise InstitutionalBooksQuarantineManifestError(
            "book aggregate is missing or unsafe"
        )
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise InstitutionalBooksQuarantineManifestError(
            "book aggregate is invalid"
        ) from error
    if not isinstance(value, dict):
        raise InstitutionalBooksQuarantineManifestError("book aggregate is invalid")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        value.get("schema") != AGGREGATE_SCHEMA
        or value.get("status")
        != "complete_nontraining_book_compiler_aggregate"
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
        or value.get("model_judgments_are_verified_admissions") is not False
        or value.get("training_ready") is not False
        or value.get("four_b_training_authorized") is not False
    ):
        raise InstitutionalBooksQuarantineManifestError(
            "book aggregate receipt differs"
        )
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
    """Emit text-free identities barred from later book materialization."""

    if output_root.exists() or output_root.is_symlink():
        raise InstitutionalBooksQuarantineManifestError(
            "book quarantine output already exists"
        )
    aggregate = _load_aggregate(aggregate_path)
    try:
        candidates, population = _validate_population(population_root)
    except RuntimeError as error:
        raise InstitutionalBooksQuarantineManifestError(
            "book population custody differs"
        ) from error
    expected_paths = {
        judgments_root
        / f"{candidate['candidate_identity_sha256']}.book-compiler.json"
        for candidate in candidates
    }
    if set(judgments_root.glob("*.book-compiler.json")) != expected_paths:
        raise InstitutionalBooksQuarantineManifestError(
            "book judgment population differs"
        )
    if aggregate.get("population", {}).get("rows") != len(candidates):
        raise InstitutionalBooksQuarantineManifestError(
            "book aggregate population coverage differs"
        )

    records = []
    for candidate in candidates:
        identity = candidate["candidate_identity_sha256"]
        path = judgments_root / f"{identity}.book-compiler.json"
        try:
            receipt = _validate_receipt(
                _load_json(path, "book compiler receipt"), candidate
            )
        except RuntimeError as error:
            raise InstitutionalBooksQuarantineManifestError(
                "book judgment custody differs"
            ) from error
        judgment = receipt["judgment"]
        if triage_route(judgment) != "quarantine":
            continue
        source = candidate.get("source")
        if not isinstance(source, dict) or not isinstance(
            source.get("barcode_src"), str
        ):
            raise InstitutionalBooksQuarantineManifestError(
                "book source identity differs"
            )
        risks = judgment.get("risks")
        if not isinstance(risks, dict):
            raise InstitutionalBooksQuarantineManifestError(
                "book judgment risks differ"
            )
        record = {
            "schema": RECORD_SCHEMA,
            "candidate_identity_sha256": identity,
            "source_content_sha256": candidate["source_content_sha256"],
            "source_book_id": source["barcode_src"],
            "source_provenance_sha256": candidate["provenance_sha256"],
            "judgment_receipt_sha256": receipt["receipt_sha256"],
            "judgment_sha256": judgment["judgment_sha256"],
            "verdict": judgment["verdict"],
            "active_risks": sorted(key for key, enabled in risks.items() if enabled),
            "route": "quarantine",
            "dataset_materialization_allowed": False,
            "source_text_persisted": False,
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)

    expected_quarantine = aggregate.get("counts", {}).get("triage_route", {}).get(
        "quarantine"
    )
    if (
        isinstance(expected_quarantine, bool)
        or not isinstance(expected_quarantine, int)
        or expected_quarantine < 0
        or expected_quarantine != len(records)
    ):
        raise InstitutionalBooksQuarantineManifestError(
            "book quarantine row coverage differs"
        )

    output_root.mkdir(parents=True)
    try:
        manifest_path = output_root / "quarantine_exclusions.jsonl"
        _atomic_jsonl(manifest_path, records)
        payload = {
            "schema": SCHEMA,
            "status": "complete_institutional_books_quarantine_exclusion_manifest",
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
