"""Build the private candidate subset for independent book verification."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books_compiler_aggregate import (
    INDEPENDENT_POPULATION_SCHEMA as SCHEMA,
)
from sai.data.institutional_books_compiler_aggregate import _validate_population
from sai.data.institutional_books_semantic_decision import (
    RECORD_SCHEMA as DECISION_RECORD_SCHEMA,
)
from sai.data.institutional_books_semantic_decision import (
    SCHEMA as DECISION_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file


class InstitutionalBooksIndependentPopulationError(RuntimeError):
    """Semantic decisions or independent population custody differs."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise InstitutionalBooksIndependentPopulationError("input is unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise InstitutionalBooksIndependentPopulationError(
            "input is invalid"
        ) from error
    if not isinstance(value, dict):
        raise InstitutionalBooksIndependentPopulationError("input differs")
    return value


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise InstitutionalBooksIndependentPopulationError("output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial.{uuid.uuid4().hex}"
    try:
        descriptor = os.open(
            temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
        )
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _decisions(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt = _load_json(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("manifest")
    if (
        receipt.get("schema") != DECISION_SCHEMA
        or receipt.get("status")
        != "complete_nontraining_conservative_book_semantic_decision"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("training_ready") is not False
        or receipt.get("independent_verification_complete") is not False
        or not isinstance(descriptor, dict)
    ):
        raise InstitutionalBooksIndependentPopulationError(
            "semantic decision receipt differs"
        )
    path = root / descriptor.get("path", "")
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise InstitutionalBooksIndependentPopulationError(
            "semantic decision manifest differs"
        )
    rows = []
    seen = set()
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            unsigned_row = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            identity = row.get("candidate_identity_sha256")
            if (
                row.get("schema") != DECISION_RECORD_SCHEMA
                or not isinstance(identity, str)
                or identity in seen
                or row.get("record_sha256") != canonical_sha256(unsigned_row)
                or row.get("training_ready") is not False
            ):
                raise InstitutionalBooksIndependentPopulationError(
                    "semantic decision row differs"
                )
            seen.add(identity)
            rows.append(row)
    if len(rows) != descriptor.get("rows") or canonical_sha256(
        [row["record_sha256"] for row in rows]
    ) != descriptor.get("ordered_records_sha256"):
        raise InstitutionalBooksIndependentPopulationError(
            "semantic decision coverage differs"
        )
    return rows, receipt


def build_population(
    source_population_root: Path,
    decision_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Copy only strict survivors into a private independent-review queue."""

    if output_root.exists() or output_root.is_symlink():
        raise InstitutionalBooksIndependentPopulationError("output root exists")
    try:
        candidates, source_population = _validate_population(source_population_root)
    except RuntimeError as error:
        raise InstitutionalBooksIndependentPopulationError(
            "source population differs"
        ) from error
    decisions, decision_receipt = _decisions(decision_root)
    by_identity = {row["candidate_identity_sha256"]: row for row in decisions}
    candidates_by_identity = {
        row["candidate_identity_sha256"]: row for row in candidates
    }
    if (
        set(by_identity) != set(candidates_by_identity)
        or decision_receipt.get("population", {}).get("receipt_sha256")
        != source_population["receipt_sha256"]
    ):
        raise InstitutionalBooksIndependentPopulationError(
            "decision-to-population binding differs"
        )
    selected_identities = sorted(
        identity
        for identity, decision in by_identity.items()
        if decision.get("disposition") == "independent_verification"
    )
    selected = [candidates_by_identity[identity] for identity in selected_identities]
    selected_tokens = sum(
        by_identity[identity]["token_count_o200k_base_gen"]
        for identity in selected_identities
    )
    output_root.mkdir(parents=True)
    try:
        candidate_path = output_root / "candidates.jsonl"
        _atomic_jsonl(candidate_path, selected)
        payload = {
            "schema": SCHEMA,
            "status": (
                "complete_nontraining_private_independent_book_candidate_population"
            ),
            "source": {
                "population_receipt_sha256": source_population["receipt_sha256"],
                "decision_receipt_sha256": decision_receipt["receipt_sha256"],
                "decision_manifest_sha256": decision_receipt["manifest"]["sha256"],
                "decision_rows": len(decisions),
            },
            "selection": {
                "required_disposition": "independent_verification",
                "rows": len(selected),
                "tokens": selected_tokens,
                "ordered_candidate_identities_sha256": canonical_sha256(
                    selected_identities
                ),
            },
            "output": {
                "path": candidate_path.name,
                "rows": len(selected),
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
                "ordered_candidate_identities_sha256": canonical_sha256(
                    selected_identities
                ),
            },
            "source_text_private": True,
            "source_text_publishable": False,
            "independent_verification_complete": False,
            "benchmark_decontamination_complete": False,
            "global_semantic_deduplication_complete": False,
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
    parser.add_argument("--source-population-root", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_population(
        args.source_population_root,
        args.decision_root,
        args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
