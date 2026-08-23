"""Create a benchmark-disjoint compiler population from a sealed audit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Container
from pathlib import Path
from typing import Any

from sai.data.decontamination import (
    _CODE,
    _WORD,
    POLICY,
    _code_overlap_count,
    _normalize,
    _overlap_count,
    binary_boundary_index,
)
from sai.data.reservoir_audit_aggregate import load_population
from sai.data.reservoir_audit_population import SCHEMA, _write_jsonl
from sai.data.token_stream import canonical_sha256, sha256_file

DECISION_SCHEMA = "sai-audit-population-decontamination-record-v1"


class AuditPopulationDecontaminationError(RuntimeError):
    """The population, benchmark boundary, or clean output differs."""


class _Union:
    def __init__(self, members: list[Container[bytes]]) -> None:
        self.members = members

    def __contains__(self, value: object) -> bool:
        return any(value in member for member in self.members)


def screen_candidate(
    candidate: dict[str, Any],
    word_boundary: Container[bytes],
    code_boundary: Container[bytes],
) -> dict[str, Any]:
    """Return one text-free contamination decision."""

    identity = candidate.get("candidate_identity_sha256")
    text = candidate.get("text")
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or not isinstance(text, str)
        or not text
    ):
        raise AuditPopulationDecontaminationError("audit candidate differs")
    normalized = _normalize(text)
    word_overlaps = _overlap_count(
        _WORD.findall(normalized), POLICY["word_shingle_tokens"], word_boundary
    )
    code_overlaps = _code_overlap_count(_CODE.findall(normalized), code_boundary)
    decision = {
        "schema": DECISION_SCHEMA,
        "candidate_identity_sha256": identity,
        "word_overlap_count": word_overlaps,
        "code_overlap_count": code_overlaps,
        "contaminated": bool(word_overlaps or code_overlaps),
        "source_text_persisted": False,
        "training_ready": False,
    }
    decision["record_sha256"] = canonical_sha256(decision)
    return decision


def promote_lineage(source: dict[str, Any], ordinal: int) -> dict[str, Any]:
    """Rebind a clean row to its filtered ordinal without changing candidate text."""

    original_ordinal = source.get("ordinal")
    original_digest = source.get("lineage_sha256")
    if (
        not isinstance(original_ordinal, int)
        or original_ordinal < 0
        or not isinstance(original_digest, str)
        or len(original_digest) != 64
        or source.get("raw_source_is_training_ready") is not False
    ):
        raise AuditPopulationDecontaminationError("audit lineage differs")
    promoted = {
        **{key: value for key, value in source.items() if key != "lineage_sha256"},
        "ordinal": ordinal,
        "pre_decontamination_ordinal": original_ordinal,
        "pre_decontamination_lineage_sha256": original_digest,
        "benchmark_decontamination_complete": True,
    }
    promoted["lineage_sha256"] = canonical_sha256(promoted)
    return promoted


def build_clean_population(
    population_root: Path, boundary_roots: list[Path], output_root: Path
) -> dict[str, Any]:
    """Screen every row and emit a complete clean population for Hermes."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or not boundary_roots
        or len(boundary_roots) != len(set(boundary_roots))
    ):
        raise AuditPopulationDecontaminationError(
            "audit decontamination output differs"
        )
    candidates, lineage, population_receipt = load_population(population_root)
    words, code, boundary_receipts = binary_boundary_index(boundary_roots)
    try:
        word_boundary = words[0] if len(words) == 1 else _Union(words)
        code_boundary = code[0] if len(code) == 1 else _Union(code)
        decisions = [
            screen_candidate(candidate, word_boundary, code_boundary)
            for candidate in candidates
        ]
    finally:
        for member in [*words, *code]:
            member.close()
    clean_candidates = []
    clean_lineage = []
    for candidate, source, decision in zip(
        candidates, lineage, decisions, strict=True
    ):
        if candidate["candidate_identity_sha256"] != decision[
            "candidate_identity_sha256"
        ]:
            raise AuditPopulationDecontaminationError(
                "audit decontamination coverage differs"
            )
        if not decision["contaminated"]:
            clean_candidates.append(candidate)
            clean_lineage.append(promote_lineage(source, len(clean_lineage)))
    if not clean_candidates:
        raise AuditPopulationDecontaminationError("all audit rows are contaminated")

    temporary = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    if temporary.exists() or temporary.is_symlink():
        raise AuditPopulationDecontaminationError(
            "audit decontamination temporary output exists"
        )
    temporary.mkdir(parents=True)
    try:
        candidates_path = temporary / "candidates.jsonl"
        lineage_path = temporary / "lineage.jsonl"
        decisions_path = temporary / "decisions.jsonl"
        receipt_path = temporary / "receipt.json"
        _write_jsonl(candidates_path, clean_candidates)
        _write_jsonl(lineage_path, clean_lineage)
        _write_jsonl(decisions_path, decisions)
        identities = [
            candidate["candidate_identity_sha256"] for candidate in clean_candidates
        ]
        totals = Counter()
        for decision in decisions:
            totals["contaminated_rows"] += decision["contaminated"]
            totals["word_overlap_shingles"] += decision["word_overlap_count"]
            totals["code_overlap_shingles"] += decision["code_overlap_count"]
        by_source = Counter(row["source_id"] for row in clean_lineage)
        by_stratum = Counter(row["stratum"] for row in clean_lineage)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "selection_method": "reject_any_official_benchmark_shingle_overlap",
            "statistically_representative": False,
            "screen_only": True,
            "source_population": {
                "root_name": population_root.name,
                "receipt_file_sha256": sha256_file(population_root / "receipt.json"),
                "receipt_sha256": population_receipt["receipt_sha256"],
                "population_file_sha256": sha256_file(
                    population_root / "candidates.jsonl"
                ),
                "lineage_file_sha256": sha256_file(
                    population_root / "lineage.jsonl"
                ),
            },
            "boundary_indexes": boundary_receipts,
            "boundary_indexes_sha256": canonical_sha256(boundary_receipts),
            "policy": POLICY,
            "policy_sha256": canonical_sha256(POLICY),
            "input_rows": len(candidates),
            "clean_rows": len(clean_candidates),
            "contaminated_rows": totals["contaminated_rows"],
            "word_overlap_shingles": totals["word_overlap_shingles"],
            "code_overlap_shingles": totals["code_overlap_shingles"],
            "population": {
                "path": candidates_path.name,
                "rows": len(clean_candidates),
                "bytes": candidates_path.stat().st_size,
                "sha256": sha256_file(candidates_path),
                "ordered_identities_sha256": canonical_sha256(identities),
            },
            "lineage": {
                "path": lineage_path.name,
                "rows": len(clean_lineage),
                "bytes": lineage_path.stat().st_size,
                "sha256": sha256_file(lineage_path),
                "ordered_rows_sha256": canonical_sha256(clean_lineage),
            },
            "decisions": {
                "path": decisions_path.name,
                "rows": len(decisions),
                "bytes": decisions_path.stat().st_size,
                "sha256": sha256_file(decisions_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in decisions]
                ),
                "source_text_persisted": False,
            },
            "by_source": dict(sorted(by_source.items())),
            "by_stratum": dict(sorted(by_stratum.items())),
            "benchmark_contamination_screen_complete": True,
            "full_source_population_decontaminated": False,
            "hermes_judgments_complete": False,
            "quality_compilation_complete": False,
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
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--boundary-index", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_clean_population(
        args.population_root, args.boundary_index, args.output_root
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
