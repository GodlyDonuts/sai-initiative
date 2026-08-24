"""Screen one sealed audit population against non-reversible benchmark indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Container
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create, normalize_candidate
from sai.data.decontamination import (
    _CODE,
    _WORD,
    POLICY,
    _code_overlap_count,
    _normalize,
    _overlap_count,
    binary_boundary_index,
)
from sai.data.pleias_parent_disjoint_audit_aggregate import (
    AGGREGATE_SCHEMA as PLEIAS_AGGREGATE_SCHEMA,
)
from sai.data.pleias_parent_disjoint_audit_aggregate import (
    load_aggregate_population as load_pleias_aggregate_population,
)
from sai.data.reservoir_audit_aggregate import load_population
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-audit-population-benchmark-contamination-screen-v2"


class BenchmarkContaminationScreenError(RuntimeError):
    """The population, boundary, or source-safe contamination result differs."""


def summarize(
    candidates: list[dict[str, Any]],
    lineage: list[dict[str, Any]],
    word_boundary: Container[bytes],
    code_boundary: Container[bytes],
) -> dict[str, Any]:
    """Return aggregate overlap evidence without retaining any source text."""

    if not candidates or len(candidates) != len(lineage):
        raise BenchmarkContaminationScreenError("screen population differs")
    totals = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_stratum: dict[str, Counter[str]] = defaultdict(Counter)
    ordered_decision_digest = hashlib.sha256()
    identities = set()
    for raw_candidate, source in zip(candidates, lineage, strict=True):
        candidate = normalize_candidate(raw_candidate)
        identity = candidate["candidate_identity_sha256"]
        source_id = source.get("source_id") if isinstance(source, dict) else None
        stratum = source.get("stratum") if isinstance(source, dict) else None
        if (
            identity in identities
            or not isinstance(source_id, str)
            or not source_id
            or not isinstance(stratum, str)
            or not stratum
        ):
            raise BenchmarkContaminationScreenError("screen lineage differs")
        identities.add(identity)
        normalized = _normalize(candidate["text"])
        word_overlaps = _overlap_count(
            _WORD.findall(normalized), POLICY["word_shingle_tokens"], word_boundary
        )
        code_overlaps = _code_overlap_count(_CODE.findall(normalized), code_boundary)
        contaminated = bool(word_overlaps or code_overlaps)
        decision = {
            "candidate_identity_sha256": identity,
            "word_overlap_count": word_overlaps,
            "code_overlap_count": code_overlaps,
            "contaminated": contaminated,
        }
        ordered_decision_digest.update(bytes.fromhex(canonical_sha256(decision)))
        for counter in (totals, by_source[source_id], by_stratum[stratum]):
            counter["rows"] += 1
            counter["contaminated_rows"] += contaminated
            counter["word_overlap_shingles"] += word_overlaps
            counter["code_overlap_shingles"] += code_overlaps
    return {
        "rows": totals["rows"],
        "contaminated_rows": totals["contaminated_rows"],
        "clean_rows": totals["rows"] - totals["contaminated_rows"],
        "word_overlap_shingles": totals["word_overlap_shingles"],
        "code_overlap_shingles": totals["code_overlap_shingles"],
        "by_source": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(by_source.items())
        },
        "by_stratum": {
            key: dict(sorted(counter.items()))
            for key, counter in sorted(by_stratum.items())
        },
        "ordered_decisions_sha256": ordered_decision_digest.hexdigest(),
        "individual_decisions_persisted": False,
        "source_text_persisted": False,
    }


def build_screen(
    population_root: Path, boundary_roots: list[Path], output_path: Path
) -> dict[str, Any]:
    """Replay the exact population and indexes, then create one safe aggregate."""

    if output_path.exists() or output_path.is_symlink():
        raise BenchmarkContaminationScreenError("screen output already exists")
    if not boundary_roots:
        raise BenchmarkContaminationScreenError("screen boundary is missing")
    try:
        receipt_header = json.loads((population_root / "receipt.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkContaminationScreenError(
            "screen population receipt differs"
        ) from error
    if receipt_header.get("schema") == PLEIAS_AGGREGATE_SCHEMA:
        candidates, lineage, population_receipt = load_pleias_aggregate_population(
            population_root
        )
    else:
        candidates, lineage, population_receipt = load_population(population_root)
    words, code, boundary_receipts = binary_boundary_index(boundary_roots)
    try:
        summary = summarize(
            candidates,
            lineage,
            words[0] if len(words) == 1 else _Union(words),
            code[0] if len(code) == 1 else _Union(code),
        )
    finally:
        for member in [*words, *code]:
            member.close()
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "population": {
            "root_name": population_root.name,
            "receipt_sha256": population_receipt["receipt_sha256"],
            "population_file_sha256": sha256_file(population_root / "candidates.jsonl"),
            "lineage_file_sha256": sha256_file(population_root / "lineage.jsonl"),
        },
        "boundary_indexes": boundary_receipts,
        "boundary_indexes_sha256": canonical_sha256(boundary_receipts),
        "policy": POLICY,
        "policy_sha256": canonical_sha256(POLICY),
        "summary": summary,
        "benchmark_contamination_screen_complete": True,
        "full_source_population_decontaminated": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


class _Union:
    """Small exact-membership union for multiple mapped boundary files."""

    def __init__(self, members: list[Container[bytes]]) -> None:
        self.members = members

    def __contains__(self, value: object) -> bool:
        return any(value in member for member in self.members)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--boundary-index", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_screen(args.population_root, args.boundary_index, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
