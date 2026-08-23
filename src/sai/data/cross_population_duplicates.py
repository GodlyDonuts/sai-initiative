"""Find exact duplicates across independently sealed audit populations."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.reservoir_audit_aggregate import load_population
from sai.data.reservoir_audit_duplicates import _normalized_tokens
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-cross-population-exact-duplicate-report-v1"


class CrossPopulationDuplicateError(RuntimeError):
    """Population custody or cross-population duplicate identity differs."""


def find_exact_pairs(
    populations: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Find byte and normalized-token equality without quadratic comparisons."""

    records = []
    identities = set()
    for population, candidates, lineage in populations:
        if (
            not population
            or not candidates
            or len(candidates) != len(lineage)
            or any(
                candidate["candidate_identity_sha256"] in identities
                for candidate in candidates
            )
        ):
            raise CrossPopulationDuplicateError(
                "cross-population candidate custody differs"
            )
        for candidate, source in zip(candidates, lineage, strict=True):
            identity = candidate["candidate_identity_sha256"]
            identities.add(identity)
            tokens = _normalized_tokens(candidate["text"])
            records.append(
                {
                    "identity": identity,
                    "population": population,
                    "source_id": source["source_id"],
                    "byte_hash": candidate["source_content_sha256"],
                    "normalized_hash": (
                        hashlib.sha256(" ".join(tokens).encode()).hexdigest()
                        if tokens
                        else None
                    ),
                }
            )

    by_byte: dict[str, list[int]] = defaultdict(list)
    by_normalized: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_byte[record["byte_hash"]].append(index)
        if record["normalized_hash"] is not None:
            by_normalized[record["normalized_hash"]].append(index)
    reasons: dict[tuple[int, int], set[str]] = defaultdict(set)
    for label, groups in (
        ("byte_exact", by_byte),
        ("normalized_token_exact", by_normalized),
    ):
        for members in groups.values():
            if len(members) < 2:
                continue
            for left_offset, left in enumerate(members):
                for right in members[left_offset + 1 :]:
                    reasons[(left, right)].add(label)
    pairs = []
    for (left_index, right_index), labels in sorted(reasons.items()):
        left, right = records[left_index], records[right_index]
        pair = {
            "left_candidate_identity_sha256": left["identity"],
            "right_candidate_identity_sha256": right["identity"],
            "left_population": left["population"],
            "right_population": right["population"],
            "left_source_id": left["source_id"],
            "right_source_id": right["source_id"],
            "cross_population": left["population"] != right["population"],
            "cross_source": left["source_id"] != right["source_id"],
            "reasons": sorted(labels),
        }
        pair["pair_sha256"] = canonical_sha256(pair)
        pairs.append(pair)
    return pairs


def build_report(population_roots: list[Path], output_path: Path) -> dict[str, Any]:
    """Replay multiple populations and seal source-safe equality evidence."""

    if (
        len(population_roots) < 2
        or len(population_roots) != len(set(population_roots))
        or output_path.exists()
        or output_path.is_symlink()
    ):
        raise CrossPopulationDuplicateError(
            "cross-population input or output boundary differs"
        )
    loaded = []
    descriptors = []
    for root in population_roots:
        candidates, lineage, receipt = load_population(root)
        name = root.name
        loaded.append((name, candidates, lineage))
        descriptors.append(
            {
                "population": name,
                "receipt_sha256": receipt["receipt_sha256"],
                "candidate_rows": len(candidates),
                "population_file_sha256": sha256_file(root / "candidates.jsonl"),
                "lineage_file_sha256": sha256_file(root / "lineage.jsonl"),
            }
        )
    pairs = find_exact_pairs(loaded)
    reason_counts = Counter(reason for pair in pairs for reason in pair["reasons"])
    total = sum(len(candidates) for _, candidates, _ in loaded)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "populations": descriptors,
        "candidate_rows": total,
        "candidate_pairs_logically_covered": total * (total - 1) // 2,
        "algorithm": (
            "hash_indexed_utf8_byte_equality_and_unicode_nfkc_casefold_"
            "word_token_equality"
        ),
        "flagged_pairs": len(pairs),
        "cross_population_flagged_pairs": sum(
            pair["cross_population"] for pair in pairs
        ),
        "cross_source_flagged_pairs": sum(pair["cross_source"] for pair in pairs),
        "reason_counts": dict(sorted(reason_counts.items())),
        "pairs": pairs,
        "sample_exact_duplicate_audit_complete": True,
        "sample_near_duplicate_audit_complete": False,
        "full_reservoir_deduplication_complete": False,
        "training_ready": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_report(args.population_root, args.output)
    import json

    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
