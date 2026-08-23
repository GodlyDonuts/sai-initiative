"""Find exact and high-confidence near duplicates in a reservoir audit population."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.reservoir_audit_aggregate import load_population
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-reservoir-audit-duplicate-report-v1"
SHINGLE_WIDTH = 5
MIN_JACCARD_SHINGLES = 40
MIN_CONTAINMENT_SHINGLES = 80
JACCARD_THRESHOLD_PPM = 800_000
CONTAINMENT_THRESHOLD_PPM = 900_000
TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class ReservoirAuditDuplicateError(RuntimeError):
    """Population identity or duplicate evidence differs."""


def _normalized_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold())


def _fingerprint(candidate: dict[str, Any]) -> dict[str, Any]:
    tokens = _normalized_tokens(candidate["text"])
    normalized = " ".join(tokens)
    shingles = {
        hashlib.sha256(
            " ".join(tokens[index : index + SHINGLE_WIDTH]).encode()
        ).digest()
        for index in range(max(0, len(tokens) - SHINGLE_WIDTH + 1))
    }
    return {
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "source_content_sha256": candidate["source_content_sha256"],
        "normalized_text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "tokens": len(tokens),
        "shingles": shingles,
    }


def find_duplicate_pairs(
    candidates: list[dict[str, Any]], lineage: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return every pair crossing one conservative duplicate threshold."""

    if not candidates or len(candidates) != len(lineage):
        raise ReservoirAuditDuplicateError("duplicate audit inputs differ")
    fingerprints = [_fingerprint(candidate) for candidate in candidates]
    pairs = []
    for left_index, left in enumerate(fingerprints):
        for right_index in range(left_index + 1, len(fingerprints)):
            right = fingerprints[right_index]
            intersection = len(left["shingles"] & right["shingles"])
            union = len(left["shingles"] | right["shingles"])
            minimum = min(len(left["shingles"]), len(right["shingles"]))
            jaccard_ppm = 0 if union == 0 else (intersection * 1_000_000) // union
            containment_ppm = (
                0 if minimum == 0 else (intersection * 1_000_000) // minimum
            )
            byte_exact = left["source_content_sha256"] == right["source_content_sha256"]
            normalized_exact = (
                left["tokens"] > 0
                and right["tokens"] > 0
                and left["normalized_text_sha256"] == right["normalized_text_sha256"]
            )
            reasons = []
            if byte_exact:
                reasons.append("byte_exact")
            elif normalized_exact:
                reasons.append("normalized_token_exact")
            if minimum >= MIN_JACCARD_SHINGLES and jaccard_ppm >= JACCARD_THRESHOLD_PPM:
                reasons.append("five_word_shingle_jaccard")
            if (
                minimum >= MIN_CONTAINMENT_SHINGLES
                and containment_ppm >= CONTAINMENT_THRESHOLD_PPM
            ):
                reasons.append("five_word_shingle_containment")
            if not reasons:
                continue
            source_left, source_right = lineage[left_index], lineage[right_index]
            pair = {
                "left_candidate_identity_sha256": left["candidate_identity_sha256"],
                "right_candidate_identity_sha256": right["candidate_identity_sha256"],
                "left_source_id": source_left["source_id"],
                "right_source_id": source_right["source_id"],
                "cross_source": source_left["source_id"] != source_right["source_id"],
                "intersection_shingles": intersection,
                "minimum_shingles": minimum,
                "union_shingles": union,
                "jaccard_ppm": jaccard_ppm,
                "containment_ppm": containment_ppm,
                "reasons": reasons,
            }
            pair["pair_sha256"] = canonical_sha256(pair)
            pairs.append(pair)
    return pairs


def _groups(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in pairs:
        union(
            pair["left_candidate_identity_sha256"],
            pair["right_candidate_identity_sha256"],
        )
    grouped: dict[str, set[str]] = {}
    for identity in parent:
        grouped.setdefault(find(identity), set()).add(identity)
    groups = []
    for members in sorted(grouped.values(), key=lambda values: sorted(values)):
        row = {"members": sorted(members), "member_count": len(members)}
        row["group_sha256"] = canonical_sha256(row)
        groups.append(row)
    return groups


def build_report(population_root: Path, output_path: Path) -> dict[str, Any]:
    """Replay the population and seal candidate-level duplicate evidence."""

    if output_path.exists() or output_path.is_symlink():
        raise ReservoirAuditDuplicateError("duplicate report output already exists")
    candidates, lineage, population_receipt = load_population(population_root)
    pairs = find_duplicate_pairs(candidates, lineage)
    groups = _groups(pairs)
    reason_counts = Counter(reason for pair in pairs for reason in pair["reasons"])
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "population_receipt_sha256": population_receipt["receipt_sha256"],
        "population_file_sha256": sha256_file(population_root / "candidates.jsonl"),
        "policy": {
            "normalization": "unicode_nfkc_casefold_word_tokens",
            "shingle_width": SHINGLE_WIDTH,
            "minimum_jaccard_shingles": MIN_JACCARD_SHINGLES,
            "minimum_containment_shingles": MIN_CONTAINMENT_SHINGLES,
            "jaccard_threshold_ppm": JACCARD_THRESHOLD_PPM,
            "containment_threshold_ppm": CONTAINMENT_THRESHOLD_PPM,
        },
        "candidate_rows": len(candidates),
        "candidate_pairs_compared": len(candidates) * (len(candidates) - 1) // 2,
        "flagged_pairs": len(pairs),
        "cross_source_flagged_pairs": sum(pair["cross_source"] for pair in pairs),
        "reason_counts": dict(sorted(reason_counts.items())),
        "pairs": pairs,
        "groups": groups,
        "audit_sample_deduplication_complete": True,
        "full_reservoir_deduplication_complete": False,
        "training_ready": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_report(args.population_root, args.output)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
