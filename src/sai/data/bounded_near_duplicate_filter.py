"""Exact high-confidence near-duplicate filtering for bounded source pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.reservoir_audit_duplicates import (
    CONTAINMENT_THRESHOLD_PPM,
    JACCARD_THRESHOLD_PPM,
    MIN_CONTAINMENT_SHINGLES,
    MIN_JACCARD_SHINGLES,
    SHINGLE_WIDTH,
    _normalized_tokens,
)
from sai.data.token_stream import (
    canonical_sha256,
    normalize_document,
    sha256_file,
)

SCHEMA = "sai-bounded-exact-near-duplicate-filter-v1"
MAXIMUM_DOCUMENTS = 10_000
MAXIMUM_TOTAL_SHINGLE_OCCURRENCES = 5_000_000
MAXIMUM_UNIQUE_SHINGLES = 5_000_000
DEFAULT_BLOCK_ROWS = 128


class BoundedNearDuplicateError(RuntimeError):
    """The pilot input, exact sparse join, or create-only output differs."""


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _load_documents(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise BoundedNearDuplicateError("near-duplicate input is missing or unsafe")
    documents = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                documents.append(normalize_document(json.loads(line)))
            except Exception as error:
                raise BoundedNearDuplicateError(
                    f"near-duplicate input row {line_number} differs"
                ) from error
            if len(documents) > MAXIMUM_DOCUMENTS:
                raise BoundedNearDuplicateError(
                    "near-duplicate pilot exceeds its document bound"
                )
    identities = [row["identity_sha256"] for row in documents]
    if not documents or len(identities) != len(set(identities)):
        raise BoundedNearDuplicateError(
            "near-duplicate input is empty or identity-duplicated"
        )
    return documents


def _shingles(tokens: list[str]) -> set[bytes]:
    return {
        hashlib.sha256(
            " ".join(tokens[index : index + SHINGLE_WIDTH]).encode()
        ).digest()
        for index in range(max(0, len(tokens) - SHINGLE_WIDTH + 1))
    }


def _matrix(
    documents: list[dict[str, Any]],
) -> tuple[Any, list[int], dict[str, list[list[int]]], dict[str, int]]:
    try:
        import numpy
        from scipy.sparse import csr_matrix
    except ImportError as error:
        raise BoundedNearDuplicateError("numpy and scipy are required") from error
    columns: dict[bytes, int] = {}
    indices = []
    indptr = [0]
    sizes = []
    exact: dict[str, dict[str, list[int]]] = {
        "byte": defaultdict(list),
        "normalized": defaultdict(list),
    }
    total = 0
    for document_index, document in enumerate(documents):
        text = document["text"]
        tokens = _normalized_tokens(text)
        normalized = " ".join(tokens)
        exact["byte"][hashlib.sha256(text.encode()).hexdigest()].append(
            document_index
        )
        if normalized:
            exact["normalized"][hashlib.sha256(normalized.encode()).hexdigest()].append(
                document_index
            )
        shingles = _shingles(tokens)
        total += len(shingles)
        if total > MAXIMUM_TOTAL_SHINGLE_OCCURRENCES:
            raise BoundedNearDuplicateError(
                "near-duplicate pilot exceeds its total shingle bound"
            )
        row_columns = []
        for digest in shingles:
            column = columns.setdefault(digest, len(columns))
            row_columns.append(column)
        if len(columns) > MAXIMUM_UNIQUE_SHINGLES:
            raise BoundedNearDuplicateError(
                "near-duplicate pilot exceeds its unique shingle bound"
            )
        indices.extend(sorted(row_columns))
        indptr.append(len(indices))
        sizes.append(len(shingles))
    data = numpy.ones(len(indices), dtype=numpy.int32)
    matrix = csr_matrix(
        (
            data,
            numpy.asarray(indices, dtype=numpy.int32),
            numpy.asarray(indptr, dtype=numpy.int64),
        ),
        shape=(len(documents), len(columns)),
        dtype=numpy.int32,
    )
    return (
        matrix,
        sizes,
        {
            label: [members for members in groups.values() if len(members) > 1]
            for label, groups in exact.items()
        },
        {
            "total_shingle_occurrences": total,
            "unique_shingles": len(columns),
        },
    )


def find_groups(
    documents: list[dict[str, Any]], *, block_rows: int = DEFAULT_BLOCK_ROWS
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Run an exhaustive sparse threshold join and return duplicate components."""

    if (
        isinstance(block_rows, bool)
        or not isinstance(block_rows, int)
        or not 1 <= block_rows <= 1_024
    ):
        raise BoundedNearDuplicateError("near-duplicate block geometry differs")
    matrix, sizes, exact_groups, geometry = _matrix(documents)
    union_find = _UnionFind(len(documents))
    evidence: Counter[str] = Counter()
    for label, groups in exact_groups.items():
        for members in groups:
            first = members[0]
            for member in members[1:]:
                union_find.union(first, member)
            evidence[f"{label}_exact_groups"] += 1
            evidence[f"{label}_exact_documents"] += len(members)
    nonzero_pair_intersections = 0
    for start in range(0, len(documents), block_rows):
        stop = min(len(documents), start + block_rows)
        intersections = (matrix[start:stop] @ matrix.T).tocoo()
        for local_left, right, intersection in zip(
            intersections.row,
            intersections.col,
            intersections.data,
            strict=True,
        ):
            left = start + int(local_left)
            right = int(right)
            if right <= left:
                continue
            intersection = int(intersection)
            nonzero_pair_intersections += 1
            minimum = min(sizes[left], sizes[right])
            union = sizes[left] + sizes[right] - intersection
            jaccard_ppm = 0 if union == 0 else (intersection * 1_000_000) // union
            containment_ppm = (
                0 if minimum == 0 else (intersection * 1_000_000) // minimum
            )
            jaccard_match = (
                minimum >= MIN_JACCARD_SHINGLES
                and jaccard_ppm >= JACCARD_THRESHOLD_PPM
            )
            containment_match = (
                minimum >= MIN_CONTAINMENT_SHINGLES
                and containment_ppm >= CONTAINMENT_THRESHOLD_PPM
            )
            if not (jaccard_match or containment_match):
                continue
            union_find.union(left, right)
            evidence["threshold_pair_matches"] += 1
            evidence["jaccard_pair_matches"] += jaccard_match
            evidence["containment_pair_matches"] += containment_match
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(documents)):
        components[union_find.find(index)].append(index)
    groups = []
    for members in components.values():
        if len(members) < 2:
            continue
        identities = sorted(documents[index]["identity_sha256"] for index in members)
        group = {
            "canonical_survivor_identity_sha256": identities[0],
            "member_identity_sha256s": identities,
            "member_count": len(identities),
        }
        group["group_sha256"] = canonical_sha256(group)
        groups.append(group)
    groups.sort(key=lambda row: row["canonical_survivor_identity_sha256"])
    evidence["nonzero_pair_intersections"] = nonzero_pair_intersections
    evidence["candidate_pairs_logically_covered"] = (
        len(documents) * (len(documents) - 1) // 2
    )
    evidence["duplicate_groups"] = len(groups)
    evidence["duplicate_documents"] = sum(row["member_count"] for row in groups)
    evidence["documents_dropped"] = sum(row["member_count"] - 1 for row in groups)
    return groups, dict(sorted(evidence.items())), geometry


def build_filter(
    input_path: Path,
    output_path: Path,
    receipt_path: Path,
    *,
    block_rows: int = DEFAULT_BLOCK_ROWS,
) -> dict[str, Any]:
    """Create one exact bounded-pilot filter and source-safe receipt."""

    if output_path.exists() or receipt_path.exists():
        raise BoundedNearDuplicateError("near-duplicate output already exists")
    input_sha256 = sha256_file(input_path)
    documents = _load_documents(input_path)
    groups, evidence, geometry = find_groups(documents, block_rows=block_rows)
    dropped = {
        identity
        for group in groups
        for identity in group["member_identity_sha256s"]
        if identity != group["canonical_survivor_identity_sha256"]
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    accepted_identity_digest = hashlib.sha256()
    accepted = 0
    try:
        with temporary.open("w") as handle:
            for document in documents:
                if document["identity_sha256"] in dropped:
                    continue
                handle.write(
                    json.dumps(
                        document,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                accepted_identity_digest.update(
                    bytes.fromhex(document["identity_sha256"])
                )
                accepted += 1
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    payload = {
        "schema": SCHEMA,
        "status": "complete_bounded_pilot_filter",
        "input": {
            "path": str(input_path.resolve()),
            "bytes": input_path.stat().st_size,
            "sha256": input_sha256,
            "documents": len(documents),
        },
        "policy": {
            "normalization": "unicode_nfkc_casefold_word_tokens",
            "shingle_width": SHINGLE_WIDTH,
            "minimum_jaccard_shingles": MIN_JACCARD_SHINGLES,
            "minimum_containment_shingles": MIN_CONTAINMENT_SHINGLES,
            "jaccard_threshold_ppm": JACCARD_THRESHOLD_PPM,
            "containment_threshold_ppm": CONTAINMENT_THRESHOLD_PPM,
            "algorithm": (
                "exact_sha256_shingle_csr_blockwise_integer_intersection_join"
            ),
            "maximum_documents": MAXIMUM_DOCUMENTS,
            "maximum_total_shingle_occurrences": (
                MAXIMUM_TOTAL_SHINGLE_OCCURRENCES
            ),
            "maximum_unique_shingles": MAXIMUM_UNIQUE_SHINGLES,
            "block_rows": block_rows,
        },
        "geometry": geometry,
        "evidence": evidence,
        "groups": groups,
        "groups_sha256": canonical_sha256(groups),
        "output": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "documents": accepted,
            "ordered_identity_sha256": accepted_identity_digest.hexdigest(),
        },
        "bounded_pilot_exact_and_high_confidence_near_duplicate_filter_complete": True,
        "global_cross_source_near_duplicate_filter_complete": False,
        "source_text_persisted_in_receipt": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    if sha256_file(input_path) != input_sha256:
        output_path.unlink(missing_ok=True)
        raise BoundedNearDuplicateError("near-duplicate input changed during filtering")
    payload["receipt_sha256"] = canonical_sha256(payload)
    try:
        _atomic_create(receipt_path, payload)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    args = parser.parse_args()
    result = build_filter(
        args.input,
        args.output,
        args.receipt,
        block_rows=args.block_rows,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
