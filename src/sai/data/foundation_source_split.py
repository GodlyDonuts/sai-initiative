"""Assign related foundation documents to one deterministic train/dev split."""

from __future__ import annotations

from typing import Any

from sai.data.token_stream import canonical_sha256

POLICY = {
    "schema": "sai-foundation-source-disjoint-split-policy-v1",
    "hash": "sha256_canonical_json",
    "bucket_modulus": 1_000,
    "development_buckets": 50,
    "development_fraction_ppm": 50_000,
    "grouping": {
        "pleias_common_corpus": "pinned_source_parent",
        "institutional_books": "global_connected_two_family_work_candidate_graph",
    },
}
POLICY_SHA256 = canonical_sha256(POLICY)


class FoundationSourceSplitError(RuntimeError):
    """Component, group identifiers, or deterministic split differs."""


def assign_source_group(
    component: str, stable_identifiers: dict[str, Any]
) -> tuple[str, str, int]:
    """Return group SHA-256, split, and stable bucket for exact identifiers."""

    if (
        component not in POLICY["grouping"]
        or not isinstance(stable_identifiers, dict)
        or not stable_identifiers
        or any(
            not isinstance(key, str)
            or not key
            or (
                not isinstance(value, str)
                and not (
                    isinstance(value, list)
                    and value
                    and len(value) == len(set(value))
                    and all(isinstance(item, str) and item for item in value)
                )
            )
            or (isinstance(value, str) and not value)
            for key, value in stable_identifiers.items()
        )
    ):
        raise FoundationSourceSplitError("source group differs")
    group = canonical_sha256(
        {
            "component": component,
            "grouping_policy": POLICY["grouping"][component],
            "stable_identifiers": stable_identifiers,
        }
    )
    bucket = int(group[:16], 16) % POLICY["bucket_modulus"]
    split = "development" if bucket < POLICY["development_buckets"] else "train"
    return group, split, bucket
