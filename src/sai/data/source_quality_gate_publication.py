"""Publish source-safe, replayed evidence from mechanical quality gates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.source_quality_gate import (
    DECISION_SCHEMA,
    POLICY,
    POLICY_SHA256,
    validate,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-source-mechanical-quality-gate-publication-v1"


class SourceQualityGatePublicationError(RuntimeError):
    """A gate receipt, decision population, or publication differs."""


def _decision_identities(
    receipt: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    path = Path(receipt["decisions"]["path"])
    identities = []
    content_identities = []
    hard_rejects = []
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                row = json.loads(line)
                identity = row.get("candidate_identity_sha256")
                content_identity = row.get("source_content_sha256")
                if (
                    row.get("schema") != DECISION_SCHEMA
                    or not isinstance(identity, str)
                    or len(identity) != 64
                    or not isinstance(content_identity, str)
                    or len(content_identity) != 64
                ):
                    raise SourceQualityGatePublicationError(
                        f"quality-gate decision row {line_number} differs"
                    )
                identities.append(identity)
                content_identities.append(content_identity)
                if row.get("decision") == "hard_reject":
                    hard_rejects.append(identity)
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise SourceQualityGatePublicationError(
            "quality-gate decision population cannot be read"
        ) from error
    if len(identities) != receipt["decisions"]["rows"]:
        raise SourceQualityGatePublicationError(
            "quality-gate decision population coverage differs"
        )
    return identities, content_identities, hard_rejects


def build_publication(receipt_paths: list[Path], output: Path) -> dict[str, Any]:
    """Replay every receipt and publish only hashes, counts, and policy."""

    if (
        not receipt_paths
        or len(receipt_paths) != len(set(receipt_paths))
        or output.exists()
        or output.is_symlink()
    ):
        raise SourceQualityGatePublicationError(
            "quality-gate publication geometry differs"
        )
    populations = []
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    identity_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    hard_rejects = []
    for order, path in enumerate(receipt_paths):
        try:
            receipt = validate(path)
        except RuntimeError as error:
            raise SourceQualityGatePublicationError(
                "quality-gate receipt replay differs"
            ) from error
        identities, contents, rejected = _decision_identities(receipt)
        identity_counts.update(identities)
        content_counts.update(contents)
        hard_rejects.extend(rejected)
        decision_counts.update(receipt["decision_counts"])
        reason_counts.update(receipt["reason_counts"])
        populations.append(
            {
                "order": order,
                "receipt_file": path.name,
                "receipt_file_sha256": sha256_file(path),
                "receipt_sha256": receipt["receipt_sha256"],
                "source_rows": receipt["source"]["rows"],
                "source_bytes": receipt["source"]["bytes"],
                "source_sha256": receipt["source"]["sha256"],
                "source_ordered_identities_sha256": receipt["source"][
                    "ordered_identities_sha256"
                ],
                "decisions_bytes": receipt["decisions"]["bytes"],
                "decisions_sha256": receipt["decisions"]["sha256"],
                "decision_counts": receipt["decision_counts"],
                "reason_counts": receipt["reason_counts"],
            }
        )
    duplicate_identities = sorted(
        identity for identity, count in identity_counts.items() if count > 1
    )
    duplicate_contents = sorted(
        identity for identity, count in content_counts.items() if count > 1
    )
    assignment_rows = sum(identity_counts.values())
    payload = {
        "schema": SCHEMA,
        "status": "complete_source_safe_mechanical_quality_gate_publication",
        "policy": POLICY,
        "policy_sha256": POLICY_SHA256,
        "populations": populations,
        "population_assignment_rows": assignment_rows,
        "unique_candidate_rows": len(identity_counts),
        "cross_population_duplicate_identity_rows": len(duplicate_identities),
        "cross_population_duplicate_assignments": assignment_rows
        - len(identity_counts),
        "ordered_cross_population_duplicate_identities_sha256": canonical_sha256(
            duplicate_identities
        ),
        "unique_source_content_rows": len(content_counts),
        "cross_population_duplicate_content_rows": len(duplicate_contents),
        "cross_population_duplicate_content_assignments": assignment_rows
        - len(content_counts),
        "ordered_cross_population_duplicate_contents_sha256": canonical_sha256(
            duplicate_contents
        ),
        "decision_counts": dict(sorted(decision_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "hard_reject_assignment_rows": len(hard_rejects),
        "ordered_hard_reject_identities_sha256": canonical_sha256(sorted(hard_rejects)),
        "publication_contains_source_text": False,
        "all_nonpass_rows_excluded_from_direct_admission": True,
        "mechanical_pass_is_semantic_admission": False,
        "semantic_admission_complete": False,
        "rights_admission_complete": False,
        "benchmark_decontamination_complete": False,
        "global_deduplication_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_publication(args.receipt, args.output)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "population_assignment_rows": payload["population_assignment_rows"],
                "unique_candidate_rows": payload["unique_candidate_rows"],
                "decision_counts": payload["decision_counts"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
