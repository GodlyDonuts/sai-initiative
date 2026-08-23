"""Bind generated cross-domain bridges to both exact anchors for verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.grounded_bridge_aggregate import (
    CANDIDATE_SCHEMA as GENERATED_SCHEMA,
)
from sai.data.grounded_bridge_aggregate import (
    SCHEMA as AGGREGATE_SCHEMA,
)
from sai.data.grounded_bridge_aggregate import (
    load_population as load_source_population,
)
from sai.data.grounded_bridge_aggregate import validate_receipt as validate_generator
from sai.data.nous_grounded_bridge_worker import OUTPUT_SUFFIX
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-cross-domain-bridge-verification-population-v1"
RECORD_SCHEMA = "sai-grounded-cross-domain-bridge-verification-candidate-v1"


class GroundedBridgeVerificationPopulationError(RuntimeError):
    """A source pair, generated bridge, or generator binding differs."""


def _generated_text(row: dict[str, Any]) -> str:
    parts = [
        f"Bridge thesis\n{row['bridge_thesis']}",
        f"Shared structure\n{row['shared_structure']}",
    ]
    for claim in row["claims"]:
        parts.append(f"Claim {claim['anchor_side']}\n{claim['claim']}")
    for representation in row["representations"]:
        parts.append(
            f"{representation['type']}\n{representation['title']}\n"
            f"{representation['text']}"
        )
    parts.append("Prerequisites\n" + "\n".join(row["prerequisite_map"]))
    parts.append("Analogy failure modes\n" + "\n".join(row["analogy_failure_modes"]))
    for question in row["verification_questions"]:
        parts.append(
            f"Verification question ({question['anchor_side']})\n"
            f"{question['question']}\n{question['expected_answer']}"
        )
    return "\n\n".join(parts)


def load_generated(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay exact unverified generated bridge candidates."""

    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("candidates")
    if (
        receipt.get("schema") != AGGREGATE_SCHEMA
        or receipt.get("status") != "complete_unverified_grounded_bridge_candidates"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("source_disjoint_pairs") is not True
        or receipt.get("grounded_synthesis_complete") is not True
        or receipt.get("independent_claim_verification_complete") is not False
        or receipt.get("independent_transfer_verification_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise GroundedBridgeVerificationPopulationError(
            "generated bridge aggregate differs"
        )
    path = _bound_file(root, descriptor)
    rows = []
    identities = []
    pairs = set()
    try:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                identity = row.get("candidate_identity_sha256")
                unsigned_row = {
                    key: value
                    for key, value in row.items()
                    if key != "candidate_identity_sha256"
                }
                if (
                    row.get("schema") != GENERATED_SCHEMA
                    or not isinstance(identity, str)
                    or len(identity) != 64
                    or identity != canonical_sha256(unsigned_row)
                    or row.get("pair_identity_sha256") in pairs
                    or row.get("source_disjoint") is not True
                    or row.get("source_quotes_retained_in_candidate") is not False
                    or row.get("grounded_synthesis_verified") is not False
                    or row.get("training_ready") is not False
                ):
                    raise GroundedBridgeVerificationPopulationError(
                        "generated bridge row differs"
                    )
                pairs.add(row["pair_identity_sha256"])
                identities.append(identity)
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GroundedBridgeVerificationPopulationError(
            "generated bridge row differs"
        ) from error
    if (
        not rows
        or len(rows) != descriptor.get("rows")
        or descriptor.get("ordered_identities_sha256") != canonical_sha256(identities)
    ):
        raise GroundedBridgeVerificationPopulationError(
            "generated bridge coverage differs"
        )
    return rows, receipt


def build_candidate(
    pair: dict[str, Any],
    generated: dict[str, Any],
    generator_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Restore exact evidence privately and bind one independent request."""

    judgment = generator_receipt.get("judgment")
    if not isinstance(judgment, dict):
        raise GroundedBridgeVerificationPopulationError("generator judgment differs")
    expected_claims = [
        {
            "claim": claim["claim"],
            "anchor_side": claim["anchor_side"],
            "evidence_quote_sha256": hashlib.sha256(
                claim["evidence_quote"].encode()
            ).hexdigest(),
        }
        for claim in judgment["claims"]
    ]
    if (
        generated.get("pair_identity_sha256") != pair.get("pair_identity_sha256")
        or generated.get("generator_receipt_sha256")
        != generator_receipt.get("receipt_sha256")
        or generated.get("generator_judgment_sha256") != judgment.get("judgment_sha256")
        or generated.get("anchor_a_candidate_identity_sha256")
        != pair.get("anchor_a", {}).get("candidate_identity_sha256")
        or generated.get("anchor_a_source_content_sha256")
        != pair.get("anchor_a", {}).get("source_content_sha256")
        or generated.get("anchor_b_candidate_identity_sha256")
        != pair.get("anchor_b", {}).get("candidate_identity_sha256")
        or generated.get("anchor_b_source_content_sha256")
        != pair.get("anchor_b", {}).get("source_content_sha256")
        or generated.get("bridge_label") != pair.get("bridge_label")
        or generated.get("claims") != expected_claims
        or generated.get("bridge_thesis") != judgment.get("bridge_thesis")
        or generated.get("shared_structure") != judgment.get("shared_structure")
        or generated.get("representations") != judgment.get("representations")
        or generated.get("prerequisite_map") != judgment.get("prerequisite_map")
        or generated.get("analogy_failure_modes")
        != judgment.get("analogy_failure_modes")
        or generated.get("verification_questions")
        != judgment.get("verification_questions")
    ):
        raise GroundedBridgeVerificationPopulationError(
            "source/generated bridge binding differs"
        )
    generated_text = _generated_text(generated)
    row = {
        "schema": RECORD_SCHEMA,
        "pair_identity_sha256": pair["pair_identity_sha256"],
        "bridge_label": pair["bridge_label"],
        "anchor_a_text": pair["anchor_a"]["text"],
        "anchor_a_source_content_sha256": pair["anchor_a"]["source_content_sha256"],
        "anchor_a_candidate_identity_sha256": pair["anchor_a"][
            "candidate_identity_sha256"
        ],
        "anchor_b_text": pair["anchor_b"]["text"],
        "anchor_b_source_content_sha256": pair["anchor_b"]["source_content_sha256"],
        "anchor_b_candidate_identity_sha256": pair["anchor_b"][
            "candidate_identity_sha256"
        ],
        "generated": generated,
        "generated_text": generated_text,
        "generated_text_sha256": hashlib.sha256(generated_text.encode()).hexdigest(),
        "generated_candidate_identity_sha256": generated["candidate_identity_sha256"],
        "generator_receipt_sha256": generator_receipt["receipt_sha256"],
        "generator_judgment_sha256": judgment["judgment_sha256"],
        "source_disjoint": True,
        "same_model_family_as_generator": True,
        "independent_request_verification_complete": False,
        "independent_model_family_verification_complete": False,
        "bridge_verified": False,
        "training_ready": False,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def build_population(
    source_population_root: Path,
    generator_judgments_root: Path,
    generated_aggregate_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Seal every source-paired generated bridge for one verifier pass."""

    if output_root.exists() or output_root.is_symlink():
        raise GroundedBridgeVerificationPopulationError(
            "bridge verification population output differs"
        )
    pairs, source_population = load_source_population(source_population_root)
    generated_rows, generated_aggregate = load_generated(generated_aggregate_root)
    pairs_by_identity = {row["pair_identity_sha256"]: row for row in pairs}
    if len(pairs_by_identity) != len(pairs) or len(generated_rows) != len(pairs):
        raise GroundedBridgeVerificationPopulationError(
            "bridge verification source coverage differs"
        )
    output_rows = []
    generator_hashes = []
    for generated in generated_rows:
        pair_identity = generated["pair_identity_sha256"]
        pair = pairs_by_identity.get(pair_identity)
        path = generator_judgments_root / f"{pair_identity}.{OUTPUT_SUFFIX}.json"
        if pair is None or not path.is_file():
            raise GroundedBridgeVerificationPopulationError(
                "bridge generator receipt is missing"
            )
        generator = validate_generator(_load_receipt(path), pair)
        generator_hashes.append(generator["receipt_sha256"])
        output_rows.append(build_candidate(pair, generated, generator))
    identities = [row["candidate_identity_sha256"] for row in output_rows]
    if len(identities) != len(set(identities)):
        raise GroundedBridgeVerificationPopulationError(
            "bridge verification identities differ"
        )
    output_root.mkdir(parents=True)
    try:
        candidates_path = output_root / "candidates.jsonl"
        _atomic_jsonl(candidates_path, output_rows)
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_bridge_verification_population",
            "source_population": {
                "root_name": source_population_root.name,
                "receipt_sha256": source_population["receipt_sha256"],
            },
            "generated_aggregate": {
                "root_name": generated_aggregate_root.name,
                "receipt_sha256": generated_aggregate["receipt_sha256"],
            },
            "ordered_generator_receipts_sha256": canonical_sha256(generator_hashes),
            "candidates": {
                "path": candidates_path.name,
                "rows": len(output_rows),
                "bytes": candidates_path.stat().st_size,
                "sha256": sha256_file(candidates_path),
                "ordered_identities_sha256": canonical_sha256(identities),
                "anchor_a_text_bytes": sum(
                    len(row["anchor_a_text"].encode()) for row in output_rows
                ),
                "anchor_b_text_bytes": sum(
                    len(row["anchor_b_text"].encode()) for row in output_rows
                ),
                "generated_text_bytes": sum(
                    len(row["generated_text"].encode()) for row in output_rows
                ),
            },
            "source_disjoint_pairs": True,
            "same_model_family_as_generator": True,
            "independent_request_verification_complete": False,
            "independent_model_family_verification_complete": False,
            "bridge_verification_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-population-root", type=Path, required=True)
    parser.add_argument("--generator-judgments-root", type=Path, required=True)
    parser.add_argument("--generated-aggregate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = build_population(
        args.source_population_root,
        args.generator_judgments_root,
        args.generated_aggregate_root,
        args.output_root,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["candidates"]["rows"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
