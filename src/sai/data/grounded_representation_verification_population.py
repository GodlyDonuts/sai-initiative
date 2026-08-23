"""Freeze exact source/generated pairs for grounded representation verification."""

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
from sai.data.grounded_representation_aggregate import (
    REPRESENTATION_SCHEMA,
    validate_receipt,
)
from sai.data.grounded_representation_aggregate import (
    load_population as load_source_population,
)
from sai.data.grounded_representation_decontamination import (
    CLEAN_SCHEMA,
)
from sai.data.grounded_representation_decontamination import (
    SCHEMA as DECONTAMINATION_SCHEMA,
)
from sai.data.nous_grounded_representation_worker import OUTPUT_SUFFIX
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-representation-verification-population-v1"
RECORD_SCHEMA = "sai-grounded-representation-verification-candidate-v1"


class GroundedRepresentationVerificationPopulationError(RuntimeError):
    """A source, generated representation, or clean-row binding differs."""


def load_clean_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the benchmark-disjoint generated representation population."""

    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("benchmark_disjoint_representations")
    if (
        receipt.get("schema") != DECONTAMINATION_SCHEMA
        or receipt.get("status") != "complete_post_generation_benchmark_screen"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("post_generation_benchmark_screen_complete") is not True
        or receipt.get("source_claims_independently_verified") is not False
        or receipt.get("representation_verification_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise GroundedRepresentationVerificationPopulationError(
            "representation decontamination receipt differs"
        )
    path = _bound_file(root, descriptor)
    rows = []
    identities = set()
    try:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                identity = row.get("record_sha256")
                unsigned_row = {
                    key: value for key, value in row.items() if key != "record_sha256"
                }
                if (
                    row.get("schema") != CLEAN_SCHEMA
                    or not isinstance(identity, str)
                    or len(identity) != 64
                    or identity in identities
                    or row.get("record_sha256") != canonical_sha256(unsigned_row)
                    or row.get("benchmark_decontamination_complete") is not True
                    or row.get("representation_verified") is not False
                    or row.get("training_ready") is not False
                ):
                    raise GroundedRepresentationVerificationPopulationError(
                        "clean representation row differs"
                    )
                identities.add(identity)
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GroundedRepresentationVerificationPopulationError(
            "clean representation row differs"
        ) from error
    if (
        not rows
        or len(rows) != descriptor.get("rows")
        or descriptor.get("ordered_records_sha256")
        != canonical_sha256([row["record_sha256"] for row in rows])
        or descriptor.get("text_bytes")
        != sum(len(row["text"].encode()) for row in rows)
    ):
        raise GroundedRepresentationVerificationPopulationError(
            "clean representation coverage differs"
        )
    return rows, receipt


def build_candidate(
    clean: dict[str, Any],
    generated: dict[str, Any],
    source: dict[str, Any],
    generator_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Bind one clean generated text to its source and literal citations."""

    index = clean.get("representation_index")
    judgment = generator_receipt.get("judgment")
    representations = (
        judgment.get("representations") if isinstance(judgment, dict) else None
    )
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or not isinstance(representations, list)
        or not 0 <= index < len(representations)
    ):
        raise GroundedRepresentationVerificationPopulationError(
            "verification representation index differs"
        )
    original = representations[index]
    evidence = original.get("evidence_quotes")
    if (
        clean.get("pre_decontamination_record_sha256") != generated.get("record_sha256")
        or clean.get("source_candidate_identity_sha256")
        != source.get("candidate_identity_sha256")
        or generated.get("source_candidate_identity_sha256")
        != source.get("candidate_identity_sha256")
        or generated.get("generator_receipt_sha256")
        != generator_receipt.get("receipt_sha256")
        or generated.get("representation_index") != index
        or clean.get("representation_type") != original.get("type")
        or clean.get("title") != original.get("title")
        or clean.get("text") != original.get("text")
        or clean.get("concepts") != original.get("concepts")
        or clean.get("difficulty") != original.get("difficulty")
        or not isinstance(evidence, list)
        or not evidence
        or clean.get("evidence_quote_sha256s")
        != [hashlib.sha256(quote.encode()).hexdigest() for quote in evidence]
        or any(quote not in source.get("text", "") for quote in evidence)
    ):
        raise GroundedRepresentationVerificationPopulationError(
            "verification source/generated binding differs"
        )
    row = {
        "schema": RECORD_SCHEMA,
        "source_text": source["text"],
        "source_text_sha256": source["source_text_sha256"],
        "generated_text": clean["text"],
        "generated_text_sha256": clean["text_sha256"],
        "source_evidence_quotes": evidence,
        "source": source["source"],
        "source_candidate_identity_sha256": source["candidate_identity_sha256"],
        "generated_record_sha256": generated["record_sha256"],
        "clean_record_sha256": clean["record_sha256"],
        "generator_receipt_sha256": generator_receipt["receipt_sha256"],
        "generator_judgment_sha256": judgment["judgment_sha256"],
        "representation_index": index,
        "representation_type": clean["representation_type"],
        "title": clean["title"],
        "concepts": clean["concepts"],
        "difficulty": clean["difficulty"],
        "benchmark_decontamination_complete": True,
        "same_model_family_as_generator": True,
        "independent_request_verification_complete": False,
        "independent_model_family_verification_complete": False,
        "representation_verified": False,
        "training_ready": False,
    }
    row["candidate_identity_sha256"] = canonical_sha256(row)
    return row


def build_population(
    source_population_root: Path,
    generator_judgments_root: Path,
    generated_aggregate_root: Path,
    decontamination_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Seal every clean source/generated pair for one verifier pass."""

    if output_root.exists() or output_root.is_symlink():
        raise GroundedRepresentationVerificationPopulationError(
            "verification population output differs"
        )
    sources, source_population = load_source_population(source_population_root)
    clean_rows, decontamination = load_clean_rows(decontamination_root)
    generated_descriptor = decontamination.get("aggregate")
    if (
        not isinstance(generated_descriptor, dict)
        or generated_descriptor.get("root_name") != generated_aggregate_root.name
        or generated_descriptor.get("receipt_file_sha256")
        != sha256_file(generated_aggregate_root / "receipt.json")
    ):
        raise GroundedRepresentationVerificationPopulationError(
            "generated aggregate binding differs"
        )
    generated_receipt = _load_receipt(generated_aggregate_root / "receipt.json")
    generated_path = _bound_file(
        generated_aggregate_root, generated_receipt["representations"]
    )
    generated_rows = []
    with generated_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("schema") != REPRESENTATION_SCHEMA:
                raise GroundedRepresentationVerificationPopulationError(
                    "generated representation row differs"
                )
            generated_rows.append(row)
    generated_by_identity = {row["record_sha256"]: row for row in generated_rows}
    source_by_identity = {row["candidate_identity_sha256"]: row for row in sources}
    if len(generated_by_identity) != len(generated_rows) or len(
        source_by_identity
    ) != len(sources):
        raise GroundedRepresentationVerificationPopulationError(
            "verification identity population differs"
        )
    receipt_cache: dict[str, dict[str, Any]] = {}
    output_rows = []
    for clean in clean_rows:
        source_identity = clean["source_candidate_identity_sha256"]
        source = source_by_identity.get(source_identity)
        generated = generated_by_identity.get(
            clean["pre_decontamination_record_sha256"]
        )
        receipt = receipt_cache.get(source_identity)
        if receipt is None:
            path = generator_judgments_root / (
                f"{source_identity}.{OUTPUT_SUFFIX}.json"
            )
            if source is None or generated is None or not path.is_file():
                raise GroundedRepresentationVerificationPopulationError(
                    "verification source identity differs"
                )
            receipt = validate_receipt(_load_receipt(path), source)
            receipt_cache[source_identity] = receipt
        if source is None or generated is None:
            raise GroundedRepresentationVerificationPopulationError(
                "verification source identity differs"
            )
        output_rows.append(build_candidate(clean, generated, source, receipt))
    identities = [row["candidate_identity_sha256"] for row in output_rows]
    if len(identities) != len(set(identities)):
        raise GroundedRepresentationVerificationPopulationError(
            "verification candidate identities differ"
        )
    output_root.mkdir(parents=True)
    try:
        candidates_path = output_root / "candidates.jsonl"
        _atomic_jsonl(candidates_path, output_rows)
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_verification_population",
            "source_population": {
                "root_name": source_population_root.name,
                "receipt_sha256": source_population["receipt_sha256"],
            },
            "generated_aggregate": {
                "root_name": generated_aggregate_root.name,
                "receipt_sha256": generated_receipt["receipt_sha256"],
            },
            "decontamination": {
                "root_name": decontamination_root.name,
                "receipt_sha256": decontamination["receipt_sha256"],
            },
            "candidates": {
                "path": candidates_path.name,
                "rows": len(output_rows),
                "bytes": candidates_path.stat().st_size,
                "sha256": sha256_file(candidates_path),
                "ordered_identities_sha256": canonical_sha256(identities),
                "source_text_bytes": sum(
                    len(row["source_text"].encode()) for row in output_rows
                ),
                "generated_text_bytes": sum(
                    len(row["generated_text"].encode()) for row in output_rows
                ),
            },
            "benchmark_decontamination_complete": True,
            "same_model_family_as_generator": True,
            "independent_request_verification_complete": False,
            "independent_model_family_verification_complete": False,
            "representation_verification_complete": False,
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
    parser.add_argument("--generator-judgments-root", type=Path, required=True)
    parser.add_argument("--generated-aggregate-root", type=Path, required=True)
    parser.add_argument("--decontamination-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_population(
        args.source_population_root,
        args.generator_judgments_root,
        args.generated_aggregate_root,
        args.decontamination_root,
        args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
