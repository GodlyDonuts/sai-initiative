"""Compare Hermès and independent book judgments without exposing text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books_compiler_aggregate import (
    _load_json,
    _validate_population,
    _validate_receipt,
)
from sai.data.institutional_books_quarantine_manifest import _load_aggregate
from sai.data.institutional_books_semantic_decision import (
    POLICY_SHA256,
    classify_judgment,
)
from sai.data.nous_label_worker import DEFAULT_MODEL
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-institutional-books-independent-agreement-receipt-v1"
RECORD_SCHEMA = "sai-institutional-books-independent-agreement-v1"
INDEPENDENT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


class InstitutionalBooksIndependentAgreementError(RuntimeError):
    """Book population, receipt, or independent agreement differs."""


def consensus_curriculum_metadata(
    original: dict[str, Any], independent: dict[str, Any]
) -> dict[str, Any]:
    """Keep only conservative, source-text-free curriculum agreement evidence."""

    quality_keys = sorted(set(original["quality"]).intersection(independent["quality"]))
    complexity_keys = sorted(
        set(original["complexity"]).intersection(independent["complexity"])
    )
    original_edges = {
        (row["prerequisite"], row["dependent"], row["relation"]): row
        for row in original["concept_edges"]
    }
    independent_edges = {
        (row["prerequisite"], row["dependent"], row["relation"]): row
        for row in independent["concept_edges"]
    }
    shared_edges = []
    for key in sorted(set(original_edges).intersection(independent_edges)):
        first = original_edges[key]
        second = independent_edges[key]
        shared_edges.append(
            {
                "prerequisite": key[0],
                "dependent": key[1],
                "relation": key[2],
                "confidence_floor_ppm": min(
                    first["confidence_ppm"], second["confidence_ppm"]
                ),
                "evidence_quote_sha256s": sorted(
                    {
                        hashlib.sha256(first["evidence_quote"].encode()).hexdigest(),
                        hashlib.sha256(second["evidence_quote"].encode()).hexdigest(),
                    }
                ),
            }
        )
    metadata = {
        "work_id_candidates": sorted(
            {original["work_id_candidate"], independent["work_id_candidate"]}
        ),
        "edition_id_candidates": sorted(
            {original["edition_id_candidate"], independent["edition_id_candidate"]}
        ),
        "shared_subdomains": sorted(
            set(original["subdomains"]).intersection(independent["subdomains"])
        ),
        "styles": sorted({original["style"], independent["style"]}),
        "quality_floor": {
            key: min(original["quality"][key], independent["quality"][key])
            for key in quality_keys
        },
        "complexity_range": {
            key: {
                "minimum": min(
                    original["complexity"][key], independent["complexity"][key]
                ),
                "maximum": max(
                    original["complexity"][key], independent["complexity"][key]
                ),
            }
            for key in complexity_keys
        },
        "curriculum_band_votes": sorted(
            {original["curriculum_band"], independent["curriculum_band"]}
        ),
        "shared_prerequisites": sorted(
            set(original["prerequisites"]).intersection(independent["prerequisites"])
        ),
        "shared_concepts": sorted(
            set(original["concepts"]).intersection(independent["concepts"])
        ),
        "shared_concept_edges": shared_edges,
        "shared_period": sorted(
            set(original["period"]).intersection(independent["period"])
        ),
        "shared_culture_geography": sorted(
            set(original["culture_geography"]).intersection(
                independent["culture_geography"]
            )
        ),
        "shared_recommended_representations": sorted(
            set(original["recommended_representations"]).intersection(
                independent["recommended_representations"]
            )
        ),
        "translation_type_votes": sorted(
            {original["translation_type"], independent["translation_type"]}
        ),
        "confidence_floor_ppm": min(
            original["confidence_ppm"], independent["confidence_ppm"]
        ),
        "source_text_persisted": False,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    return metadata


def assign_work_families(records: list[dict[str, Any]]) -> None:
    """Union overlapping two-family work candidates before split assignment."""

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        root = value
        while parent[root] != root:
            root = parent[root]
        while value != root:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def union(first: str, second: str) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[max(left, right)] = min(left, right)

    for record in records:
        metadata = record.get("consensus_curriculum")
        if metadata is None:
            continue
        candidates = metadata.get("work_id_candidates")
        if (
            not isinstance(candidates, list)
            or not candidates
            or len(candidates) != len(set(candidates))
        ):
            raise InstitutionalBooksIndependentAgreementError(
                "work candidate family differs"
            )
        for value in candidates:
            parent.setdefault(value, value)
        for value in candidates[1:]:
            union(candidates[0], value)
    members: dict[str, list[str]] = {}
    for value in sorted(parent):
        members.setdefault(find(value), []).append(value)
    family_by_work = {
        value: canonical_sha256(
            {
                "schema": "sai-institutional-book-work-family-v1",
                "connected_work_id_candidates": values,
            }
        )
        for values in members.values()
        for value in values
    }
    for record in records:
        metadata = record.get("consensus_curriculum")
        record["work_family_sha256"] = (
            family_by_work[metadata["work_id_candidates"][0]]
            if metadata is not None
            else None
        )
        record["record_sha256"] = canonical_sha256(record)


def agreement_disposition(
    original: dict[str, Any], independent: dict[str, Any]
) -> tuple[str, list[str]]:
    """Require both strict policy passes plus conservative taxonomy agreement."""

    original_disposition, original_reasons = classify_judgment(original)
    independent_disposition, independent_reasons = classify_judgment(independent)
    reasons = []
    if original_disposition != "independent_verification":
        reasons.extend(f"original:{reason}" for reason in original_reasons)
        reasons.append("original_no_longer_satisfies_policy")
    if independent_disposition != "independent_verification":
        reasons.extend(f"independent:{reason}" for reason in independent_reasons)
        reasons.append("independent_does_not_satisfy_policy")
    if original.get("genre") != independent.get("genre"):
        reasons.append("genre_disagreement")
    original_domains = original.get("domains")
    independent_domains = independent.get("domains")
    if (
        not isinstance(original_domains, list)
        or not isinstance(independent_domains, list)
        or not set(original_domains).intersection(independent_domains)
    ):
        reasons.append("domain_disagreement")
    if reasons:
        return "agreement_hold", sorted(set(reasons))
    return "consensus_candidate", []


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise InstitutionalBooksIndependentAgreementError("agreement output exists")
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


def build_agreement(
    source_population_root: Path,
    independent_population_root: Path,
    original_judgments_root: Path,
    independent_judgments_root: Path,
    independent_aggregate_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Replay two model families and emit strict text-free agreement records."""

    if output_root.exists() or output_root.is_symlink():
        raise InstitutionalBooksIndependentAgreementError("agreement root exists")
    try:
        source_candidates, source_population = _validate_population(
            source_population_root
        )
        independent_candidates, independent_population = _validate_population(
            independent_population_root
        )
    except RuntimeError as error:
        raise InstitutionalBooksIndependentAgreementError(
            "book population differs"
        ) from error
    aggregate = _load_aggregate(independent_aggregate_path)
    if (
        aggregate.get("population", {}).get("receipt_sha256")
        != independent_population["receipt_sha256"]
        or aggregate.get("population", {}).get("rows") != len(independent_candidates)
        or aggregate.get("model") != INDEPENDENT_MODEL
    ):
        raise InstitutionalBooksIndependentAgreementError(
            "independent aggregate differs"
        )
    source_by_identity = {
        row["candidate_identity_sha256"]: row for row in source_candidates
    }
    independent_by_identity = {
        row["candidate_identity_sha256"]: row for row in independent_candidates
    }
    if (
        not set(independent_by_identity).issubset(source_by_identity)
        or independent_population.get("source", {}).get("population_receipt_sha256")
        != source_population["receipt_sha256"]
    ):
        raise InstitutionalBooksIndependentAgreementError(
            "independent subset binding differs"
        )
    expected_independent = {
        independent_judgments_root / f"{identity}.book-compiler.json"
        for identity in independent_by_identity
    }
    if (
        set(independent_judgments_root.glob("*.book-compiler.json"))
        != expected_independent
    ):
        raise InstitutionalBooksIndependentAgreementError(
            "independent judgment population differs"
        )
    records = []
    counts: Counter[str] = Counter()
    tokens: Counter[str] = Counter()
    for identity in sorted(independent_by_identity):
        candidate = independent_by_identity[identity]
        try:
            original_receipt = _validate_receipt(
                _load_json(
                    original_judgments_root / f"{identity}.book-compiler.json",
                    "original book receipt",
                ),
                candidate,
                DEFAULT_MODEL,
            )
            independent_receipt = _validate_receipt(
                _load_json(
                    independent_judgments_root / f"{identity}.book-compiler.json",
                    "independent book receipt",
                ),
                candidate,
                INDEPENDENT_MODEL,
            )
        except RuntimeError as error:
            raise InstitutionalBooksIndependentAgreementError(
                "book judgment differs"
            ) from error
        disposition, reasons = agreement_disposition(
            original_receipt["judgment"], independent_receipt["judgment"]
        )
        token_count = candidate["measurements"]["token_count_o200k_base_gen"]
        record = {
            "schema": RECORD_SCHEMA,
            "candidate_identity_sha256": identity,
            "source_book_id": candidate["source"]["barcode_src"],
            "source_content_sha256": candidate["source_content_sha256"],
            "source_provenance_sha256": candidate["provenance_sha256"],
            "policy_sha256": POLICY_SHA256,
            "original_model": DEFAULT_MODEL,
            "original_receipt_sha256": original_receipt["receipt_sha256"],
            "original_judgment_sha256": original_receipt["judgment"]["judgment_sha256"],
            "independent_model": INDEPENDENT_MODEL,
            "independent_receipt_sha256": independent_receipt["receipt_sha256"],
            "independent_judgment_sha256": independent_receipt["judgment"][
                "judgment_sha256"
            ],
            "disposition": disposition,
            "reasons": reasons,
            "agreed_genre": original_receipt["judgment"]["genre"]
            if disposition == "consensus_candidate"
            else None,
            "shared_domains": sorted(
                set(original_receipt["judgment"]["domains"]).intersection(
                    independent_receipt["judgment"]["domains"]
                )
            ),
            "consensus_curriculum": (
                consensus_curriculum_metadata(
                    original_receipt["judgment"], independent_receipt["judgment"]
                )
                if disposition == "consensus_candidate"
                else None
            ),
            "token_count_o200k_base_gen": token_count,
            "benchmark_decontamination_complete": False,
            "global_semantic_deduplication_complete": False,
            "source_text_persisted": False,
            "training_ready": False,
        }
        records.append(record)
        counts[disposition] += 1
        tokens[disposition] += token_count
    assign_work_families(records)
    output_root.mkdir(parents=True)
    try:
        manifest_path = output_root / "agreement.jsonl"
        _atomic_jsonl(manifest_path, records)
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_independent_book_agreement",
            "source_population_receipt_sha256": source_population["receipt_sha256"],
            "independent_population_receipt_sha256": independent_population[
                "receipt_sha256"
            ],
            "independent_aggregate": {
                "receipt_sha256": aggregate["receipt_sha256"],
                "file_sha256": sha256_file(independent_aggregate_path),
            },
            "policy_sha256": POLICY_SHA256,
            "counts": dict(sorted(counts.items())),
            "tokens": dict(sorted(tokens.items())),
            "manifest": {
                "path": manifest_path.name,
                "rows": len(records),
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in records]
                ),
            },
            "consensus_is_training_admission": False,
            "source_text_persisted": False,
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
    parser.add_argument("--independent-population-root", type=Path, required=True)
    parser.add_argument("--original-judgments-root", type=Path, required=True)
    parser.add_argument("--independent-judgments-root", type=Path, required=True)
    parser.add_argument("--independent-aggregate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_agreement(
        args.source_population_root,
        args.independent_population_root,
        args.original_judgments_root,
        args.independent_judgments_root,
        args.independent_aggregate,
        args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
