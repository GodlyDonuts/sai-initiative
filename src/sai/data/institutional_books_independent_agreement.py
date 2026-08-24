"""Compare Hermès and independent book judgments without exposing text."""

from __future__ import annotations

import argparse
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
            "token_count_o200k_base_gen": token_count,
            "benchmark_decontamination_complete": False,
            "global_semantic_deduplication_complete": False,
            "source_text_persisted": False,
            "training_ready": False,
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)
        counts[disposition] += 1
        tokens[disposition] += token_count
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
