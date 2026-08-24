"""Compile benchmark-disjoint bridges into nontraining curriculum candidates."""

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
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.data_compiler_labeling import DOMAINS
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.grounded_bridge_decontamination import (
    CLEAN_SCHEMA,
)
from sai.data.grounded_bridge_decontamination import (
    SCHEMA as DECONTAMINATION_SCHEMA,
)
from sai.data.grounded_bridge_population import ROW_SCHEMA as ANCHOR_ROW_SCHEMA
from sai.data.grounded_bridge_population import SCHEMA as ANCHOR_POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

ROW_SCHEMA = "sai-grounded-bridge-curriculum-candidate-v1"
RECEIPT_SCHEMA = "sai-grounded-bridge-curriculum-candidates-v1"
STATUS = "complete_nontraining_grounded_bridge_curriculum_candidates"
DOCUMENT_TYPES = (
    "bridge_overview",
    "verified_representation",
    "analogy_limits",
    "verification_questions",
)
SPLIT_POLICY = {
    "name": "sai-grounded-bridge-pair-disjoint-split-v1",
    "grouping": "pair_identity_sha256",
    "bucket_modulus": 1_000,
    "development_buckets": 50,
    "development_fraction_ppm": 50_000,
}
SPLIT_POLICY_SHA256 = canonical_sha256(SPLIT_POLICY)


class GroundedBridgeCurriculumCandidatesError(RuntimeError):
    """Bridge screen, generated lesson, split, or output custody differs."""


def _split(pair_identity_sha256: str) -> tuple[int, str]:
    if (
        not isinstance(pair_identity_sha256, str)
        or len(pair_identity_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in pair_identity_sha256
        )
    ):
        raise GroundedBridgeCurriculumCandidatesError("bridge pair identity differs")
    bucket = int(pair_identity_sha256[:16], 16) % SPLIT_POLICY["bucket_modulus"]
    split = "development" if bucket < SPLIT_POLICY["development_buckets"] else "train"
    return bucket, split


def _strings(value: Any, *, allow_empty: bool = False) -> list[str]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise GroundedBridgeCurriculumCandidatesError("bridge strings differ")
    return value


def _domains(label: Any) -> list[str]:
    if not isinstance(label, str):
        raise GroundedBridgeCurriculumCandidatesError("bridge label differs")
    domains = sorted(set(label.split("::")))
    if len(domains) != 2 or any(domain not in DOMAINS for domain in domains):
        raise GroundedBridgeCurriculumCandidatesError("bridge label differs")
    return domains


def load_anchor_registry(
    population_path: Path, receipt_path: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Recover source metadata dropped from later text-safe verification rows."""

    if (
        not population_path.is_file()
        or population_path.is_symlink()
        or population_path.stat().st_nlink != 1
        or not receipt_path.is_file()
        or receipt_path.is_symlink()
        or receipt_path.stat().st_nlink != 1
    ):
        raise GroundedBridgeCurriculumCandidatesError(
            "bridge anchor population is unsafe"
        )
    receipt = _load_receipt(receipt_path)
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("population")
    if (
        receipt.get("schema") != ANCHOR_POPULATION_SCHEMA
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("source_disjoint_pairs") is not True
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
        or descriptor.get("rows") != receipt.get("selection", {}).get("selected_pairs")
        or population_path.stat().st_size != descriptor.get("bytes")
        or sha256_file(population_path) != descriptor.get("sha256")
    ):
        raise GroundedBridgeCurriculumCandidatesError(
            "bridge anchor population differs"
        )
    registry = {}
    try:
        with population_path.open() as handle:
            for line in handle:
                row = json.loads(line)
                pair = row.get("pair_identity_sha256")
                anchor_a = row.get("anchor_a")
                anchor_b = row.get("anchor_b")
                if (
                    row.get("schema") != ANCHOR_ROW_SCHEMA
                    or row.get("candidate_identity_sha256") != pair
                    or not isinstance(pair, str)
                    or pair in registry
                    or not isinstance(anchor_a, dict)
                    or not isinstance(anchor_b, dict)
                    or row.get("source_disjoint") is not True
                    or row.get("training_ready") is not False
                ):
                    raise GroundedBridgeCurriculumCandidatesError(
                        "bridge anchor row differs"
                    )
                anchors = []
                for anchor in (anchor_a, anchor_b):
                    source = anchor.get("source")
                    if (
                        not isinstance(source, dict)
                        or not source
                        or any(
                            not isinstance(key, str)
                            or not key
                            or not isinstance(value, str)
                            or not value
                            for key, value in source.items()
                        )
                    ):
                        raise GroundedBridgeCurriculumCandidatesError(
                            "bridge anchor source differs"
                        )
                    anchors.append(
                        {
                            "candidate_identity_sha256": anchor[
                                "candidate_identity_sha256"
                            ],
                            "source_content_sha256": anchor["source_content_sha256"],
                            "source": dict(sorted(source.items())),
                        }
                    )
                registry[pair] = anchors
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as error:
        raise GroundedBridgeCurriculumCandidatesError(
            "bridge anchor population differs"
        ) from error
    if len(registry) != descriptor.get("rows"):
        raise GroundedBridgeCurriculumCandidatesError("bridge anchor coverage differs")
    return registry, receipt


def _lesson_texts(row: dict[str, Any]) -> list[tuple[str, int, str]]:
    """Render only verified generated fields; introduce no factual claims."""

    thesis = row.get("bridge_thesis")
    shared = row.get("shared_structure")
    prerequisites = _strings(row.get("prerequisite_map"))
    failures = _strings(row.get("analogy_failure_modes"))
    representations = row.get("representations")
    questions = row.get("verification_questions")
    if (
        not isinstance(thesis, str)
        or not thesis.strip()
        or not isinstance(shared, str)
        or not shared.strip()
        or not isinstance(representations, list)
        or not representations
        or not isinstance(questions, list)
        or not questions
    ):
        raise GroundedBridgeCurriculumCandidatesError("bridge lesson differs")
    lessons = [
        (
            "bridge_overview",
            0,
            "Bridge thesis\n\n"
            + thesis
            + "\n\nShared structure\n\n"
            + shared
            + "\n\nPrerequisite path\n\n"
            + "\n".join(f"- {value}" for value in prerequisites),
        )
    ]
    for index, representation in enumerate(representations):
        if not isinstance(representation, dict):
            raise GroundedBridgeCurriculumCandidatesError(
                "bridge representation differs"
            )
        title = representation.get("title")
        kind = representation.get("type")
        text = representation.get("text")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (title, kind, text)
        ):
            raise GroundedBridgeCurriculumCandidatesError(
                "bridge representation differs"
            )
        lessons.append(
            (
                "verified_representation",
                index,
                f"{title}\n\nRepresentation type: {kind}\n\n{text}",
            )
        )
    lessons.append(
        (
            "analogy_limits",
            0,
            "Where this connection fails\n\n"
            + "\n".join(f"- {value}" for value in failures),
        )
    )
    rendered_questions = []
    for question in questions:
        if not isinstance(question, dict):
            raise GroundedBridgeCurriculumCandidatesError(
                "bridge verification question differs"
            )
        prompt = question.get("question")
        answer = question.get("expected_answer")
        side = question.get("anchor_side")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            or side not in {"A", "B", "both"}
        ):
            raise GroundedBridgeCurriculumCandidatesError(
                "bridge verification question differs"
            )
        rendered_questions.append(
            f"Question ({side}): {prompt}\nExpected answer: {answer}"
        )
    lessons.append(
        (
            "verification_questions",
            0,
            "Connection self-check\n\n" + "\n\n".join(rendered_questions),
        )
    )
    return lessons


def compile_bridge(
    row: dict[str, Any],
    receipt_sha256: str,
    anchors: list[dict[str, Any]],
    anchor_population_receipt_sha256: str,
) -> list[dict[str, Any]]:
    unsigned = {key: value for key, value in row.items() if key != "record_sha256"}
    if (
        row.get("schema") != CLEAN_SCHEMA
        or row.get("record_sha256") != canonical_sha256(unsigned)
        or row.get("source_disjoint") is not True
        or row.get("independent_model_family_verification_complete") is not True
        or row.get("same_family_route") != "retain"
        or row.get("independent_family_route") != "retain"
        or row.get("same_family_retention_passed") is not True
        or row.get("independent_family_retention_passed") is not True
        or row.get("benchmark_decontamination_complete") is not True
        or row.get("global_deduplication_complete") is not False
        or row.get("transfer_ablation_complete") is not False
        or row.get("bridge_verified") is not False
        or row.get("training_ready") is not False
    ):
        raise GroundedBridgeCurriculumCandidatesError("clean bridge differs")
    domains = _domains(row.get("bridge_label"))
    prerequisites = _strings(row.get("prerequisite_map"))
    pair = row.get("pair_identity_sha256")
    if (
        not isinstance(anchors, list)
        or len(anchors) != 2
        or [anchor.get("candidate_identity_sha256") for anchor in anchors]
        != [
            row.get("anchor_a_candidate_identity_sha256"),
            row.get("anchor_b_candidate_identity_sha256"),
        ]
        or [anchor.get("source_content_sha256") for anchor in anchors]
        != [
            row.get("anchor_a_source_content_sha256"),
            row.get("anchor_b_source_content_sha256"),
        ]
    ):
        raise GroundedBridgeCurriculumCandidatesError("bridge anchor lineage differs")
    bucket, split = _split(pair)
    confidence = row.get("verification_confidence_ppm")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int)
        or not 0 <= confidence <= 1_000_000
    ):
        raise GroundedBridgeCurriculumCandidatesError("bridge confidence differs")
    source_custody = canonical_sha256(
        {
            "decontamination_receipt_sha256": receipt_sha256,
            "anchor_population_receipt_sha256": anchor_population_receipt_sha256,
            "clean_bridge_record_sha256": row["record_sha256"],
            "anchor_candidate_identity_sha256s": [
                anchor["candidate_identity_sha256"] for anchor in anchors
            ],
            "anchor_source_content_sha256s": [
                anchor["source_content_sha256"] for anchor in anchors
            ],
            "anchor_sources": [anchor["source"] for anchor in anchors],
            "generator_receipt_sha256": row["generator_receipt_sha256"],
            "same_family_route": row["same_family_route"],
            "independent_verification_receipt_sha256": row[
                "verification_receipt_sha256"
            ],
        }
    )
    result = []
    for document_type, document_index, text in _lesson_texts(row):
        text = text.strip()
        content_sha256 = hashlib.sha256(text.encode()).hexdigest()
        normalized_content_sha256 = hashlib.sha256(
            " ".join(text.casefold().split()).encode()
        ).hexdigest()
        candidate = {
            "schema": ROW_SCHEMA,
            "pair_identity_sha256": pair,
            "clean_bridge_record_sha256": row["record_sha256"],
            "document_type": document_type,
            "document_index": document_index,
            "document_identity_sha256": canonical_sha256(
                {
                    "pair_identity_sha256": pair,
                    "document_type": document_type,
                    "document_index": document_index,
                }
            ),
            "content_sha256": content_sha256,
            "normalized_content_sha256": normalized_content_sha256,
            "text": text,
            "text_utf8_bytes": len(text.encode()),
            "semantic_domains": domains,
            "prerequisites": sorted(set(prerequisites)),
            "difficulty_milli": min(4_000, max(1_000, len(prerequisites) * 500)),
            "verification_confidence_ppm": confidence,
            "source_group_sha256": canonical_sha256(
                {"bridge_pair_identity_sha256": pair}
            ),
            "source_group_bucket": bucket,
            "corpus_split": split,
            "split_policy_sha256": SPLIT_POLICY_SHA256,
            "source_custody_sha256": source_custody,
            "bridge_pair_disjoint_split_complete": True,
            "source_disjoint_against_foundation_complete": False,
            "benchmark_decontamination_complete": True,
            "independent_model_family_verification_complete": True,
            "global_deduplication_against_foundation_complete": False,
            "transfer_ablation_complete": False,
            "bridge_verified": False,
            "training_ready": False,
        }
        candidate["record_sha256"] = canonical_sha256(candidate)
        result.append(candidate)
    return result


def build_candidates(
    decontamination_root: Path,
    anchor_population_path: Path,
    anchor_population_receipt_path: Path,
    output_root: Path,
    durable_receipt: Path,
) -> dict[str, Any]:
    """Compile every clean bridge while preserving all remaining open gates."""

    if (
        output_root.exists()
        or output_root.is_symlink()
        or durable_receipt.exists()
        or durable_receipt.is_symlink()
    ):
        raise GroundedBridgeCurriculumCandidatesError("bridge output differs")
    anchors_by_pair, anchor_population_receipt = load_anchor_registry(
        anchor_population_path, anchor_population_receipt_path
    )
    receipt = _load_receipt(decontamination_root / "receipt.json")
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    descriptor = receipt.get("benchmark_disjoint_bridges")
    if (
        receipt.get("schema") != DECONTAMINATION_SCHEMA
        or receipt.get("status") != "complete_post_generation_bridge_benchmark_screen"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned_receipt)
        or receipt.get("post_generation_benchmark_screen_complete") is not True
        or receipt.get("independent_model_family_verification_complete") is not True
        or receipt.get("global_deduplication_complete") is not False
        or receipt.get("transfer_ablation_complete") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
    ):
        raise GroundedBridgeCurriculumCandidatesError("bridge decontamination differs")
    path = _bound_file(decontamination_root, descriptor)
    rows = []
    seen_documents = set()
    seen_contents = set()
    seen_normalized_contents = set()
    clean_records = []
    try:
        with path.open() as handle:
            for line in handle:
                clean = json.loads(line)
                clean_records.append(clean["record_sha256"])
                anchors = anchors_by_pair.get(clean.get("pair_identity_sha256"))
                if anchors is None:
                    raise GroundedBridgeCurriculumCandidatesError(
                        "clean bridge anchor population differs"
                    )
                for row in compile_bridge(
                    clean,
                    receipt["receipt_sha256"],
                    anchors,
                    anchor_population_receipt["receipt_sha256"],
                ):
                    identity = row["document_identity_sha256"]
                    content = row["content_sha256"]
                    normalized_content = row["normalized_content_sha256"]
                    if (
                        identity in seen_documents
                        or content in seen_contents
                        or normalized_content in seen_normalized_contents
                    ):
                        raise GroundedBridgeCurriculumCandidatesError(
                            "bridge curriculum candidate duplicate differs"
                        )
                    seen_documents.add(identity)
                    seen_contents.add(content)
                    seen_normalized_contents.add(normalized_content)
                    rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError) as error:
        raise GroundedBridgeCurriculumCandidatesError(
            "bridge candidate population differs"
        ) from error
    if (
        len(clean_records) != descriptor.get("rows")
        or canonical_sha256(clean_records) != descriptor.get("ordered_records_sha256")
        or not rows
    ):
        raise GroundedBridgeCurriculumCandidatesError(
            "bridge candidate coverage differs"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.partial.{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        output = stage / "curriculum_candidates.jsonl"
        _atomic_jsonl(output, rows)
        counts: Counter[str] = Counter()
        for row in rows:
            counts["documents"] += 1
            counts["text_utf8_bytes"] += row["text_utf8_bytes"]
            counts[f"split::{row['corpus_split']}::documents"] += 1
            counts[f"split::{row['corpus_split']}::text_utf8_bytes"] += row[
                "text_utf8_bytes"
            ]
            counts[f"type::{row['document_type']}::documents"] += 1
            for domain in row["semantic_domains"]:
                counts[f"domain::{domain}::documents"] += 1
        payload = {
            "schema": RECEIPT_SCHEMA,
            "status": STATUS,
            "source_decontamination_receipt_sha256": receipt["receipt_sha256"],
            "anchor_population_receipt_sha256": anchor_population_receipt[
                "receipt_sha256"
            ],
            "anchor_population_file_sha256": sha256_file(anchor_population_path),
            "split_policy": SPLIT_POLICY,
            "split_policy_sha256": SPLIT_POLICY_SHA256,
            "clean_bridges": len(clean_records),
            "private_candidate_root_name": output_root.name,
            "counts": dict(sorted(counts.items())),
            "curriculum_candidates": {
                "path": output.name,
                "rows": len(rows),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in rows]
                ),
                "generated_text_persisted": True,
                "source_anchor_text_persisted": False,
            },
            "exact_document_identity_unique": True,
            "exact_content_identity_unique_within_component": True,
            "normalized_content_identity_unique_within_component": True,
            "bridge_pair_disjoint_split_complete": True,
            "source_disjoint_against_foundation_complete": False,
            "source_disjoint_split_complete": False,
            "benchmark_decontamination_complete": True,
            "independent_model_family_verification_complete": True,
            "global_deduplication_against_foundation_complete": False,
            "transfer_ablation_complete": False,
            "bridge_verified": False,
            "huggingface_publication_authorized": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(stage / "receipt.json", payload)
        os.replace(stage, output_root)
        try:
            _atomic_create(durable_receipt, payload)
        except BaseException:
            shutil.rmtree(output_root, ignore_errors=True)
            raise
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decontamination-root", type=Path, required=True)
    parser.add_argument("--anchor-population", type=Path, required=True)
    parser.add_argument("--anchor-population-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--durable-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build_candidates(
        args.decontamination_root,
        args.anchor_population,
        args.anchor_population_receipt,
        args.output_root,
        args.durable_receipt,
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
