"""Compose semantic prerequisite order with within-phase model learnability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from sai.data.curriculum import PHASES
from sai.data.curriculum_control import _multiset_sha256, _Records
from sai.data.learnability_curriculum import (
    BANDS,
    _copy_permutation,
    _descriptor,
    _load_scores,
    _read_regular,
)
from sai.data.learnability_score import (
    OUTPUT_NAME,
    RECEIPT_NAME,
    validate_score_population,
)
from sai.data.prerequisite import replay_curriculum_annotation_files
from sai.data.token_stream import (
    SCHEMA as TOKEN_STREAM_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file, validate_frozen_stream

SCHEMA = "sai-semantic-learnability-curriculum-v1"
ORDER_SEED = 2026082203
_MAX_EVIDENCE_BYTES = 16 << 20


class SemanticLearnabilityError(RuntimeError):
    """Semantic lineage, phase locking, scores, or an exact record differs."""


def _phase_ranges(parent: dict[str, Any]) -> dict[str, range]:
    curriculum = parent.get("curriculum")
    if (
        not isinstance(curriculum, dict)
        or curriculum.get("phase_order") != list(PHASES)
        or curriculum.get("phase_token_budget_enforced") is not True
        or curriculum.get("phase_sequence_targets")
        != curriculum.get("phase_sequences_emitted")
    ):
        raise SemanticLearnabilityError("semantic phase stream contract differs")
    counts = curriculum["phase_sequences_emitted"]
    skipped = curriculum.get("phase_source_documents_skipped")
    truncated = curriculum.get("phase_documents_truncated")
    declared_documents = curriculum.get("declared_phase_documents")
    consumed_documents = curriculum.get("consumed_phase_documents")
    if (
        not isinstance(counts, dict)
        or set(counts) != set(PHASES)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in counts.values()
        )
        or sum(counts.values()) != parent["sequences"]
        or not all(
            isinstance(mapping, dict) and set(mapping) == set(PHASES)
            for mapping in (
                skipped,
                truncated,
                declared_documents,
                consumed_documents,
            )
        )
        or any(skipped[phase] != 0 or truncated[phase] != 0 for phase in PHASES)
        or declared_documents != consumed_documents
    ):
        raise SemanticLearnabilityError("semantic phase sequence geometry differs")
    result = {}
    offset = 0
    for phase in PHASES:
        result[phase] = range(offset, offset + counts[phase])
        offset += counts[phase]
    return result


def _evidence_bytes(path: Path, label: str) -> bytes:
    try:
        return _read_regular(path, label, maximum_bytes=_MAX_EVIDENCE_BYTES)
    except Exception as error:
        raise SemanticLearnabilityError(f"{label} differs") from error


def _semantic_evidence(
    parent: dict[str, Any],
    *,
    taxonomy: Path,
    curriculum_receipt: Path,
    annotations: Path,
    progression_report: Path,
    workers: int,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    evidence = {
        "taxonomy": _evidence_bytes(taxonomy, "taxonomy"),
        "curriculum_receipt": _evidence_bytes(curriculum_receipt, "curriculum receipt"),
        "annotations": _evidence_bytes(annotations, "semantic annotations"),
        "progression_report": _evidence_bytes(
            progression_report, "semantic progression report"
        ),
    }
    try:
        supplied = json.loads(evidence["progression_report"])
        replayed = replay_curriculum_annotation_files(
            taxonomy,
            curriculum_receipt,
            annotations,
            workers=workers,
        )
    except Exception as error:
        raise SemanticLearnabilityError("semantic progression replay failed") from error
    lineage = replayed.get("curriculum_lineage")
    source = parent.get("source_receipts")
    if (
        supplied != replayed
        or replayed.get("status") != "qualified"
        or replayed.get("progression_qualified") is not True
        or replayed.get("training_authorized") is not False
        or replayed.get("four_b_training_authorized") is not False
        or replayed.get("violations") != []
        or replayed.get("premature_exposure_violations") != []
        or replayed.get("concept_density_violations") != []
        or replayed.get("phase_coverage_violations") != []
        or replayed.get("missing_concepts") != []
        or not isinstance(lineage, dict)
        or not isinstance(source, list)
        or len(source) != 1
        or parent.get("source_qualification_sha256")
        != hashlib.sha256(evidence["curriculum_receipt"]).hexdigest()
    ):
        raise SemanticLearnabilityError("semantic progression evidence differs")
    if (
        source[0]["path"] != str(Path(source[0]["path"]).resolve())
        or source[0]["bytes"] != lineage["curriculum_output_bytes"]
        or source[0]["sha256"] != lineage["curriculum_output_sha256"]
        or str(annotations.resolve()) != lineage["annotations_path"]
        or len(evidence["annotations"]) != lineage["annotations_bytes"]
        or hashlib.sha256(evidence["annotations"]).hexdigest()
        != lineage["annotations_file_sha256"]
    ):
        raise SemanticLearnabilityError("semantic source lineage differs")
    return replayed, evidence


def _score_inputs(
    scores_root: Path,
    parent_root: Path,
    parent: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], bytes, bytes]:
    try:
        receipt = validate_score_population(scores_root)
        receipt_bytes = _evidence_bytes(
            scores_root / RECEIPT_NAME, "learnability score receipt"
        )
        target = receipt["target_stream"]
        if (
            target["path"] != str(parent_root.resolve())
            or target["receipt_file_sha256"]
            != sha256_file(parent_root / "stream_receipt.json")
            or target["ordered_stream_identity_sha256"]
            != parent["ordered_stream_identity_sha256"]
            or target["source_manifest_sha256"] != parent["source_manifest_sha256"]
            or target["tokenizer_identity_sha256"]
            != parent["tokenizer_identity_sha256"]
            or target["sequences"] != parent["sequences"]
            or target["sequence_length"] != parent["sequence_length"]
        ):
            raise SemanticLearnabilityError("learnability target stream differs")
        with _Records(parent_root, parent) as records:
            rows, score_bytes = _load_scores(
                scores_root / OUTPUT_NAME,
                records,
                parent,
            )
    except SemanticLearnabilityError:
        raise
    except Exception as error:
        raise SemanticLearnabilityError(
            "learnability score evidence differs"
        ) from error
    return receipt, rows, score_bytes, receipt_bytes


def _band_counts(size: int) -> dict[str, int]:
    quotient, remainder = divmod(size, len(BANDS))
    return {band: quotient + (index < remainder) for index, band in enumerate(BANDS)}


def _order_key(seed: int, phase: str, band: str, identity: str) -> bytes:
    return hashlib.sha256(f"{seed}:{phase}:{band}:{identity}".encode()).digest()


def _derive_permutation(
    rows: list[dict[str, Any]], phase_ranges: dict[str, range]
) -> tuple[list[int], dict[str, dict[str, Any]]]:
    permutation = []
    evidence = {}
    for phase in PHASES:
        indices = list(phase_ranges[phase])
        ranked = sorted(
            indices,
            key=lambda index: (
                rows[index]["strong_nll_microunits_per_target"],
                -rows[index]["preference_delta_microunits"],
                rows[index]["weak_nll_microunits_per_target"],
                rows[index]["record_sha256"],
            ),
        )
        counts = _band_counts(len(indices))
        phase_evidence: dict[str, Any] = {"sequences": len(indices), "bands": {}}
        offset = 0
        for band in BANDS:
            selected = ranked[offset : offset + counts[band]]
            selected.sort(
                key=lambda index: _order_key(
                    ORDER_SEED,
                    phase,
                    band,
                    rows[index]["record_sha256"],
                )
            )
            deltas = [rows[index]["preference_delta_microunits"] for index in selected]
            strong_nlls = [
                rows[index]["strong_nll_microunits_per_target"] for index in selected
            ]
            phase_evidence["bands"][band] = {
                "sequences": len(selected),
                "minimum_preference_delta_microunits": min(deltas) if deltas else None,
                "maximum_preference_delta_microunits": max(deltas) if deltas else None,
                "minimum_strong_nll_microunits_per_target": (
                    min(strong_nlls) if strong_nlls else None
                ),
                "maximum_strong_nll_microunits_per_target": (
                    max(strong_nlls) if strong_nlls else None
                ),
            }
            permutation.extend(selected)
            offset += counts[band]
        if offset != len(indices):
            raise SemanticLearnabilityError("within-phase band allocation differs")
        evidence[phase] = phase_evidence
    if len(permutation) != len(rows) or len(set(permutation)) != len(rows):
        raise SemanticLearnabilityError("semantic learnability permutation differs")
    return permutation, evidence


def _permutation_sha256(permutation: list[int]) -> str:
    digest = hashlib.sha256()
    for index in permutation:
        digest.update(index.to_bytes(8, "little"))
    return digest.hexdigest()


def _curriculum_receipt(parent: dict[str, Any]) -> dict[str, Any]:
    curriculum = parent["curriculum"]
    sequences = parent["sequences"]
    phase_sequences = curriculum["phase_sequences_emitted"]
    phase_bytes = curriculum["consumed_phase_utf8_bytes"]
    return {
        **curriculum,
        "prefixes": {
            str(sequences): {
                phase: {
                    "tokens": phase_sequences[phase] * parent["sequence_length"],
                    "utf8_bytes": phase_bytes[phase],
                }
                for phase in PHASES
            }
        },
        "required_all_phase_prefixes": [sequences],
    }


def _schedule(
    *,
    parent_root: Path,
    parent: dict[str, Any],
    score_root: Path,
    score_receipt: dict[str, Any],
    score_receipt_bytes: bytes,
    semantic_report: dict[str, Any],
    semantic_inputs: dict[str, bytes],
    semantic_paths: dict[str, Path],
    permutation: list[int],
    phase_evidence: dict[str, dict[str, Any]],
    multiset_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "method": (
            "semantic_phase_locked_strong_nll_primary_progress_secondary_bands_"
            "then_hash_order"
        ),
        "order_seed": ORDER_SEED,
        "parent_stream": {
            "path": str(parent_root.resolve()),
            "receipt_file_sha256": sha256_file(parent_root / "stream_receipt.json"),
            "ordered_stream_identity_sha256": parent["ordered_stream_identity_sha256"],
        },
        "semantic_evidence": {
            name: _descriptor(semantic_paths[name], semantic_inputs[name])
            for name in sorted(semantic_inputs)
        },
        "semantic_progression_receipt_sha256": semantic_report["receipt_sha256"],
        "score_receipt": {
            **_descriptor(score_root / RECEIPT_NAME, score_receipt_bytes),
            "receipt_sha256": score_receipt["receipt_sha256"],
        },
        "scheduler_sha256": sha256_file(Path(__file__)),
        "phase_order": list(PHASES),
        "phase_locked": True,
        "within_phase_evidence": phase_evidence,
        "permutation_sha256": _permutation_sha256(permutation),
        "sequence_multiset_sha256": multiset_sha256,
        "same_tokens_and_boundary_masks": True,
        "same_sequence_multiset": True,
        "only_within_semantic_phase_order_changed": True,
        "semantic_prerequisites_override_model_difficulty": True,
        "limitations": [
            "semantic_taxonomy_coverage_limits_the_proven_prerequisite_scope",
            "model_learnability_may_be_probe_checkpoint_specific",
            "composite_schedule_does_not_authorize_training_or_4b",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def build_semantic_learnability_curriculum(
    parent_root: Path,
    scores_root: Path,
    taxonomy: Path,
    curriculum_receipt: Path,
    annotations: Path,
    progression_report: Path,
    output: Path,
    *,
    workers: int = 1,
) -> dict[str, Any]:
    """Preserve semantic phase membership and pace records only within phases."""

    parent = validate_frozen_stream(parent_root, verify_sources=True)
    phase_ranges = _phase_ranges(parent)
    if output.exists() or output.is_symlink():
        raise SemanticLearnabilityError("composite curriculum output already exists")
    semantic_report, semantic_inputs = _semantic_evidence(
        parent,
        taxonomy=taxonomy,
        curriculum_receipt=curriculum_receipt,
        annotations=annotations,
        progression_report=progression_report,
        workers=workers,
    )
    score_receipt, rows, _, score_receipt_bytes = _score_inputs(
        scores_root, parent_root, parent
    )
    permutation, phase_evidence = _derive_permutation(rows, phase_ranges)
    stage = output.parent / f".{output.name}.partial.{uuid.uuid4().hex}"
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(mode=0o700)
    try:
        with _Records(parent_root, parent) as records:
            multiset_sha256 = _multiset_sha256(records, parent["sequences"])
            shards = _copy_permutation(
                parent_root,
                stage,
                parent,
                records,
                permutation,
            )
        semantic_paths = {
            "taxonomy": taxonomy,
            "curriculum_receipt": curriculum_receipt,
            "annotations": annotations,
            "progression_report": progression_report,
        }
        schedule = _schedule(
            parent_root=parent_root,
            parent=parent,
            score_root=scores_root,
            score_receipt=score_receipt,
            score_receipt_bytes=score_receipt_bytes,
            semantic_report=semantic_report,
            semantic_inputs=semantic_inputs,
            semantic_paths=semantic_paths,
            permutation=permutation,
            phase_evidence=phase_evidence,
            multiset_sha256=multiset_sha256,
        )
        report = {
            **{
                key: value
                for key, value in parent.items()
                if key
                not in {
                    "ordered_stream_identity_sha256",
                    "prefix_utf8_bytes",
                    "shards",
                    "curriculum",
                    "semantic_learnability_curriculum",
                }
            },
            "schema": TOKEN_STREAM_SCHEMA,
            "prefix_utf8_bytes": {
                str(parent["sequences"]): parent["admitted_utf8_bytes"]
            },
            "shards": shards,
            "curriculum": _curriculum_receipt(parent),
            "semantic_learnability_curriculum": schedule,
        }
        report["ordered_stream_identity_sha256"] = canonical_sha256(report)
        (stage / "stream_receipt.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        _validate_composite(
            stage,
            parent_root=parent_root,
            scores_root=scores_root,
            taxonomy=taxonomy,
            curriculum_receipt=curriculum_receipt,
            annotations=annotations,
            progression_report=progression_report,
            workers=workers,
        )
        os.replace(stage, output)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _validate_composite(
    output: Path,
    *,
    parent_root: Path,
    scores_root: Path,
    taxonomy: Path,
    curriculum_receipt: Path,
    annotations: Path,
    progression_report: Path,
    workers: int,
) -> dict[str, Any]:
    parent = validate_frozen_stream(parent_root, verify_sources=True)
    report = validate_frozen_stream(output, verify_sources=True)
    phase_ranges = _phase_ranges(parent)
    semantic_report, semantic_inputs = _semantic_evidence(
        parent,
        taxonomy=taxonomy,
        curriculum_receipt=curriculum_receipt,
        annotations=annotations,
        progression_report=progression_report,
        workers=workers,
    )
    score_receipt, rows, _, score_receipt_bytes = _score_inputs(
        scores_root, parent_root, parent
    )
    permutation, phase_evidence = _derive_permutation(rows, phase_ranges)
    semantic_paths = {
        "taxonomy": taxonomy,
        "curriculum_receipt": curriculum_receipt,
        "annotations": annotations,
        "progression_report": progression_report,
    }
    with (
        _Records(parent_root, parent) as parent_records,
        _Records(output, report) as output_records,
    ):
        multiset_sha256 = _multiset_sha256(parent_records, parent["sequences"])
        if _multiset_sha256(output_records, report["sequences"]) != multiset_sha256:
            raise SemanticLearnabilityError("composite sequence multiset differs")
        for output_index, parent_index in enumerate(permutation):
            if output_records.record(output_index) != parent_records.record(
                parent_index
            ):
                raise SemanticLearnabilityError("composite permutation differs")
    expected = _schedule(
        parent_root=parent_root,
        parent=parent,
        score_root=scores_root,
        score_receipt=score_receipt,
        score_receipt_bytes=score_receipt_bytes,
        semantic_report=semantic_report,
        semantic_inputs=semantic_inputs,
        semantic_paths=semantic_paths,
        permutation=permutation,
        phase_evidence=phase_evidence,
        multiset_sha256=multiset_sha256,
    )
    if (
        report.get("semantic_learnability_curriculum") != expected
        or report["sequences"] != parent["sequences"]
        or report["admitted_utf8_bytes"] != parent["admitted_utf8_bytes"]
        or report["tokenizer_identity_sha256"] != parent["tokenizer_identity_sha256"]
        or report["source_receipts"] != parent["source_receipts"]
        or report["curriculum"] != _curriculum_receipt(parent)
    ):
        raise SemanticLearnabilityError("composite curriculum receipt differs")
    for phase in PHASES:
        output_range = phase_ranges[phase]
        parent_indices = set(permutation[output_range.start : output_range.stop])
        if parent_indices != set(output_range):
            raise SemanticLearnabilityError("semantic phase membership changed")
    return report


def validate_semantic_learnability_curriculum(
    output: Path,
    *,
    parent_root: Path,
    scores_root: Path,
    taxonomy: Path,
    curriculum_receipt: Path,
    annotations: Path,
    progression_report: Path,
    workers: int = 1,
) -> dict[str, Any]:
    """Replay semantic evidence, scores, phases, permutation, and every record."""

    return _validate_composite(
        output,
        parent_root=parent_root,
        scores_root=scores_root,
        taxonomy=taxonomy,
        curriculum_receipt=curriculum_receipt,
        annotations=annotations,
        progression_report=progression_report,
        workers=workers,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--parent-stream", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--curriculum-receipt", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--progression-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers <= 0:
        raise SemanticLearnabilityError("worker count differs")
    if args.command == "build":
        result = build_semantic_learnability_curriculum(
            args.parent_stream,
            args.scores,
            args.taxonomy,
            args.curriculum_receipt,
            args.annotations,
            args.progression_report,
            args.output,
            workers=args.workers,
        )
    else:
        result = validate_semantic_learnability_curriculum(
            args.output,
            parent_root=args.parent_stream,
            scores_root=args.scores,
            taxonomy=args.taxonomy,
            curriculum_receipt=args.curriculum_receipt,
            annotations=args.annotations,
            progression_report=args.progression_report,
            workers=args.workers,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "ordered_stream_identity_sha256": result[
                    "ordered_stream_identity_sha256"
                ],
                "training_authorized": False,
                "four_b_training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
