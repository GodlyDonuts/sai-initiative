from __future__ import annotations

import json
from pathlib import Path

import pytest

import sai.data.semantic_learnability_curriculum as composite
from sai.data.curriculum import PHASES
from sai.data.curriculum_control import _record_sha256, _Records
from sai.data.learnability_curriculum import SCORE_SCHEMA
from sai.data.learnability_score import OUTPUT_NAME, RECEIPT_NAME
from sai.data.semantic_learnability_curriculum import (
    SemanticLearnabilityError,
    build_semantic_learnability_curriculum,
    validate_semantic_learnability_curriculum,
)
from sai.data.token_stream import (
    canonical_sha256,
    causal_loss_mask_from_start_bits,
    freeze,
    sha256_file,
)
from tests.test_learnability_score import _receipt as _write_score_receipt
from tests.test_token_stream import CharacterTokenizer, document, write_documents


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path, Path, Path, dict]:
    documents = []
    domains = ("english", "math", "code", "science", "technical")
    for index in range(4):
        row = document(index, chr(65 + index) * 7 + chr(97 + index) * 8)
        row["source"]["domain"] = domains[index % len(domains)]
        documents.append(row)
    source = write_documents(tmp_path / "semantic.jsonl", documents)
    curriculum_receipt = tmp_path / "curriculum.receipt.json"
    curriculum_receipt.write_text('{"qualified":true}\n')
    phase_documents = [(phase, 1) for phase in PHASES]
    phase_sequences = [(phase, 4) for phase in PHASES]
    parent_root = tmp_path / "parent"
    parent = freeze(
        CharacterTokenizer(),
        [source],
        parent_root,
        tokenizer_identity_sha256="1" * 64,
        sequence_length=4,
        prefix_sequences={16},
        sequences_per_shard=2,
        source_qualification_sha256=sha256_file(curriculum_receipt),
        curriculum_phases=phase_documents,
        curriculum_phase_sequence_targets=phase_sequences,
        required_phase_complete_prefixes={16},
    )
    scores_root = tmp_path / "scores"
    scores_root.mkdir()
    rows = []
    with _Records(parent_root, parent) as records:
        for index in range(parent["sequences"]):
            tokens, starts = records.record(index)
            local = index % 4
            strong = (1_200_000, 800_000, 1_000_000, 900_000)[local]
            delta = (20_000, 50_000, 200_000, 150_000)[local]
            weak = strong + delta
            rows.append(
                {
                    "schema": SCORE_SCHEMA,
                    "sequence_index": index,
                    "record_sha256": _record_sha256(tokens, starts).hex(),
                    "target_count": sum(
                        causal_loss_mask_from_start_bits(
                            starts, parent["sequence_length"]
                        )
                    ),
                    "weak_nll_microunits_per_target": weak,
                    "strong_nll_microunits_per_target": strong,
                    "preference_delta_microunits": delta,
                }
            )
    score_receipt = _write_score_receipt(scores_root, rows)
    score_receipt["target_stream"] = {
        "path": str(parent_root.resolve()),
        "receipt_file_sha256": sha256_file(parent_root / "stream_receipt.json"),
        "ordered_stream_identity_sha256": parent["ordered_stream_identity_sha256"],
        "source_manifest_sha256": parent["source_manifest_sha256"],
        "tokenizer_identity_sha256": parent["tokenizer_identity_sha256"],
        "sequences": parent["sequences"],
        "sequence_length": parent["sequence_length"],
    }
    score_receipt["probe_training_stream"]["tokenizer_identity_sha256"] = parent[
        "tokenizer_identity_sha256"
    ]
    score_receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in score_receipt.items() if key != "receipt_sha256"}
    )
    (scores_root / RECEIPT_NAME).write_text(
        json.dumps(score_receipt, sort_keys=True) + "\n"
    )
    assert (scores_root / OUTPUT_NAME).is_file()

    taxonomy = tmp_path / "taxonomy.json"
    annotations = tmp_path / "annotations.jsonl"
    progression = tmp_path / "progression.json"
    taxonomy.write_text('{"taxonomy":"frozen"}\n')
    annotations.write_text('{"annotation":"frozen"}\n')
    report = {
        "schema": "sai-semantic-prerequisite-progression-report-v3",
        "status": "qualified",
        "progression_qualified": True,
        "training_authorized": False,
        "four_b_training_authorized": False,
        "violations": [],
        "premature_exposure_violations": [],
        "concept_density_violations": [],
        "phase_coverage_violations": [],
        "missing_concepts": [],
        "curriculum_lineage": {
            "curriculum_receipt_sha256": "a" * 64,
            "curriculum_output_bytes": source.stat().st_size,
            "curriculum_output_sha256": sha256_file(source),
            "annotations_path": str(annotations.resolve()),
            "annotations_bytes": annotations.stat().st_size,
            "annotations_file_sha256": sha256_file(annotations),
        },
    }
    report["receipt_sha256"] = canonical_sha256(report)
    progression.write_text(json.dumps(report, sort_keys=True) + "\n")
    monkeypatch.setattr(
        composite,
        "replay_curriculum_annotation_files",
        lambda *_args, **_kwargs: report,
    )
    return (
        parent_root,
        scores_root,
        taxonomy,
        curriculum_receipt,
        annotations,
        progression,
        parent,
    )


def test_composite_preserves_semantic_phase_membership_and_exact_multiset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, scores, taxonomy, receipt, annotations, progression, parent_report = (
        _fixture(tmp_path, monkeypatch)
    )
    output = tmp_path / "composite"
    report = build_semantic_learnability_curriculum(
        parent,
        scores,
        taxonomy,
        receipt,
        annotations,
        progression,
        output,
    )
    schedule = report["semantic_learnability_curriculum"]
    assert schedule["phase_locked"] is True
    assert schedule["semantic_prerequisites_override_model_difficulty"] is True
    assert report["curriculum"]["phase_sequences_emitted"] == {
        phase: 4 for phase in PHASES
    }
    with _Records(parent, parent_report) as left, _Records(output, report) as right:
        for phase_index, _phase in enumerate(PHASES):
            start = phase_index * 4
            stop = start + 4
            assert right.record(start) == left.record(start + 1)
            assert right.record(start + 1) == left.record(start + 3)
            assert right.record(start + 2) == left.record(start + 2)
            assert right.record(start + 3) == left.record(start)
            parent_hashes = {
                _record_sha256(*left.record(index)) for index in range(start, stop)
            }
            output_hashes = {
                _record_sha256(*right.record(index)) for index in range(start, stop)
            }
            assert output_hashes == parent_hashes
    assert (
        validate_semantic_learnability_curriculum(
            output,
            parent_root=parent,
            scores_root=scores,
            taxonomy=taxonomy,
            curriculum_receipt=receipt,
            annotations=annotations,
            progression_report=progression,
        )
        == report
    )


def test_composite_rejects_semantic_report_or_score_target_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent, scores, taxonomy, receipt, annotations, progression, _ = _fixture(
        tmp_path, monkeypatch
    )
    payload = json.loads(progression.read_text())
    payload["progression_qualified"] = False
    progression.write_text(json.dumps(payload) + "\n")
    with pytest.raises(SemanticLearnabilityError, match="semantic progression"):
        build_semantic_learnability_curriculum(
            parent,
            scores,
            taxonomy,
            receipt,
            annotations,
            progression,
            tmp_path / "rejected",
        )

    progression.write_text(
        json.dumps(
            composite.replay_curriculum_annotation_files(
                taxonomy, receipt, annotations
            ),
            sort_keys=True,
        )
        + "\n"
    )
    score_receipt = json.loads((scores / RECEIPT_NAME).read_text())
    score_receipt["target_stream"]["ordered_stream_identity_sha256"] = "f" * 64
    score_receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in score_receipt.items() if key != "receipt_sha256"}
    )
    (scores / RECEIPT_NAME).write_text(json.dumps(score_receipt) + "\n")
    with pytest.raises(SemanticLearnabilityError, match="target stream"):
        build_semantic_learnability_curriculum(
            parent,
            scores,
            taxonomy,
            receipt,
            annotations,
            progression,
            tmp_path / "rejected-score",
        )
