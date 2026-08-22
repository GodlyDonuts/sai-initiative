from __future__ import annotations

import json
from pathlib import Path

import pytest

import sai.data.authored_review_model_compare as compare
from sai.data.authored_review_model_compare import (
    AuthoredReviewModelCompareError,
    run,
    validate,
)

ROOT = Path(__file__).parents[1]
JOB = ROOT / "jobs" / "sai-compare-authored-model-reviews-cpu.sbatch"


def _draft(identity: str, concept: str, quality: int, recommendation: str) -> dict:
    return {
        "schema": "sai-authored-curriculum-quoted-review-draft-row-v1",
        "review_identity_sha256": identity,
        "instructional_quality_ppm": quality,
        "assumed_prior_concepts": [],
        "taught_concepts": [
            {
                "concept_id": concept,
                "confidence_ppm": 900_000,
                "evidence_quotes": ["sixteen exact source codepoints"],
            }
        ],
        "defects": [],
        "admission_recommendation": recommendation,
    }


def _write(root: Path, rows: list[dict]) -> None:
    root.mkdir()
    root.joinpath("draft.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )


def test_cross_family_comparison_prioritizes_disagreement_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identities = [f"{index + 1:064x}" for index in range(127)]
    qwen = [
        _draft(identity, "code.literal", 900_000, "admit") for identity in identities
    ]
    smol = list(qwen)
    smol = [dict(row) for row in smol]
    smol[3] = _draft(identities[3], "code.symbols", 700_000, "revise")
    qwen_root, smol_root = tmp_path / "qwen", tmp_path / "smol"
    _write(qwen_root, qwen)
    _write(smol_root, smol)
    monkeypatch.setattr(
        compare,
        "validate_result",
        lambda **kwargs: {
            "receipt_sha256": (
                "1" * 64 if kwargs["reviewer"] == "qwen35_9b" else "2" * 64
            )
        },
    )
    common = {
        "qwen_model_root": tmp_path / "qm",
        "qwen_manifest": tmp_path / "qmanifest",
        "qwen_restoration_receipt": tmp_path / "qr",
        "qwen_review_root": qwen_root,
        "smol_model_root": tmp_path / "sm",
        "smol_manifest": tmp_path / "smanifest",
        "smol_restoration_receipt": tmp_path / "sr",
        "smol_review_root": smol_root,
        "review_packet": ROOT
        / "artifacts"
        / "authored-curriculum-sources-r1"
        / "authored-curriculum-blind-review.jsonl",
        "review_packet_receipt": ROOT
        / "artifacts"
        / "authored-curriculum-sources-r1"
        / "authored-curriculum-review-receipt.json",
        "expected_review_packet_sha256": "a" * 64,
        "expected_review_packet_receipt_sha256": "b" * 64,
        "concept_list": ROOT
        / "docs"
        / "SAI_SEMANTIC_PREREQUISITE_CONCEPTS_CANDIDATE.json",
        "annotation_policy": ROOT / "docs" / "SAI_SEMANTIC_ANNOTATION_POLICY.json",
        "output": tmp_path / "comparison.json",
    }
    payload = run(**common)
    assert payload["status"] == "candidate_model_triage_complete"
    assert payload["disagreement_documents"] == 1
    assert payload["review_priority"][0]["index"] == 3
    assert payload["audit_qualified"] is False
    assert validate(**common) == payload
    common["output"].chmod(0o644)
    common["output"].write_text(common["output"].read_text().replace("127", "126", 1))
    with pytest.raises(AuthoredReviewModelCompareError, match="receipt differs"):
        validate(**common)


def test_model_compare_job_is_cpu_only_and_nonqualifying() -> None:
    script = JOB.read_text()
    assert "#SBATCH --gres" not in script
    assert "#SBATCH --no-requeue" in script
    assert 'export PYTHONPATH="$SAI_ROOT/src"' in script
    assert "sai.data.authored_review_model_compare" in script
    assert "torchrun" not in script
