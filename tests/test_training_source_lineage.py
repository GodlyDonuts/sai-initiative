import json
from pathlib import Path

import pytest

from sai.data.token_stream import canonical_sha256, sha256_file
from sai.evaluation.training_source_lineage import (
    TrainingSourceLineageError,
    validate_training_source_lineage,
)

PARENT_SHA = "1" * 64


def _write_receipt(path: Path, payload: dict) -> dict:
    payload = dict(payload)
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def _derived_fixture(tmp_path: Path) -> tuple[dict, Path, str]:
    curriculum_output = tmp_path / "curriculum.jsonl"
    curriculum_output.write_text("curriculum\n")
    curriculum_receipt_path = tmp_path / "curriculum.receipt.json"
    curriculum = _write_receipt(
        curriculum_receipt_path,
        {
            "schema": "sai-curriculum-order-receipt-v1",
            "status": "qualified",
            "curriculum_qualified": True,
            "source": {"sha256": PARENT_SHA},
            "output": {"sha256": sha256_file(curriculum_output)},
        },
    )
    train = tmp_path / "train.jsonl"
    train.write_text("train\n")
    split_path = tmp_path / "split.receipt.json"
    _write_receipt(
        split_path,
        {
            "schema": "sai-curriculum-train-development-split-v1",
            "status": "qualified",
            "split_qualified": True,
            "checks": {
                "all_curriculum_documents_emitted_once": True,
                "both_populations_have_every_phase": True,
                "exact_identity_assignment_disjoint": True,
                "train_progression_qualified": True,
            },
            "source_curriculum": {
                "receipt_path": str(curriculum_receipt_path.resolve()),
                "receipt_bytes": curriculum_receipt_path.stat().st_size,
                "receipt_file_sha256": sha256_file(curriculum_receipt_path),
                "receipt_sha256": curriculum["receipt_sha256"],
                "output_sha256": sha256_file(curriculum_output),
            },
            "train": {
                "path": str(train.resolve()),
                "bytes": train.stat().st_size,
                "sha256": sha256_file(train),
            },
        },
    )
    stream_source = {
        "order": 0,
        "path": str(train.resolve()),
        "bytes": train.stat().st_size,
        "sha256": sha256_file(train),
    }
    return stream_source, split_path, sha256_file(split_path)


def test_direct_and_qualified_curriculum_lineage_pass(tmp_path: Path) -> None:
    direct = {"order": 0, "path": "/source", "bytes": 1, "sha256": PARENT_SHA}
    direct_result = validate_training_source_lineage(
        stream_source=direct,
        benchmark_training_source_sha256=PARENT_SHA,
    )
    assert direct_result["lineage"]["method"] == "direct_decontaminated_source"

    stream_source, split, split_sha = _derived_fixture(tmp_path)
    derived = validate_training_source_lineage(
        stream_source=stream_source,
        benchmark_training_source_sha256=PARENT_SHA,
        split_receipt=split,
        split_receipt_file_sha256=split_sha,
    )
    assert derived["lineage"]["method"] == "qualified_curriculum_train_split"
    assert len(derived["source_lineage_sha256"]) == 64


@pytest.mark.parametrize("tamper", ["parent", "train", "split", "curriculum"])
def test_derived_lineage_tamper_fails(tmp_path: Path, tamper: str) -> None:
    stream_source, split, split_sha = _derived_fixture(tmp_path)
    parent = PARENT_SHA
    if tamper == "parent":
        parent = "2" * 64
    elif tamper == "train":
        stream_source["sha256"] = "3" * 64
    elif tamper == "split":
        payload = json.loads(split.read_text())
        payload["status"] = "failed"
        split.write_text(json.dumps(payload))
    else:
        split_payload = json.loads(split.read_text())
        curriculum = Path(split_payload["source_curriculum"]["receipt_path"])
        payload = json.loads(curriculum.read_text())
        payload["status"] = "failed"
        curriculum.write_text(json.dumps(payload))
    with pytest.raises(TrainingSourceLineageError):
        validate_training_source_lineage(
            stream_source=stream_source,
            benchmark_training_source_sha256=parent,
            split_receipt=split,
            split_receipt_file_sha256=split_sha,
        )
