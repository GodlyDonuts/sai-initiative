from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from sai.data.authored_curriculum import (
    AuthoredCurriculumError,
    build,
    validate,
)
from sai.data.authored_review_adjudication import adjudicate
from sai.data.authored_review_packet import (
    AuthoredReviewPacketError,
    validate_packet,
)
from sai.data.authored_review_packet import build as build_review_packet
from sai.data.token_stream import canonical_sha256

RUST_REVISION = "1" * 40
PYTHON_REVISION = "2" * 40


def _archive(
    path: Path, members: dict[str, bytes], *, symlink: tuple[str, str] | None = None
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, encoded in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(encoded)
            info.mode = 0o444
            archive.addfile(info, io.BytesIO(encoded))
        if symlink is not None:
            info = tarfile.TarInfo(symlink[0])
            info.type = tarfile.SYMTYPE
            info.linkname = symlink[1]
            archive.addfile(info)


def _sources(tmp_path: Path, *, python_symlink: bool = False) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rust_root = f"book-{RUST_REVISION}"
    rust_summary = ["# Book", "", "[Title](title-page.md)"]
    rust_members = {
        f"{rust_root}/LICENSE-APACHE": b"apache\n",
        f"{rust_root}/LICENSE-MIT": b"mit\n",
        f"{rust_root}/src/title-page.md": b"# Title\n",
    }
    for index in range(1, 111):
        name = f"ch{index:02d}-00.md"
        rust_summary.append(f"- [Chapter {index}]({name})")
        rust_members[f"{rust_root}/src/{name}"] = (
            f"# Chapter {index}\n\n```rust\nfn main() {{}}\n```\n".encode()
        )
    rust_members[f"{rust_root}/src/SUMMARY.md"] = (
        "\n".join(rust_summary) + "\n"
    ).encode()
    rust = tmp_path / "rust.tar.gz"
    _archive(rust, rust_members)

    python_root = f"cpython-{PYTHON_REVISION}"
    tutorial_names = [
        "appetite",
        "interpreter",
        "introduction",
        "controlflow",
        "datastructures",
        "modules",
        "inputoutput",
        "errors",
        "classes",
        "stdlib",
        "stdlib2",
        "venv",
        "whatnow",
        "interactive",
        "floatingpoint",
        "appendix",
    ]
    python_index = "Tutorial\n========\n\n.. toctree::\n   :numbered:\n\n" + "".join(
        f"   {name}.rst\n" for name in tutorial_names
    )
    python_members = {
        f"{python_root}/LICENSE": b"python license\n",
        f"{python_root}/Doc/tutorial/index.rst": python_index.encode(),
    }
    for name in tutorial_names:
        chapter = (
            f"{name.title()}\n{'=' * len(name)}\n\n"
            ".. code-block:: python\n\n   print('ok')\n"
        )
        python_members[f"{python_root}/Doc/tutorial/{name}.rst"] = (chapter).encode()
    python = tmp_path / "python.tar.gz"
    symlink = (
        (
            f"{python_root}/Doc/tutorial/appetite.rst",
            "../../LICENSE",
        )
        if python_symlink
        else None
    )
    if python_symlink:
        del python_members[f"{python_root}/Doc/tutorial/appetite.rst"]
    _archive(python, python_members, symlink=symlink)
    return rust, python


def test_build_preserves_progression_and_holds_training(tmp_path: Path) -> None:
    rust, python = _sources(tmp_path)
    output = tmp_path / "candidate.jsonl"
    receipt = tmp_path / "receipt.json"
    report = build(
        rust_archive=rust,
        rust_revision=RUST_REVISION,
        python_archive=python,
        python_revision=PYTHON_REVISION,
        output=output,
        receipt_output=receipt,
    )
    assert report["training_authorized"] is False
    assert report["four_b_training_authorized"] is False
    assert report["summary"]["rows"] == 127
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["source_path"] == "src/title-page.md"
    assert rows[110]["source_path"] == "src/ch110-00.md"
    assert rows[111]["source_path"] == "Doc/tutorial/appetite.rst"
    assert rows[111]["required_prior_concepts"] == ["programming_foundations"]
    assert ".. code-block:: python" in rows[111]["text"]
    assert output.stat().st_mode & 0o222 == 0
    assert validate(output, receipt)["receipt_sha256"] == report["receipt_sha256"]


def test_rejects_selected_symlink_and_create_overwrite(tmp_path: Path) -> None:
    rust, python = _sources(tmp_path, python_symlink=True)
    with pytest.raises(AuthoredCurriculumError, match="missing or unsafe"):
        build(
            rust_archive=rust,
            rust_revision=RUST_REVISION,
            python_archive=python,
            python_revision=PYTHON_REVISION,
            output=tmp_path / "candidate.jsonl",
            receipt_output=tmp_path / "receipt.json",
        )

    rust, python = _sources(tmp_path / "second")
    output = tmp_path / "exists.jsonl"
    output.write_text("occupied")
    with pytest.raises(AuthoredCurriculumError, match="boundary"):
        build(
            rust_archive=rust,
            rust_revision=RUST_REVISION,
            python_archive=python,
            python_revision=PYTHON_REVISION,
            output=output,
            receipt_output=tmp_path / "new-receipt.json",
        )


def test_validation_rejects_tamper(tmp_path: Path) -> None:
    rust, python = _sources(tmp_path)
    output = tmp_path / "candidate.jsonl"
    receipt = tmp_path / "receipt.json"
    build(
        rust_archive=rust,
        rust_revision=RUST_REVISION,
        python_archive=python,
        python_revision=PYTHON_REVISION,
        output=output,
        receipt_output=receipt,
    )
    output.chmod(0o644)
    output.write_text(
        output.read_text().replace("programming_foundations", "advanced_python", 1)
    )
    with pytest.raises(AuthoredCurriculumError, match="receipt differs"):
        validate(output, receipt)


def test_blind_review_packet_hides_progression_key(tmp_path: Path) -> None:
    rust, python = _sources(tmp_path)
    candidate = tmp_path / "candidate.jsonl"
    candidate_receipt = tmp_path / "candidate-receipt.json"
    build(
        rust_archive=rust,
        rust_revision=RUST_REVISION,
        python_archive=python,
        python_revision=PYTHON_REVISION,
        output=candidate,
        receipt_output=candidate_receipt,
    )
    review = tmp_path / "review.jsonl"
    key = tmp_path / "key.jsonl"
    receipt = tmp_path / "review-receipt.json"
    report = build_review_packet(
        candidate=candidate,
        candidate_receipt=candidate_receipt,
        review_output=review,
        key_output=key,
        receipt_output=receipt,
    )
    review_rows = [json.loads(line) for line in review.read_text().splitlines()]
    key_rows = [json.loads(line) for line in key.read_text().splitlines()]
    assert report["status"] == "awaiting_independent_review"
    assert report["training_authorized"] is False
    assert len(review_rows) == len(key_rows) == 127
    assert [row["review_identity_sha256"] for row in review_rows] == sorted(
        row["review_identity_sha256"] for row in review_rows
    )
    assert "candidate_stage" not in review_rows[0]
    assert "source_path" not in review_rows[0]
    assert "candidate_stage" in key_rows[0]
    assert review_rows[0]["requested_review"]["admission_recommendation"] == "unlabeled"
    assert (
        validate_packet(
            candidate=candidate,
            candidate_receipt=candidate_receipt,
            review_output=review,
            key_output=key,
            receipt_output=receipt,
        )["receipt_sha256"]
        == report["receipt_sha256"]
    )
    with pytest.raises(AuthoredReviewPacketError, match="boundary"):
        build_review_packet(
            candidate=candidate,
            candidate_receipt=candidate_receipt,
            review_output=review,
            key_output=tmp_path / "key-2.jsonl",
            receipt_output=tmp_path / "receipt-2.json",
        )
    review.chmod(0o644)
    review.write_text(review.read_text().replace("unlabeled", "pass", 1))
    with pytest.raises(AuthoredReviewPacketError, match="output differs"):
        validate_packet(
            candidate=candidate,
            candidate_receipt=candidate_receipt,
            review_output=review,
            key_output=key,
            receipt_output=receipt,
        )


def _review_inputs(tmp_path: Path) -> dict[str, Path]:
    rust, python = _sources(tmp_path)
    paths = {
        "candidate": tmp_path / "candidate.jsonl",
        "candidate_receipt": tmp_path / "candidate-receipt.json",
        "review_packet": tmp_path / "review.jsonl",
        "review_key": tmp_path / "key.jsonl",
        "review_packet_receipt": tmp_path / "review-receipt.json",
        "concept_list": tmp_path / "concepts.json",
        "annotation_policy": tmp_path / "policy.json",
        "annotator_identity": tmp_path / "annotator.txt",
        "reviewer_identity": tmp_path / "reviewer.txt",
        "annotator_reviews": tmp_path / "annotator.jsonl",
        "reviewer_reviews": tmp_path / "reviewer.jsonl",
    }
    build(
        rust_archive=rust,
        rust_revision=RUST_REVISION,
        python_archive=python,
        python_revision=PYTHON_REVISION,
        output=paths["candidate"],
        receipt_output=paths["candidate_receipt"],
    )
    build_review_packet(
        candidate=paths["candidate"],
        candidate_receipt=paths["candidate_receipt"],
        review_output=paths["review_packet"],
        key_output=paths["review_key"],
        receipt_output=paths["review_packet_receipt"],
    )
    concepts = {
        "schema": "sai-semantic-prerequisite-concept-list-v1",
        "status": "candidate",
        "concepts": [{"concept_id": "code.symbols"}],
    }
    concept_encoded = json.dumps(concepts, sort_keys=True).encode() + b"\n"
    paths["concept_list"].write_bytes(concept_encoded)
    positive_rule = (
        "explicit_instruction_or_demonstrated_use_with_verifiable_source_span"
    )
    negative_rule = "omit_when_direct_source_evidence_is_absent_or_ambiguous"
    policy = {
        "schema": "sai-semantic-annotation-policy-v2",
        "status": "prospective",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "concept_list_sha256": hashlib.sha256(concept_encoded).hexdigest(),
        "annotation_unit": "document_concept_presence",
        "positive_label_rule": positive_rule,
        "negative_label_rule": negative_rule,
        "evidence_span_contract": {
            "coordinate_system": "unicode_codepoint_half_open",
            "minimum_spans_per_positive_label": 1,
            "minimum_codepoints_per_positive_label": 16,
            "source_hash_required": True,
            "exact_text_match_required": True,
        },
        "confidence_contract": {
            "minimum_confidence_ppm": 800_000,
            "confidence_is_probability_of_policy_compliance": True,
            "below_threshold_action": "omit_and_flag_for_review",
        },
        "prerequisite_contract": {
            "same_document_exposure_counts_as_prior": False,
            "phase_source": "bound_curriculum_receipt_only",
            "unmet_prerequisite_action": "record_violation_and_reject_progression",
        },
        "review_contract": {
            "blind_independent_review": True,
            "disagreement_unit": "unordered_unique_concept_identity_set_per_document",
            "minimum_reviewed_documents": 100,
            "maximum_disagreement_ppm": 50_000,
            "adjudication_may_not_change_measured_disagreement": True,
        },
    }
    policy["receipt_sha256"] = canonical_sha256(policy)
    paths["annotation_policy"].write_text(json.dumps(policy, sort_keys=True))
    paths["annotator_identity"].write_text("annotator-one")
    paths["reviewer_identity"].write_text("reviewer-two")
    packet = [
        json.loads(line) for line in paths["review_packet"].read_text().splitlines()
    ]
    rows = []
    for source in packet:
        span = source["text"][0]
        rows.append(
            {
                "schema": "sai-authored-curriculum-completed-review-row-v1",
                "review_identity_sha256": source["review_identity_sha256"],
                "instructional_quality_ppm": 900_000,
                "assumed_prior_concepts": [],
                "taught_concepts": [
                    {
                        "concept_id": "code.symbols",
                        "confidence_ppm": 900_000,
                        "evidence_spans": [
                            {
                                "start": 0,
                                "end": 1,
                                "text_sha256": hashlib.sha256(
                                    span.encode()
                                ).hexdigest(),
                            }
                        ],
                    }
                ],
                "defects": [],
                "admission_recommendation": "admit",
            }
        )
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    paths["annotator_reviews"].write_text(encoded)
    paths["reviewer_reviews"].write_text(encoded)
    return paths


def test_semantic_review_adjudication_preserves_measured_disagreement(
    tmp_path: Path,
) -> None:
    paths = _review_inputs(tmp_path)
    passed = adjudicate(**paths, output=tmp_path / "passed.json")
    assert passed["status"] == "passed"
    assert passed["audit_qualified"] is True
    assert passed["training_authorized"] is False
    assert set(passed["observed_disagreement_ppm"].values()) == {0}

    reviewer_rows = [
        json.loads(line) for line in paths["reviewer_reviews"].read_text().splitlines()
    ]
    for row in reviewer_rows[:7]:
        row["taught_concepts"] = []
    paths["reviewer_reviews"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in reviewer_rows)
    )
    failed = adjudicate(**paths, output=tmp_path / "failed.json")
    assert failed["status"] == "failed"
    assert failed["audit_qualified"] is False
    assert failed["observed_disagreement_ppm"]["taught_concepts"] > 50_000
