from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

from sai.data.mixture_evidence import (
    SCHEMA,
    DataMixtureEvidenceError,
    validate_payload,
    validate_plan,
)
from sai.data.token_stream import canonical_sha256


def _receipt(
    path: Path,
    *,
    schema: str,
    status: str,
    source_id: str,
    source_manifest_sha256: str,
) -> dict:
    qualification_fields = {
        "test-license_review-v1": "license_approved",
        "test-quality_audit-v1": "quality_qualified",
        "test-decontamination-v1": "decontamination_qualified",
        "test-pedagogical_progression-v1": "progression_qualified",
    }
    payload = {
        "schema": schema,
        "status": status,
        "training_authorized": False,
        "four_b_training_authorized": False,
        "source_id": source_id,
        "covered_source_manifest_sha256s": [source_manifest_sha256],
        qualification_fields[schema]: True,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return payload


def _descriptor(
    path: Path,
    root: Path,
    *,
    role: str,
    payload: dict | None = None,
) -> dict:
    return {
        "role": role,
        "relative_path": str(path.relative_to(root)),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema": payload["schema"] if payload else None,
        "status": payload["status"] if payload else None,
        "receipt_sha256": payload["receipt_sha256"] if payload else None,
    }


def _source(
    root: Path,
    index: int,
    *,
    source_class: str,
    domain: str,
    planned_tokens: int,
    minimum_phase: str = "grounding",
    rehearsal_required: bool = True,
) -> dict:
    source_id = f"source-{index}"
    source_root = root / source_id
    source_root.mkdir()
    manifest = source_root / "manifest.txt"
    manifest.write_text(f"{source_id} exact source member\n")
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    policy = source_root / "selection-policy.json"
    policy.write_text(json.dumps({"source_id": source_id, "rule": "frozen"}) + "\n")
    evidence = {
        "source_manifest": _descriptor(manifest, root, role="source_manifest"),
        "selection_policy": _descriptor(policy, root, role="selection_policy"),
    }
    for role, status in (
        ("license_review", "approved"),
        ("quality_audit", "audit_complete"),
        ("decontamination", "qualified"),
        ("pedagogical_progression", "passed"),
    ):
        path = source_root / f"{role}.json"
        payload = _receipt(
            path,
            schema=f"test-{role}-v1",
            status=status,
            source_id=source_id,
            source_manifest_sha256=manifest_sha256,
        )
        evidence[role] = _descriptor(path, root, role=role, payload=payload)
    return {
        "source_id": source_id,
        "source_class": source_class,
        "revision": f"{index + 1:040x}",
        "license": "ODC-BY-1.0",
        "domain": domain,
        "evidence": evidence,
        "minimum_phase": minimum_phase,
        "rehearsal_required": rehearsal_required,
        "planned_tokens": planned_tokens,
    }


def _plan(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    sources = [
        _source(
            root,
            0,
            source_class="educational_web",
            domain="english",
            planned_tokens=64,
        ),
        _source(root, 1, source_class="code", domain="code", planned_tokens=32),
        _source(
            root,
            2,
            source_class="mathematics",
            domain="math",
            planned_tokens=24,
        ),
        _source(
            root,
            3,
            source_class="science_technical",
            domain="science",
            planned_tokens=32,
        ),
        _source(
            root,
            4,
            source_class="science_technical",
            domain="technical",
            planned_tokens=8,
            minimum_phase="reasoning",
            rehearsal_required=False,
        ),
    ]
    allocations = (
        (16, 8, 8, 8, 0),
        (16, 8, 8, 8, 0),
        (16, 8, 4, 8, 4),
        (16, 8, 4, 8, 4),
    )
    phases = []
    cumulative = 0
    for index, (name, allocation) in enumerate(
        zip(
            ("grounding", "integration", "reasoning", "specialization"),
            allocations,
            strict=True,
        )
    ):
        tokens = sum(allocation)
        cumulative += tokens
        phases.append(
            {
                "phase": name,
                "index": index,
                "tokens": tokens,
                "cumulative_tokens": cumulative,
                "by_source": {
                    source["source_id"]: value
                    for source, value in zip(sources, allocation, strict=True)
                },
            }
        )
    payload = {
        "schema": SCHEMA,
        "status": "prospective_evidence_bound",
        "training_authorized": False,
        "four_b_training_authorized": False,
        "sequence_length": 8,
        "sequences_per_update": 1,
        "total_tokens": 160,
        "sources": sources,
        "phases": phases,
        "controls": {
            "same_sequence_multiset_order_control": True,
            "tokenizer_factor_isolated": True,
            "architecture_factor_isolated": True,
            "terminal_benchmarks_used_for_tuning": False,
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _resign(payload: dict) -> None:
    payload["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_sha256"}
    )


def test_reopens_every_mixture_evidence_artifact_and_relocates(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    payload = _plan(root)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(payload, sort_keys=True) + "\n")
    assert validate_payload(payload, evidence_root=root) == payload
    assert validate_plan(plan, root) == payload

    relocated = tmp_path / "relocated"
    shutil.copytree(root, relocated)
    assert validate_plan(plan, relocated) == payload


def test_rejects_tamper_missing_and_re_signed_bare_hash(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    payload = _plan(root)
    descriptor = payload["sources"][0]["evidence"]["quality_audit"]
    evidence_path = root / descriptor["relative_path"]
    evidence_path.write_text(
        evidence_path.read_text().replace("audit_complete", "passed")
    )
    with pytest.raises(DataMixtureEvidenceError, match="file hash"):
        validate_payload(payload, evidence_root=root)
    _receipt(
        evidence_path,
        schema=descriptor["schema"],
        status=descriptor["status"],
        source_id="source-0",
        source_manifest_sha256=payload["sources"][0]["evidence"]["source_manifest"][
            "file_sha256"
        ],
    )

    missing = deepcopy(payload)
    missing["sources"][1]["evidence"]["source_manifest"][
        "relative_path"
    ] = "source-1/missing.txt"
    _resign(missing)
    with pytest.raises(DataMixtureEvidenceError, match="missing or unsafe"):
        validate_payload(missing, evidence_root=root)


def test_rejects_unsafe_and_receipt_self_hash_drift(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    payload = _plan(root)
    descriptor = payload["sources"][2]["evidence"]["decontamination"]
    path = root / descriptor["relative_path"]
    content = json.loads(path.read_text())
    content["source_id"] = "other-source"
    path.write_text(json.dumps(content, sort_keys=True) + "\n")
    descriptor["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _resign(payload)
    with pytest.raises(DataMixtureEvidenceError, match="receipt differs"):
        validate_payload(payload, evidence_root=root)

    target = root / "source-0" / "manifest.txt"
    link = root / "source-0" / "manifest-link.txt"
    link.symlink_to(target)
    payload = _plan(tmp_path / "second")
    payload["sources"][0]["evidence"]["source_manifest"]["relative_path"] = str(
        link.relative_to(root)
    )
    _resign(payload)
    with pytest.raises(DataMixtureEvidenceError, match="missing or unsafe"):
        validate_payload(payload, evidence_root=root)


def test_rejects_re_signed_receipt_without_role_qualification(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    payload = _plan(root)
    descriptor = payload["sources"][3]["evidence"]["quality_audit"]
    path = root / descriptor["relative_path"]
    receipt = json.loads(path.read_text())
    receipt["quality_qualified"] = False
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    descriptor["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor["receipt_sha256"] = receipt["receipt_sha256"]
    _resign(payload)

    with pytest.raises(DataMixtureEvidenceError, match="quality_audit receipt differs"):
        validate_payload(payload, evidence_root=root)


def test_rejects_positive_receipt_bound_to_a_different_source(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    payload = _plan(root)
    descriptor = payload["sources"][4]["evidence"]["license_review"]
    path = root / descriptor["relative_path"]
    receipt = json.loads(path.read_text())
    receipt["covered_source_manifest_sha256s"] = [
        payload["sources"][0]["evidence"]["source_manifest"]["file_sha256"]
    ]
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    descriptor["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor["receipt_sha256"] = receipt["receipt_sha256"]
    _resign(payload)

    with pytest.raises(
        DataMixtureEvidenceError, match="license_review receipt differs"
    ):
        validate_payload(payload, evidence_root=root)


def test_accepts_shared_receipt_only_when_it_covers_both_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    payload = _plan(root)
    descriptor = payload["sources"][0]["evidence"]["license_review"]
    path = root / descriptor["relative_path"]
    receipt = json.loads(path.read_text())
    receipt["source_id"] = "shared-license-review"
    receipt["covered_source_manifest_sha256s"] = sorted(
        source["evidence"]["source_manifest"]["file_sha256"]
        for source in payload["sources"][:2]
    )
    receipt["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    descriptor["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    descriptor["receipt_sha256"] = receipt["receipt_sha256"]
    payload["sources"][1]["evidence"]["license_review"] = deepcopy(descriptor)
    _resign(payload)

    assert validate_payload(payload, evidence_root=root) == payload


def test_rejects_symlinked_intermediate_evidence_directory(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    payload = _plan(root)
    source = root / "source-0"
    relocated = root / "relocated-source-0"
    source.rename(relocated)
    source.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(DataMixtureEvidenceError, match="missing or unsafe"):
        validate_payload(payload, evidence_root=root)
