"""Validate a relocation-safe Sai mixture plan against every evidence artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from sai.data.mixture_plan import (
    SCHEMA as STRUCTURAL_SCHEMA,
)
from sai.data.mixture_plan import (
    DataMixturePlanError,
)
from sai.data.mixture_plan import (
    validate_payload as validate_structural_payload,
)
from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-4b-data-mixture-plan-v3"
EVIDENCE_ROLES = (
    "source_manifest",
    "license_review",
    "quality_audit",
    "selection_policy",
    "decontamination",
    "pedagogical_progression",
)
_RECEIPT_ROLES = {
    "license_review",
    "quality_audit",
    "decontamination",
    "pedagogical_progression",
}
_ALLOWED_STATUSES = {
    "license_review": {"approved", "complete", "passed", "qualified"},
    "quality_audit": {"audit_complete", "complete", "passed", "qualified"},
    "decontamination": {"complete", "passed", "qualified"},
    "pedagogical_progression": {"complete", "passed", "qualified"},
}
_QUALIFICATION_FIELDS = {
    "license_review": "license_approved",
    "quality_audit": "quality_qualified",
    "decontamination": "decontamination_qualified",
    "pedagogical_progression": "progression_qualified",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "sequence_length",
    "sequences_per_update",
    "total_tokens",
    "sources",
    "phases",
    "controls",
    "receipt_sha256",
}
_SOURCE_KEYS = {
    "source_id",
    "source_class",
    "revision",
    "license",
    "domain",
    "evidence",
    "minimum_phase",
    "rehearsal_required",
    "planned_tokens",
}
_EVIDENCE_KEYS = {
    "role",
    "relative_path",
    "file_sha256",
    "schema",
    "status",
    "receipt_sha256",
}
_MAX_EVIDENCE_BYTES = 128 << 20


class DataMixtureEvidenceError(RuntimeError):
    """The final mixture plan or one of its evidence artifacts differs."""


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise DataMixtureEvidenceError(f"{label} fields differ")
    return value


def _hash(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        or not any(bytes.fromhex(value))
    ):
        raise DataMixtureEvidenceError(f"{label} differs")
    return value


def _relative(value: Any) -> PurePosixPath:
    if not isinstance(value, str):
        raise DataMixtureEvidenceError("evidence path differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != value
    ):
        raise DataMixtureEvidenceError("evidence path differs")
    return path


def _read_regular(root: Path, relative: PurePosixPath) -> bytes:
    if root.is_symlink():
        raise DataMixtureEvidenceError("evidence root differs")
    root_resolved = root.resolve(strict=True)
    if not root_resolved.is_dir():
        raise DataMixtureEvidenceError("evidence root differs")
    directory_descriptors: list[int] = []
    try:
        directory_descriptor = os.open(
            root_resolved,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        directory_descriptors.append(directory_descriptor)
        for part in relative.parts[:-1]:
            directory_descriptor = os.open(
                part,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            directory_descriptors.append(directory_descriptor)
        descriptor = os.open(
            relative.parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        for opened in reversed(directory_descriptors):
            os.close(opened)
        raise DataMixtureEvidenceError(
            "evidence artifact is missing or unsafe"
        ) from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_EVIDENCE_BYTES
        ):
            raise DataMixtureEvidenceError("evidence artifact is missing or unsafe")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
        for opened in reversed(directory_descriptors):
            os.close(opened)

    def identity(value):
        return (
            value.st_dev,
            value.st_ino,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
        )

    if len(encoded) != before.st_size or identity(before) != identity(after):
        raise DataMixtureEvidenceError("evidence artifact changed while reading")
    return encoded


def _receipt(
    encoded: bytes,
    descriptor: dict[str, Any],
    role: str,
    *,
    source_manifest_sha256: str,
) -> None:
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataMixtureEvidenceError(f"{role} receipt JSON differs") from error
    if not isinstance(payload, dict):
        raise DataMixtureEvidenceError(f"{role} receipt differs")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    covered = payload.get("covered_source_manifest_sha256s")
    if (
        payload.get("schema") != descriptor["schema"]
        or payload.get("status") != descriptor["status"]
        or payload.get("receipt_sha256") != descriptor["receipt_sha256"]
        or payload.get("receipt_sha256") != canonical_sha256(unsigned)
        or descriptor["status"] not in _ALLOWED_STATUSES[role]
        or payload.get(_QUALIFICATION_FIELDS[role]) is not True
        or not isinstance(covered, list)
        or not covered
        or any(
            not isinstance(item, str) or _SHA256.fullmatch(item) is None
            for item in covered
        )
        or covered != sorted(set(covered))
        or source_manifest_sha256 not in covered
        or payload.get("training_authorized") not in {None, False}
        or payload.get("four_b_training_authorized") not in {None, False}
    ):
        raise DataMixtureEvidenceError(f"{role} receipt differs")


def _descriptor(
    raw: Any,
    *,
    role: str,
    evidence_root: Path,
    seen_paths: dict[str, dict[str, Any]],
    source_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    descriptor = _exact(raw, _EVIDENCE_KEYS, f"{role} evidence")
    if descriptor["role"] != role:
        raise DataMixtureEvidenceError("evidence role differs")
    relative = _relative(descriptor["relative_path"])
    previous = seen_paths.get(str(relative))
    if previous is not None and previous != descriptor:
        raise DataMixtureEvidenceError("shared evidence descriptor differs")
    seen_paths[str(relative)] = descriptor
    file_sha256 = _hash(descriptor["file_sha256"], f"{role} file hash")
    if role in _RECEIPT_ROLES:
        if (
            not isinstance(descriptor["schema"], str)
            or not descriptor["schema"]
            or not isinstance(descriptor["status"], str)
            or descriptor["status"] not in _ALLOWED_STATUSES[role]
        ):
            raise DataMixtureEvidenceError(f"{role} receipt identity differs")
        shadow_hash = _hash(descriptor["receipt_sha256"], f"{role} receipt hash")
    else:
        if any(
            descriptor[field] is not None
            for field in ("schema", "status", "receipt_sha256")
        ):
            raise DataMixtureEvidenceError(f"{role} non-receipt descriptor differs")
        shadow_hash = file_sha256
    encoded = _read_regular(evidence_root, relative)
    if hashlib.sha256(encoded).hexdigest() != file_sha256:
        raise DataMixtureEvidenceError(f"{role} file hash differs")
    if role in _RECEIPT_ROLES:
        if source_manifest_sha256 is None:
            raise DataMixtureEvidenceError("source evidence lineage differs")
        _receipt(
            encoded,
            descriptor,
            role,
            source_manifest_sha256=source_manifest_sha256,
        )
    return descriptor, shadow_hash


def validate_payload(payload: Any, *, evidence_root: Path) -> dict[str, Any]:
    """Validate v3 structure by reopening every evidence artifact."""

    plan = _exact(payload, _TOP_KEYS, "evidenced mixture plan")
    unsigned = {key: value for key, value in plan.items() if key != "receipt_sha256"}
    if (
        plan["schema"] != SCHEMA
        or plan["status"] != "prospective_evidence_bound"
        or plan["training_authorized"] is not False
        or plan["four_b_training_authorized"] is not False
        or _hash(plan["receipt_sha256"], "plan receipt") != canonical_sha256(unsigned)
        or not isinstance(plan["sources"], list)
    ):
        raise DataMixtureEvidenceError("evidenced mixture boundary differs")
    seen_paths: dict[str, dict[str, Any]] = {}
    shadow_sources = []
    for raw_source in plan["sources"]:
        source = _exact(raw_source, _SOURCE_KEYS, "evidenced source")
        evidence = source["evidence"]
        if not isinstance(evidence, dict) or set(evidence) != set(EVIDENCE_ROLES):
            raise DataMixtureEvidenceError("source evidence roles differ")
        hashes = {}
        for role in EVIDENCE_ROLES:
            _, hashes[role] = _descriptor(
                evidence[role],
                role=role,
                evidence_root=evidence_root,
                seen_paths=seen_paths,
                source_manifest_sha256=hashes.get("source_manifest"),
            )
        shadow_sources.append(
            {
                **{key: value for key, value in source.items() if key != "evidence"},
                "source_manifest_sha256": hashes["source_manifest"],
                "license_review_receipt_sha256": hashes["license_review"],
                "quality_audit_receipt_sha256": hashes["quality_audit"],
                "selection_policy_sha256": hashes["selection_policy"],
                "decontamination_receipt_sha256": hashes["decontamination"],
                "pedagogical_progression_receipt_sha256": hashes[
                    "pedagogical_progression"
                ],
            }
        )
    shadow = {
        **{
            key: value
            for key, value in plan.items()
            if key not in {"schema", "sources", "receipt_sha256", "status"}
        },
        "schema": STRUCTURAL_SCHEMA,
        "status": "prospective",
        "sources": shadow_sources,
    }
    shadow["receipt_sha256"] = canonical_sha256(shadow)
    try:
        validate_structural_payload(shadow)
    except DataMixturePlanError as error:
        raise DataMixtureEvidenceError("evidenced mixture structure differs") from error
    return plan


def validate_plan(path: Path, evidence_root: Path) -> dict[str, Any]:
    """Open a v3 plan and replay all evidence relative to an explicit root."""

    try:
        encoded = _read_regular(path.parent, PurePosixPath(path.name))
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataMixtureEvidenceError("evidenced mixture plan differs") from error
    return validate_payload(payload, evidence_root=evidence_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--evidence-root", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = validate_plan(args.plan, args.evidence_root)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "status": "validated_prospective_evidence_bound",
                "receipt_sha256": payload["receipt_sha256"],
                "training_authorized": False,
                "four_b_training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
