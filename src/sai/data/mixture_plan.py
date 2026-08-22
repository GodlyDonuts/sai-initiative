"""Validate a prospective, data-first Sai 4B source-mixture plan."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from sai.data.token_stream import ALLOWED_DOMAINS, canonical_sha256

SCHEMA = "sai-4b-data-mixture-plan-v2"
PHASES = ("grounding", "integration", "reasoning", "specialization")
SOURCE_CLASSES = {
    "educational_web",
    "diverse_open_corpus",
    "mathematics",
    "code",
    "foundational_reference",
    "science_technical",
}
_SOURCE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{1,63}")
_PLACEHOLDER_LICENSES = {"", "none", "pending", "tbd", "unknown", "unverified"}
_MAX_PLAN_BYTES = 8 << 20
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
    "source_manifest_sha256",
    "license_review_receipt_sha256",
    "quality_audit_receipt_sha256",
    "selection_policy_sha256",
    "decontamination_receipt_sha256",
    "pedagogical_progression_receipt_sha256",
    "minimum_phase",
    "rehearsal_required",
    "planned_tokens",
}
_PHASE_KEYS = {"phase", "index", "tokens", "cumulative_tokens", "by_source"}
_CONTROL_KEYS = {
    "same_sequence_multiset_order_control",
    "tokenizer_factor_isolated",
    "architecture_factor_isolated",
    "terminal_benchmarks_used_for_tuning",
}


class DataMixturePlanError(RuntimeError):
    """The prospective source mixture is unsafe, inconsistent, or mutable."""


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise DataMixturePlanError(f"{label} fields differ")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DataMixturePlanError(f"{label} differs")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataMixturePlanError(f"{label} differs")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DataMixturePlanError(f"{label} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise DataMixturePlanError(f"{label} differs") from error
    if not any(bytes.fromhex(value)):
        raise DataMixturePlanError(f"{label} is a placeholder")
    return value


def _revision(value: Any) -> str:
    if not isinstance(value, str) or len(value) not in {40, 64}:
        raise DataMixturePlanError("source revision differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise DataMixturePlanError("source revision differs") from error
    if value != value.lower() or not any(bytes.fromhex(value)):
        raise DataMixturePlanError("source revision differs")
    return value


def validate_payload(payload: Any) -> dict[str, Any]:
    """Validate and return one canonical prospective mixture payload."""

    plan = _exact_keys(payload, _TOP_KEYS, "data mixture plan")
    if (
        plan["schema"] != SCHEMA
        or plan["status"] != "prospective"
        or plan["training_authorized"] is not False
        or plan["four_b_training_authorized"] is not False
    ):
        raise DataMixturePlanError("data mixture authorization boundary differs")

    receipt_sha256 = _sha256(plan["receipt_sha256"], "plan receipt hash")
    unsigned = {key: value for key, value in plan.items() if key != "receipt_sha256"}
    if receipt_sha256 != canonical_sha256(unsigned):
        raise DataMixturePlanError("plan receipt hash differs")

    sequence_length = _positive_int(plan["sequence_length"], "sequence length")
    sequences_per_update = _positive_int(
        plan["sequences_per_update"], "sequences per update"
    )
    total_tokens = _positive_int(plan["total_tokens"], "total tokens")
    if total_tokens % sequence_length:
        raise DataMixturePlanError("total token budget is not sequence aligned")

    raw_sources = plan["sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) < 2:
        raise DataMixturePlanError("source mixture lacks independent sources")
    sources: dict[str, dict[str, Any]] = {}
    domains: set[str] = set()
    planned_source_tokens = 0
    for raw_source in raw_sources:
        source = _exact_keys(raw_source, _SOURCE_KEYS, "data source")
        source_id = source["source_id"]
        if (
            not isinstance(source_id, str)
            or _SOURCE_ID.fullmatch(source_id) is None
            or source_id in sources
        ):
            raise DataMixturePlanError("source identity differs or is duplicated")
        if source["source_class"] not in SOURCE_CLASSES:
            raise DataMixturePlanError("source class differs")
        _revision(source["revision"])
        if (
            not isinstance(source["license"], str)
            or source["license"].strip().casefold() in _PLACEHOLDER_LICENSES
        ):
            raise DataMixturePlanError("source license differs")
        if source["domain"] not in ALLOWED_DOMAINS:
            raise DataMixturePlanError("source domain differs")
        domains.add(source["domain"])
        for field in (
            "source_manifest_sha256",
            "license_review_receipt_sha256",
            "quality_audit_receipt_sha256",
            "selection_policy_sha256",
            "decontamination_receipt_sha256",
            "pedagogical_progression_receipt_sha256",
        ):
            _sha256(source[field], field)
        if source["minimum_phase"] not in PHASES:
            raise DataMixturePlanError("source minimum phase differs")
        if not isinstance(source["rehearsal_required"], bool):
            raise DataMixturePlanError("source rehearsal policy differs")
        planned = _positive_int(source["planned_tokens"], "source token budget")
        planned_source_tokens += planned
        sources[source_id] = source

    if domains != ALLOWED_DOMAINS:
        raise DataMixturePlanError("source mixture does not cover every Sai domain")
    if planned_source_tokens != total_tokens:
        raise DataMixturePlanError("source token budgets do not equal total tokens")

    raw_phases = plan["phases"]
    if not isinstance(raw_phases, list) or len(raw_phases) != len(PHASES):
        raise DataMixturePlanError("curriculum phase count differs")
    emitted_by_source = dict.fromkeys(sources, 0)
    cumulative_tokens = 0
    for index, (expected_phase, raw_phase) in enumerate(
        zip(PHASES, raw_phases, strict=True)
    ):
        phase = _exact_keys(raw_phase, _PHASE_KEYS, "curriculum phase")
        if phase["phase"] != expected_phase or phase["index"] != index:
            raise DataMixturePlanError("curriculum phase order differs")
        tokens = _positive_int(phase["tokens"], "phase token budget")
        if tokens % sequence_length:
            raise DataMixturePlanError("phase token budget is not sequence aligned")
        cumulative_tokens += tokens
        if phase["cumulative_tokens"] != cumulative_tokens:
            raise DataMixturePlanError("phase cumulative token budget differs")
        if (
            index < len(PHASES) - 1
            and (cumulative_tokens // sequence_length) % sequences_per_update
        ):
            raise DataMixturePlanError("phase boundary splits an optimizer update")
        by_source = phase["by_source"]
        if not isinstance(by_source, dict) or set(by_source) != set(sources):
            raise DataMixturePlanError("phase source membership differs")
        phase_sum = 0
        for source_id, source in sources.items():
            allocated = _nonnegative_int(
                by_source[source_id], "phase source token budget"
            )
            phase_sum += allocated
            emitted_by_source[source_id] += allocated
            minimum_index = PHASES.index(source["minimum_phase"])
            if index < minimum_index and allocated:
                raise DataMixturePlanError("source appears before its minimum phase")
            if (
                source["rehearsal_required"]
                and index >= minimum_index
                and not allocated
            ):
                raise DataMixturePlanError("required source rehearsal is absent")
        if phase_sum != tokens:
            raise DataMixturePlanError("phase source budgets do not equal phase tokens")

    if cumulative_tokens != total_tokens:
        raise DataMixturePlanError("phase token budgets do not equal total tokens")
    for source_id, source in sources.items():
        if emitted_by_source[source_id] != source["planned_tokens"]:
            raise DataMixturePlanError(
                "source phase allocations differ from its budget"
            )

    controls = _exact_keys(plan["controls"], _CONTROL_KEYS, "mixture controls")
    if controls != {
        "same_sequence_multiset_order_control": True,
        "tokenizer_factor_isolated": True,
        "architecture_factor_isolated": True,
        "terminal_benchmarks_used_for_tuning": False,
    }:
        raise DataMixturePlanError("data-factor isolation controls differ")
    return plan


def validate_plan(path: Path) -> dict[str, Any]:
    """Open and validate an immutable prospective mixture plan."""

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise DataMixturePlanError("data mixture plan is missing or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > _MAX_PLAN_BYTES
        ):
            raise DataMixturePlanError("data mixture plan is missing or unsafe")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            encoded = handle.read(_MAX_PLAN_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(encoded) != before.st_size or len(encoded) > _MAX_PLAN_BYTES:
        raise DataMixturePlanError("data mixture plan size differs")
    if (
        before.st_dev,
        before.st_ino,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise DataMixturePlanError("data mixture plan changed while reading")
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataMixturePlanError("data mixture plan JSON differs") from error
    return validate_payload(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    payload = validate_plan(args.plan)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "status": "validated_prospective",
                "receipt_sha256": payload["receipt_sha256"],
                "training_authorized": False,
                "four_b_training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
