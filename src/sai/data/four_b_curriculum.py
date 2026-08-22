"""Validate Sai's exact prospective 120B-token curriculum candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256

SCHEMA = "sai-4b-120b-token-curriculum-candidate-v1"
PHASES = ("grounding", "integration", "reasoning", "specialization")
POOLS = ("foundation", "math", "code", "science", "reasoning")
TOP_KEYS = {
    "schema",
    "status",
    "user_four_b_training_authorization_received",
    "data_ready",
    "full_training_launch_allowed",
    "architecture_improvement_claimed",
    "target",
    "source_inventories",
    "canonical_source_pools",
    "phases",
    "total_by_pool",
    "admission_policy",
    "remaining_gates",
    "receipt_sha256",
}
TARGET = {
    "model_scale": "4b",
    "sequence_length": 2048,
    "total_sequences": 58_593_750,
    "total_tokens": 120_000_000_000,
    "token_ids_encoding": "uint16_little_endian_for_exact_48000_token_vocabulary",
    "document_boundary_mask_encoding": "one_bit_per_token",
    "terminal_partial_sequence_policy": "none",
}
ADMISSION_POLICY = {
    "canonical_document_stored_once_before_exposure_weighting": True,
    "global_exact_cross_source_deduplication_required": True,
    "high_confidence_near_duplicate_filter_required": True,
    "benchmark_decontamination_after_every_text_transformation": True,
    "stack_code_requires_per_file_permissive_license": True,
    "no_license_and_missing_license_code_excluded": True,
    "synthetic_generator_and_prompt_provenance_required": True,
    "synthetic_reasoning_forbidden_before_reasoning_phase": True,
    "foundational_rehearsal_required_in_every_phase": True,
    "phase_boundaries_must_align_with_optimizer_updates": True,
    "tokenizer_must_be_selected_before_packing": True,
    "source_shortfall_policy": "fail_without_silent_reweighting",
}
REMAINING_GATES = [
    "complete_source_component_selection_and_license_decisions",
    "global_exact_and_high_confidence_near_duplicate_clustering",
    "benchmark_decontamination",
    "semantic_prerequisite_phase_assignment",
    "numeric_tokenizer_ablation_and_exact_48k_selection",
    "lossless_tokenization_and_packed_stream_replay",
    "bounded_one_update_4b_execution_canary",
]


class FourBCurriculumError(RuntimeError):
    """The prospective 4B curriculum differs from its frozen contract."""


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FourBCurriculumError(f"{field} must be a lowercase SHA256")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FourBCurriculumError(f"{field} must be a positive integer")
    return value


def validate_payload(payload: Any) -> dict[str, Any]:
    """Recompute every total and immutable prospective decision."""

    if not isinstance(payload, dict) or set(payload) != TOP_KEYS:
        raise FourBCurriculumError("curriculum top-level keys differ")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload["receipt_sha256"] != canonical_sha256(unsigned):
        raise FourBCurriculumError("curriculum receipt hash differs")
    _sha256(payload["receipt_sha256"], "curriculum receipt")
    if (
        payload["schema"] != SCHEMA
        or payload["status"] != "prospective_source_population_build_required"
        or payload["user_four_b_training_authorization_received"] is not True
        or payload["data_ready"] is not False
        or payload["full_training_launch_allowed"] is not False
        or payload["architecture_improvement_claimed"] is not False
        or payload["target"] != TARGET
        or payload["admission_policy"] != ADMISSION_POLICY
        or payload["remaining_gates"] != REMAINING_GATES
    ):
        raise FourBCurriculumError("curriculum boundary differs")

    inventories = payload["source_inventories"]
    if not isinstance(inventories, list) or len(inventories) != 2:
        raise FourBCurriculumError("source inventory count differs")
    roles: set[str] = set()
    datasets: set[tuple[str, str]] = set()
    for row in inventories:
        if not isinstance(row, dict) or set(row) != {
            "role",
            "dataset",
            "revision",
            "inventory_receipt_sha256",
            "stratified_audit_receipt_sha256",
        }:
            raise FourBCurriculumError("source inventory keys differ")
        if row["role"] not in {"foundation", "late_curriculum"}:
            raise FourBCurriculumError("source inventory role differs")
        if row["role"] in roles:
            raise FourBCurriculumError("source inventory role is duplicated")
        roles.add(row["role"])
        if not isinstance(row["dataset"], str) or not row["dataset"]:
            raise FourBCurriculumError("source dataset differs")
        revision = row["revision"]
        if (
            not isinstance(revision, str)
            or len(revision) != 40
            or any(character not in "0123456789abcdef" for character in revision)
        ):
            raise FourBCurriculumError("source revision differs")
        identity = (row["dataset"], revision)
        if identity in datasets:
            raise FourBCurriculumError("source dataset identity is duplicated")
        datasets.add(identity)
        _sha256(row["inventory_receipt_sha256"], "inventory receipt")
        _sha256(row["stratified_audit_receipt_sha256"], "audit receipt")

    pool_rows = payload["canonical_source_pools"]
    if not isinstance(pool_rows, list) or len(pool_rows) != len(POOLS):
        raise FourBCurriculumError("canonical source-pool count differs")
    seen_pools = set()
    for row in pool_rows:
        if not isinstance(row, dict) or set(row) != {
            "pool",
            "minimum_phase",
            "candidate_components",
        }:
            raise FourBCurriculumError("canonical source-pool keys differ")
        pool = row["pool"]
        if pool not in POOLS or pool in seen_pools:
            raise FourBCurriculumError("canonical source pool differs or is duplicated")
        seen_pools.add(pool)
        expected_minimum = "reasoning" if pool == "reasoning" else "grounding"
        components = row["candidate_components"]
        if (
            row["minimum_phase"] != expected_minimum
            or not isinstance(components, list)
            or not components
            or len(components) != len(set(components))
            or any(not isinstance(value, str) or not value for value in components)
        ):
            raise FourBCurriculumError("canonical source-pool policy differs")

    phases = payload["phases"]
    if not isinstance(phases, list) or len(phases) != len(PHASES):
        raise FourBCurriculumError("curriculum phase count differs")
    cumulative_tokens = 0
    totals = {pool: 0 for pool in POOLS}
    for index, (name, phase) in enumerate(zip(PHASES, phases, strict=True)):
        if not isinstance(phase, dict) or set(phase) != {
            "phase",
            "index",
            "sequences",
            "tokens",
            "cumulative_tokens",
            "by_pool_sequences",
        }:
            raise FourBCurriculumError("curriculum phase keys differ")
        sequences = _positive_int(phase["sequences"], "phase sequences")
        by_pool = phase["by_pool_sequences"]
        if (
            phase["phase"] != name
            or phase["index"] != index
            or not isinstance(by_pool, dict)
            or tuple(by_pool) != POOLS
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in by_pool.values()
            )
            or sum(by_pool.values()) != sequences
        ):
            raise FourBCurriculumError("curriculum phase allocation differs")
        if by_pool["foundation"] <= 0 or (index < 2 and by_pool["reasoning"] != 0):
            raise FourBCurriculumError(
                "curriculum prerequisite or rehearsal policy differs"
            )
        tokens = sequences * TARGET["sequence_length"]
        cumulative_tokens += tokens
        if phase["tokens"] != tokens or phase["cumulative_tokens"] != cumulative_tokens:
            raise FourBCurriculumError("curriculum phase token ledger differs")
        for pool, value in by_pool.items():
            totals[pool] += value

    if (
        sum(phase["sequences"] for phase in phases) != TARGET["total_sequences"]
        or cumulative_tokens != TARGET["total_tokens"]
    ):
        raise FourBCurriculumError("curriculum total differs")
    declared_totals = payload["total_by_pool"]
    if not isinstance(declared_totals, dict) or tuple(declared_totals) != POOLS:
        raise FourBCurriculumError("pool-total keys differ")
    for pool, sequences in totals.items():
        if declared_totals[pool] != {
            "sequences": sequences,
            "tokens": sequences * TARGET["sequence_length"],
        }:
            raise FourBCurriculumError("pool-total ledger differs")
    return payload


def validate_file(path: Path) -> dict[str, Any]:
    """Read a regular single-link JSON file and validate its exact payload."""

    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise FourBCurriculumError("curriculum file is missing or unsafe")
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FourBCurriculumError("curriculum file is unreadable") from error
    return validate_payload(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("curriculum", type=Path)
    args = parser.parse_args()
    payload = validate_file(args.curriculum)
    print(
        json.dumps(
            {"receipt_sha256": payload["receipt_sha256"], "status": payload["status"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
