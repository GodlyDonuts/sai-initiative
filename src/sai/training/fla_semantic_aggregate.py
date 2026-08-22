"""Aggregate exactly three immutable Sai FLA semantic-parity v2 receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256
from sai.training.fla_semantic_parity import (
    CALIBRATION_SEEDS,
    CONV_THRESHOLDS,
    FAMILIES,
    PRODUCTION_SEEDS,
    RECURRENCE_THRESHOLDS,
    SEQUENCE_LENGTHS,
)
from sai.training.fla_semantic_parity import (
    SCHEMA as SEED_SCHEMA,
)

SCHEMA = "sai-fla-semantic-parity-aggregate-v2"
_RECEIPT_KEYS = {
    "schema",
    "status",
    "production_cuda_qualified",
    "scope",
    "fla_version",
    "seed",
    "production_seed_allowlist",
    "excluded_calibration_seeds",
    "dtype",
    "environment",
    "operators",
    "thresholds",
    "family_results",
    "cases",
    "checks",
    "optimizer_steps",
    "training_gpu_jobs_submitted",
    "training_authorized",
    "architecture_promoted",
    "four_b_training_authorized",
    "limitations",
    "receipt_sha256",
}
_CASE_KEYS = {
    "family",
    "sequence_length",
    "structural_mapping",
    "causal_convolution",
    "packed_recurrence",
    "passed",
}
_STRUCTURAL_KEYS = {
    "packed_cu_seqlens",
    "equal_id_across_row_boundary",
    "explicit_scale_one",
    "external_qk_normalization",
    "family_flags_passed",
    "passed",
}
_METRIC_KEYS = {
    "threshold",
    "root_mean_square_error",
    "reference_root_mean_square",
    "relative_root_mean_square_error",
    "max_absolute_error",
    "elements_compared",
    "all_finite",
    "passed",
}
_LIMITATIONS = [
    "semantic_kernel_mapping_only_not_model_quality_evidence",
    "bounded_lengths_1_63_64_65_not_exact_b8_x_2048",
    "one_seed_per_receipt_requires_all_frozen_seeds_separately",
    "does_not_reinterpret_or_replace_v1_receipts",
]


class FlaSemanticAggregateError(RuntimeError):
    """A seed receipt, cross-seed identity, or aggregate differs."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FlaSemanticAggregateError("semantic seed receipt is missing or unsafe")
    return path


def _finite_number(value: Any, *, nonnegative: bool = True) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and (not nonnegative or float(value) >= 0)
    )


def _validate_metric(value: Any, threshold: float) -> bool:
    if not isinstance(value, dict) or set(value) != _METRIC_KEYS:
        raise FlaSemanticAggregateError("semantic metric keys differ")
    if (
        value["threshold"] != threshold
        or not _finite_number(value["root_mean_square_error"])
        or not _finite_number(value["reference_root_mean_square"])
        or not _finite_number(value["relative_root_mean_square_error"])
        or not _finite_number(value["max_absolute_error"])
        or isinstance(value["elements_compared"], bool)
        or not isinstance(value["elements_compared"], int)
        or value["elements_compared"] <= 0
        or not isinstance(value["all_finite"], bool)
        or not isinstance(value["passed"], bool)
    ):
        raise FlaSemanticAggregateError("semantic metric value differs")
    computed = value["all_finite"] and (
        value["relative_root_mean_square_error"] < threshold
    )
    if value["passed"] is not computed:
        raise FlaSemanticAggregateError("semantic metric summary differs")
    return computed


def _expected_offsets(sequence_length: int) -> list[int]:
    return {
        1: [0, 1, 2],
        63: [0, 63, 126],
        64: [0, 63, 64, 128],
        65: [0, 1, 64, 65, 130],
    }[sequence_length]


def _validate_case(case: Any, family: str, sequence_length: int) -> bool:
    if not isinstance(case, dict) or set(case) != _CASE_KEYS:
        raise FlaSemanticAggregateError("semantic case keys differ")
    if case["family"] != family or case["sequence_length"] != sequence_length:
        raise FlaSemanticAggregateError("semantic case identity differs")
    structural = case["structural_mapping"]
    if not isinstance(structural, dict) or set(structural) != _STRUCTURAL_KEYS:
        raise FlaSemanticAggregateError("semantic structural evidence keys differ")
    structural_passed = structural["packed_cu_seqlens"] == _expected_offsets(
        sequence_length
    ) and all(
        structural[field] is True
        for field in (
            "equal_id_across_row_boundary",
            "explicit_scale_one",
            "external_qk_normalization",
            "family_flags_passed",
        )
    )
    if structural["passed"] is not structural_passed:
        raise FlaSemanticAggregateError("semantic structural summary differs")

    convolution = case["causal_convolution"]
    if not isinstance(convolution, list) or len(convolution) != 3:
        raise FlaSemanticAggregateError("semantic convolution probes differ")
    conv_passes = []
    for probe, label in zip(convolution, ("q", "k", "v"), strict=True):
        if not isinstance(probe, dict) or set(probe) != {"label", "metrics", "passed"}:
            raise FlaSemanticAggregateError("semantic convolution probe keys differ")
        metrics = probe["metrics"]
        if probe["label"] != label or not isinstance(metrics, dict):
            raise FlaSemanticAggregateError("semantic convolution probe differs")
        if set(metrics) != set(CONV_THRESHOLDS):
            raise FlaSemanticAggregateError("semantic convolution metrics differ")
        passed = all(
            _validate_metric(metrics[name], threshold)
            for name, threshold in CONV_THRESHOLDS.items()
        )
        if probe["passed"] is not passed:
            raise FlaSemanticAggregateError("semantic convolution summary differs")
        conv_passes.append(passed)

    recurrence = case["packed_recurrence"]
    if not isinstance(recurrence, dict) or set(recurrence) != {"metrics", "passed"}:
        raise FlaSemanticAggregateError("semantic recurrence keys differ")
    metrics = recurrence["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != set(RECURRENCE_THRESHOLDS):
        raise FlaSemanticAggregateError("semantic recurrence metrics differ")
    recurrence_passed = all(
        _validate_metric(metrics[name], threshold)
        for name, threshold in RECURRENCE_THRESHOLDS.items()
    )
    if recurrence["passed"] is not recurrence_passed:
        raise FlaSemanticAggregateError("semantic recurrence summary differs")
    passed = structural_passed and all(conv_passes) and recurrence_passed
    if case["passed"] is not passed:
        raise FlaSemanticAggregateError("semantic case summary differs")
    return passed


def _validate_environment(value: Any) -> dict[str, Any]:
    expected_keys = {
        "device_type",
        "torch",
        "torch_cuda",
        "cuda_available",
        "cuda_bf16_supported",
        "cuda_device_name",
        "cuda_capability",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value["device_type"] != "cuda"
        or value["cuda_available"] is not True
        or value["cuda_bf16_supported"] is not True
        or not isinstance(value["torch"], str)
        or not value["torch"]
        or not isinstance(value["torch_cuda"], str)
        or not value["torch_cuda"]
        or not isinstance(value["cuda_device_name"], str)
        or not value["cuda_device_name"]
        or not isinstance(value["cuda_capability"], list)
        or len(value["cuda_capability"]) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value["cuda_capability"]
        )
    ):
        raise FlaSemanticAggregateError("semantic CUDA environment differs")
    return value


def _validate_operators(value: Any) -> dict[str, str]:
    expected = {"causal_conv1d", "gated_delta_chunk", "kda_chunk"}
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or not all(isinstance(item, str) and item for item in value.values())
        or not value["causal_conv1d"].endswith(".causal_conv1d")
        or not value["gated_delta_chunk"].endswith(".chunk_gated_delta_rule")
        or not value["kda_chunk"].endswith(".chunk_kda")
    ):
        raise FlaSemanticAggregateError("semantic operator identity differs")
    return value


def load_seed_receipt(path: Path) -> dict[str, Any]:
    path = _regular_file(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FlaSemanticAggregateError(
            "semantic seed receipt is unreadable"
        ) from error
    if not isinstance(payload, dict) or set(payload) != _RECEIPT_KEYS:
        raise FlaSemanticAggregateError("semantic seed receipt keys differ")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if payload["receipt_sha256"] != canonical_sha256(unsigned):
        raise FlaSemanticAggregateError("semantic seed receipt hash differs")
    seed = payload["seed"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in PRODUCTION_SEEDS
    ):
        raise FlaSemanticAggregateError("semantic production seed differs")
    if payload["thresholds"] != {
        "strict_less_than": True,
        "causal_convolution": CONV_THRESHOLDS,
        "packed_recurrence": RECURRENCE_THRESHOLDS,
    }:
        raise FlaSemanticAggregateError("semantic strict thresholds differ")
    if (
        payload["schema"] != SEED_SCHEMA
        or payload["scope"] != "direct_packed_fla_semantic_forward_backward_parity"
        or payload["fla_version"] != "0.4.2"
        or payload["production_seed_allowlist"] != list(PRODUCTION_SEEDS)
        or payload["excluded_calibration_seeds"] != list(CALIBRATION_SEEDS)
        or payload["dtype"] != "torch.bfloat16"
        or payload["checks"]
        != {
            "no_cross_case_averaging": True,
            "all_tensors_finite_required": True,
            "structural_mapping_required": True,
            "every_family_case_required": True,
        }
        or payload["optimizer_steps"] != 0
        or payload["training_gpu_jobs_submitted"] != 0
        or payload["training_authorized"] is not False
        or payload["architecture_promoted"] is not False
        or payload["four_b_training_authorized"] is not False
        or payload["limitations"] != _LIMITATIONS
    ):
        raise FlaSemanticAggregateError("semantic seed receipt contract differs")
    _validate_environment(payload["environment"])
    _validate_operators(payload["operators"])
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) != 8:
        raise FlaSemanticAggregateError("semantic eight-case geometry differs")
    indexed = {}
    for case in cases:
        if not isinstance(case, dict):
            raise FlaSemanticAggregateError("semantic case differs")
        identity = (case.get("family"), case.get("sequence_length"))
        if identity in indexed:
            raise FlaSemanticAggregateError("semantic case identity is duplicated")
        indexed[identity] = case
    expected_identities = {
        (family, sequence_length)
        for family in FAMILIES
        for sequence_length in SEQUENCE_LENGTHS
    }
    if set(indexed) != expected_identities:
        raise FlaSemanticAggregateError("semantic case population differs")
    family_passes = {}
    for family in FAMILIES:
        results = [
            _validate_case(indexed[(family, sequence_length)], family, sequence_length)
            for sequence_length in SEQUENCE_LENGTHS
        ]
        family_passes[family] = all(results)
    expected_family_results = {
        family: {
            "status": (
                "production_semantics_qualified" if passed else "semantic_parity_failed"
            ),
            "production_semantics_qualified": passed,
            "passed_cases": (
                4
                if passed
                else sum(
                    bool(indexed[(family, length)]["passed"])
                    for length in SEQUENCE_LENGTHS
                )
            ),
            "required_cases": 4,
        }
        for family, passed in family_passes.items()
    }
    all_passed = all(family_passes.values())
    expected_status = (
        "all_families_production_qualified"
        if all_passed
        else "one_or_more_families_failed"
    )
    if (
        payload["family_results"] != expected_family_results
        or payload["production_cuda_qualified"] is not all_passed
        or payload["status"] != expected_status
    ):
        raise FlaSemanticAggregateError("semantic family or receipt summary differs")
    return payload


def aggregate_seed_receipts(paths: list[Path]) -> dict[str, Any]:
    if not isinstance(paths, list) or len(paths) != len(PRODUCTION_SEEDS):
        raise FlaSemanticAggregateError("exactly three seed receipts are required")
    loaded = []
    file_identities = []
    for original in paths:
        path = _regular_file(original)
        before = path.stat()
        before_sha256 = _file_sha256(path)
        payload = load_seed_receipt(path)
        after = path.stat()
        after_sha256 = _file_sha256(path)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or before_sha256 != after_sha256:
            raise FlaSemanticAggregateError("semantic seed receipt changed during read")
        loaded.append((path, payload, after_sha256))
        file_identities.append((after.st_dev, after.st_ino))
    if len(set(file_identities)) != len(file_identities):
        raise FlaSemanticAggregateError("semantic seed receipt file is duplicated")
    seeds = [payload["seed"] for _, payload, _ in loaded]
    if len(set(seeds)) != len(seeds) or set(seeds) != set(PRODUCTION_SEEDS):
        raise FlaSemanticAggregateError("semantic seed population differs")
    environments = [payload["environment"] for _, payload, _ in loaded]
    operators = [payload["operators"] for _, payload, _ in loaded]
    if any(value != environments[0] for value in environments[1:]):
        raise FlaSemanticAggregateError("semantic environment identity drifted")
    if any(value != operators[0] for value in operators[1:]):
        raise FlaSemanticAggregateError("semantic operator identity drifted")
    ordered = sorted(loaded, key=lambda item: item[1]["seed"])
    family_results = {}
    for family in FAMILIES:
        failed = [
            payload["seed"]
            for _, payload, _ in ordered
            if not payload["family_results"][family]["production_semantics_qualified"]
        ]
        qualified = not failed
        family_results[family] = {
            "status": (
                "production_semantics_qualified"
                if qualified
                else "semantic_parity_failed"
            ),
            "production_semantics_qualified": qualified,
            "passed_seeds": len(PRODUCTION_SEEDS) - len(failed),
            "required_seeds": len(PRODUCTION_SEEDS),
            "failed_seeds": failed,
        }
    all_qualified = all(
        result["production_semantics_qualified"] for result in family_results.values()
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "all_families_production_qualified"
            if all_qualified
            else "one_or_more_families_failed"
        ),
        "production_cuda_qualified": all_qualified,
        "scope": "three_seed_fla_semantic_parity_aggregate",
        "fla_version": "0.4.2",
        "seeds": list(PRODUCTION_SEEDS),
        "thresholds": {
            "strict_less_than": True,
            "causal_convolution": dict(CONV_THRESHOLDS),
            "packed_recurrence": dict(RECURRENCE_THRESHOLDS),
        },
        "environment": environments[0],
        "operators": operators[0],
        "inputs": [
            {
                "seed": receipt["seed"],
                "receipt_sha256": receipt["receipt_sha256"],
                "file_sha256": file_sha256,
            }
            for _, receipt, file_sha256 in ordered
        ],
        "family_results": family_results,
        "checks": {
            "exact_three_seed_population": True,
            "unique_seed_receipts": True,
            "environment_identity_consistent": True,
            "operator_identity_consistent": True,
            "no_cross_seed_or_family_averaging": True,
        },
        "optimizer_steps": 0,
        "training_gpu_jobs_submitted": 0,
        "training_authorized": False,
        "architecture_promoted": False,
        "four_b_training_authorized": False,
        "limitations": [
            "aggregate_of_bounded_semantic_kernel_receipts_only",
            "not_model_quality_or_exact_b8_x_2048_evidence",
            "does_not_reinterpret_or_replace_v1_receipts",
        ],
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FlaSemanticAggregateError("semantic aggregate output already exists")
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise FlaSemanticAggregateError("semantic aggregate output parent is unsafe")
    stage = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(stage, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate_seed_receipts(args.receipt)
    _write_create_only(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["production_cuda_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
