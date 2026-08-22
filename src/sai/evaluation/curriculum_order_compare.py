"""Compare matched curriculum and exact-sequence-order-control training runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from sai.data.curriculum_control import validate_order_control
from sai.data.curriculum_split import validate_curriculum_split
from sai.data.token_stream import canonical_sha256, sha256_file, validate_frozen_stream
from sai.training.checkpoint import MANIFEST_SCHEMA

SCHEMA = "sai-curriculum-order-training-comparison-v1"
SHARED_SPECIFICATION_FIELDS = (
    "config",
    "config_sha256",
    "model_sha256",
    "delta_backend",
    "initialization_policy_sha256",
    "initialization_seed",
    "development_stream_identity_sha256",
    "code_sha256",
    "environment_sha256",
    "optimizer",
    "precision",
    "micro_batch_size_sequences",
    "sequences_per_update",
    "training_sequences",
    "development_sequences",
    "development_batch_size_sequences",
    "checkpoint_interval_steps",
    "mechanics_only",
    "parameter_count",
    "initialization",
)
RUN_SPECIFICATION_FIELDS = (
    "schema",
    "evidence_class",
    "scientific_promotion_authorized",
    "four_b_training_authorized",
    "config",
    "config_sha256",
    "model_sha256",
    "delta_backend",
    "initialization_policy_sha256",
    "initialization_seed",
    "training_stream_identity_sha256",
    "development_stream_identity_sha256",
    "code_sha256",
    "environment_sha256",
    "optimizer",
    "precision",
    "micro_batch_size_sequences",
    "sequences_per_update",
    "training_sequences",
    "training_utf8_bytes",
    "development_sequences",
    "development_batch_size_sequences",
    "checkpoint_interval_steps",
    "mechanics_only",
)


class CurriculumOrderComparisonError(RuntimeError):
    """A stream, result, matched binding, or score differs."""


def _load_result(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise CurriculumOrderComparisonError("training result is missing or unsafe")
    file_sha256 = sha256_file(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CurriculumOrderComparisonError("training result is unreadable") from error
    if not isinstance(payload, dict):
        raise CurriculumOrderComparisonError("training result differs")
    unsigned = dict(payload)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    validation = payload.get("development_nll")
    counters = payload.get("counters")
    checkpoint = payload.get("checkpoint")
    specification = {field: payload.get(field) for field in RUN_SPECIFICATION_FIELDS}
    if (
        payload.get("schema") != "sai-sub-4b-short-screen-v1"
        or payload.get("status") != "complete"
        or payload.get("delta_backend") != "reference"
        or payload.get("config", {}).get("mixer_family") != "gated_gqa"
        or payload.get("scientific_promotion_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("mechanics_only") is not False
        or payload.get("training_sequences") != 244_140
        or payload.get("sequences_per_update") != 256
        or payload.get("micro_batch_size_sequences") != 8
        or payload.get("optimizer", {}).get("optimizer_steps") != 954
        or payload.get("development_sequences") != 1_024
        or receipt_sha256 != canonical_sha256(unsigned)
        or not isinstance(validation, dict)
        or not isinstance(counters, dict)
        or counters.get("optimizer_steps") != 954
        or counters.get("sequences") != 244_140
        or isinstance(counters.get("targets"), bool)
        or not isinstance(counters.get("targets"), int)
        or counters["targets"] <= 0
        or not isinstance(checkpoint, dict)
        or payload.get("run_sha256") != canonical_sha256(specification)
        or not isinstance(checkpoint.get("sha256"), str)
        or len(checkpoint["sha256"]) != 64
        or any(
            character not in "0123456789abcdef" for character in checkpoint["sha256"]
        )
    ):
        raise CurriculumOrderComparisonError("training result receipt differs")
    numeric = (
        "negative_log_likelihood",
        "nll_per_target",
        "perplexity",
        "nll_per_utf8_byte",
    )
    if (
        validation.get("sequences") != 1_024
        or isinstance(validation.get("targets"), bool)
        or not isinstance(validation.get("targets"), int)
        or validation["targets"] <= 0
        or isinstance(validation.get("admitted_utf8_bytes"), bool)
        or not isinstance(validation.get("admitted_utf8_bytes"), int)
        or validation["admitted_utf8_bytes"] <= 0
        or any(
            isinstance(validation.get(field), bool)
            or not isinstance(validation.get(field), (int, float))
            or not math.isfinite(validation[field])
            or validation[field] <= 0
            for field in numeric
        )
    ):
        raise CurriculumOrderComparisonError("development NLL differs")
    return payload, file_sha256


def _checkpoint_bundle(result_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the exact checkpoint bundle identity used by development MC."""

    descriptor = result.get("checkpoint")
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"path", "bytes", "sha256"}
        or not isinstance(descriptor["path"], str)
        or Path(descriptor["path"]).name != descriptor["path"]
        or isinstance(descriptor["bytes"], bool)
        or not isinstance(descriptor["bytes"], int)
        or descriptor["bytes"] <= 0
    ):
        raise CurriculumOrderComparisonError("checkpoint descriptor differs")
    checkpoint = result_path.parent / descriptor["path"]
    manifest = checkpoint.with_name(f"{checkpoint.name}.manifest.json")
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or checkpoint.stat().st_size != descriptor["bytes"]
        or sha256_file(checkpoint) != descriptor["sha256"]
        or not manifest.is_file()
        or manifest.is_symlink()
    ):
        raise CurriculumOrderComparisonError("checkpoint artifact differs")
    try:
        manifest_payload = json.loads(manifest.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CurriculumOrderComparisonError(
            "checkpoint manifest is unreadable"
        ) from error
    if (
        not isinstance(manifest_payload, dict)
        or manifest_payload.get("schema") != MANIFEST_SCHEMA
        or manifest_payload.get("checkpoint") != descriptor
    ):
        raise CurriculumOrderComparisonError("checkpoint manifest differs")
    manifest_sha256 = sha256_file(manifest)
    rows = [
        {
            "name": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": descriptor["sha256"],
        },
        {
            "name": manifest.name,
            "bytes": manifest.stat().st_size,
            "sha256": manifest_sha256,
        },
    ]
    return {
        "checkpoint_file_sha256": descriptor["sha256"],
        "checkpoint_manifest_file_sha256": manifest_sha256,
        "checkpoint_bundle_sha256": canonical_sha256(rows),
    }


def _validate_phase_strata(
    result: dict[str, Any], development: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    curriculum = development.get("curriculum")
    if not isinstance(curriculum, dict):
        raise CurriculumOrderComparisonError("development curriculum strata are absent")
    phase_order = curriculum.get("phase_order")
    phase_sequences = curriculum.get("phase_sequences_emitted")
    phase_bytes = curriculum.get("consumed_phase_utf8_bytes")
    if (
        curriculum.get("phase_token_budget_enforced") is not True
        or not isinstance(phase_order, list)
        or not phase_order
        or len(phase_order) != len(set(phase_order))
        or not isinstance(phase_sequences, dict)
        or set(phase_sequences) != set(phase_order)
        or not isinstance(phase_bytes, dict)
        or set(phase_bytes) != set(phase_order)
    ):
        raise CurriculumOrderComparisonError("development curriculum strata differ")
    validation = result["development_nll"]
    strata = validation.get("strata")
    expected_fields = {
        "sequences",
        "targets",
        "admitted_utf8_bytes",
        "negative_log_likelihood",
        "nll_per_target",
        "perplexity",
        "nll_per_utf8_byte",
    }
    if not isinstance(strata, dict) or set(strata) != set(phase_order):
        raise CurriculumOrderComparisonError("development phase evidence differs")
    for phase in phase_order:
        row = strata[phase]
        if (
            not isinstance(row, dict)
            or set(row) != expected_fields
            or row.get("sequences") != phase_sequences[phase]
            or row.get("admitted_utf8_bytes") != phase_bytes[phase]
            or isinstance(row.get("targets"), bool)
            or not isinstance(row.get("targets"), int)
            or row["targets"] <= 0
            or any(
                isinstance(row.get(field), bool)
                or not isinstance(row.get(field), (int, float))
                or not math.isfinite(row[field])
                or row[field] <= 0
                for field in (
                    "negative_log_likelihood",
                    "nll_per_target",
                    "perplexity",
                    "nll_per_utf8_byte",
                )
            )
        ):
            raise CurriculumOrderComparisonError("development phase evidence differs")
    if (
        sum(row["sequences"] for row in strata.values()) != validation["sequences"]
        or sum(row["targets"] for row in strata.values()) != validation["targets"]
        or sum(row["admitted_utf8_bytes"] for row in strata.values())
        != validation["admitted_utf8_bytes"]
        or not math.isclose(
            sum(row["negative_log_likelihood"] for row in strata.values()),
            validation["negative_log_likelihood"],
            rel_tol=1e-9,
            abs_tol=1e-4,
        )
    ):
        raise CurriculumOrderComparisonError("development phase totals differ")
    return strata


def compare_curriculum_order(
    curriculum_result: Path,
    control_result: Path,
    *,
    curriculum_stream: Path,
    control_stream: Path,
    development_stream: Path,
    split_receipt: Path,
    curriculum_workers: int = 1,
) -> dict[str, Any]:
    """Prove exact matching and report the held-out effect of order alone."""

    split = validate_curriculum_split(
        split_receipt, curriculum_workers=curriculum_workers
    )
    curriculum = validate_frozen_stream(curriculum_stream, verify_sources=True)
    control = validate_order_control(control_stream)
    development = validate_frozen_stream(development_stream, verify_sources=True)
    split_file_sha256 = sha256_file(split_receipt)
    ordering = control["ordering_control"]
    if (
        curriculum.get("source_qualification_sha256") != split_file_sha256
        or control.get("source_qualification_sha256") != split_file_sha256
        or development.get("source_qualification_sha256") != split_file_sha256
        or curriculum["tokenizer_identity_sha256"]
        != control["tokenizer_identity_sha256"]
        or curriculum["tokenizer_identity_sha256"]
        != development["tokenizer_identity_sha256"]
        or curriculum["sequence_length"] != 2_048
        or control["sequence_length"] != 2_048
        or development["sequence_length"] != 2_048
        or curriculum["sequences"] != 244_140
        or control["sequences"] != 244_140
        or development["sequences"] != 1_024
        or curriculum["admitted_utf8_bytes"] != control["admitted_utf8_bytes"]
        or ordering["parent_stream"]["ordered_stream_identity_sha256"]
        != curriculum["ordered_stream_identity_sha256"]
        or ordering["same_tokens_and_boundary_masks"] is not True
        or ordering["same_sequence_multiset"] is not True
        or ordering["only_sequence_order_changed"] is not True
        or curriculum["ordered_stream_identity_sha256"]
        == control["ordered_stream_identity_sha256"]
        or split["train"]["path"] != curriculum["source_receipts"][0]["path"]
        or split["development"]["path"] != development["source_receipts"][0]["path"]
    ):
        raise CurriculumOrderComparisonError("matched stream contract differs")

    curriculum_result_payload, curriculum_result_sha256 = _load_result(
        curriculum_result
    )
    control_result_payload, control_result_sha256 = _load_result(control_result)
    curriculum_checkpoint = _checkpoint_bundle(
        curriculum_result, curriculum_result_payload
    )
    control_checkpoint = _checkpoint_bundle(control_result, control_result_payload)
    curriculum_strata = _validate_phase_strata(curriculum_result_payload, development)
    control_strata = _validate_phase_strata(control_result_payload, development)
    if any(
        curriculum_result_payload.get(field) != control_result_payload.get(field)
        for field in SHARED_SPECIFICATION_FIELDS
    ):
        raise CurriculumOrderComparisonError("matched training specification differs")
    development_identity = development["ordered_stream_identity_sha256"]
    curriculum_identity = curriculum["ordered_stream_identity_sha256"]
    control_identity = control["ordered_stream_identity_sha256"]
    if (
        curriculum_result_payload.get("training_stream_identity_sha256")
        != curriculum_identity
        or control_result_payload.get("training_stream_identity_sha256")
        != control_identity
        or curriculum_result_payload.get("development_stream_identity_sha256")
        != development_identity
        or control_result_payload.get("development_stream_identity_sha256")
        != development_identity
        or curriculum_result_payload.get("training_utf8_bytes")
        != curriculum["admitted_utf8_bytes"]
        or control_result_payload.get("training_utf8_bytes")
        != control["admitted_utf8_bytes"]
        or curriculum_result_payload["development_nll"]["stream_identity_sha256"]
        != development_identity
        or control_result_payload["development_nll"]["stream_identity_sha256"]
        != development_identity
        or curriculum_result_payload["development_nll"]["targets"]
        != control_result_payload["development_nll"]["targets"]
        or curriculum_result_payload["development_nll"]["admitted_utf8_bytes"]
        != control_result_payload["development_nll"]["admitted_utf8_bytes"]
        or curriculum_result_payload["checkpoint"].get("sha256")
        == control_result_payload["checkpoint"].get("sha256")
        or any(
            curriculum_strata[phase][field] != control_strata[phase][field]
            for phase in curriculum_strata
            for field in ("sequences", "targets", "admitted_utf8_bytes")
        )
    ):
        raise CurriculumOrderComparisonError("training result lineage differs")
    curriculum_nll = curriculum_result_payload["development_nll"]["nll_per_target"]
    control_nll = control_result_payload["development_nll"]["nll_per_target"]
    curriculum_byte_nll = curriculum_result_payload["development_nll"][
        "nll_per_utf8_byte"
    ]
    control_byte_nll = control_result_payload["development_nll"]["nll_per_utf8_byte"]
    phase_deltas = {
        phase: {
            "nll_per_target": curriculum_strata[phase]["nll_per_target"]
            - control_strata[phase]["nll_per_target"],
            "perplexity": curriculum_strata[phase]["perplexity"]
            - control_strata[phase]["perplexity"],
            "nll_per_utf8_byte": curriculum_strata[phase]["nll_per_utf8_byte"]
            - control_strata[phase]["nll_per_utf8_byte"],
        }
        for phase in curriculum_strata
    }
    phase_no_regression = all(
        row["nll_per_target"] <= 0 and row["nll_per_utf8_byte"] <= 0
        for row in phase_deltas.values()
    )
    supported = bool(
        curriculum_nll < control_nll
        and curriculum_byte_nll < control_byte_nll
        and phase_no_regression
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "comparison": "curriculum_order_vs_exact_sequence_multiset_permutation",
        "same_documents_tokens_targets_masks": True,
        "only_training_sequence_order_changed": True,
        "same_model_initialization_optimizer_budget_compute": True,
        "development_population_disjoint_from_training": True,
        "training_tokens_per_arm": 499_998_720,
        "optimizer_steps_per_arm": 954,
        "development_sequences": 1_024,
        "curriculum_order_supported_by_heldout_nll": supported,
        "heldout_phase_no_regression": phase_no_regression,
        "curriculum_minus_control": {
            "nll_per_target": curriculum_nll - control_nll,
            "perplexity": curriculum_result_payload["development_nll"]["perplexity"]
            - control_result_payload["development_nll"]["perplexity"],
            "nll_per_utf8_byte": curriculum_byte_nll - control_byte_nll,
        },
        "phase_minus_control": phase_deltas,
        "arms": {
            "curriculum": {
                "stream_identity_sha256": curriculum_identity,
                "result_file_sha256": curriculum_result_sha256,
                "result_receipt_sha256": curriculum_result_payload["receipt_sha256"],
                "checkpoint_sha256": curriculum_checkpoint["checkpoint_bundle_sha256"],
                **curriculum_checkpoint,
                "development_nll": curriculum_result_payload["development_nll"],
            },
            "order_control": {
                "stream_identity_sha256": control_identity,
                "result_file_sha256": control_result_sha256,
                "result_receipt_sha256": control_result_payload["receipt_sha256"],
                "checkpoint_sha256": control_checkpoint["checkpoint_bundle_sha256"],
                **control_checkpoint,
                "development_nll": control_result_payload["development_nll"],
            },
        },
        "bindings": {
            "split_receipt_file_sha256": split_file_sha256,
            "split_receipt_sha256": split["receipt_sha256"],
            "sequence_multiset_sha256": ordering["sequence_multiset_sha256"],
            "permutation_sha256": ordering["permutation_sha256"],
            "development_stream_identity_sha256": development_identity,
            "tokenizer_identity_sha256": curriculum["tokenizer_identity_sha256"],
            "code_sha256": curriculum_result_payload["code_sha256"],
            "environment_sha256": curriculum_result_payload["environment_sha256"],
        },
        "replication_required_for_architecture_claim": True,
        "real_benchmark_gate_required": True,
        "scientific_promotion_authorized": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def write_comparison(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise CurriculumOrderComparisonError("comparison output path differs")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-result", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--curriculum-stream", type=Path, required=True)
    parser.add_argument("--control-stream", type=Path, required=True)
    parser.add_argument("--development-stream", type=Path, required=True)
    parser.add_argument("--split-receipt", type=Path, required=True)
    parser.add_argument("--curriculum-validation-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare_curriculum_order(
        args.curriculum_result,
        args.control_result,
        curriculum_stream=args.curriculum_stream,
        control_stream=args.control_stream,
        development_stream=args.development_stream,
        split_receipt=args.split_receipt,
        curriculum_workers=args.curriculum_validation_workers,
    )
    write_comparison(args.output, payload)
    print(
        json.dumps(
            {"status": payload["status"], "receipt_sha256": payload["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
