"""Compare an equal-token source addition with its selected web-only control."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file, validate_frozen_stream
from sai.model.config import SaiModelConfig, parameter_ledger
from sai.model.initialization import POLICY_SHA256
from sai.training.checkpoint import MANIFEST_SCHEMA

SCHEMA = "sai-source-addition-nll-comparison-v1"
RESULT_SCHEMA = "sai-sub-4b-short-screen-v1"
SHARED_RESULT_FIELDS = (
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


class SourceAdditionComparisonError(RuntimeError):
    """A source, run, matched budget, or likelihood differs."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise SourceAdditionComparisonError("output parent or target is unsafe")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise SourceAdditionComparisonError("output already exists") from error
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_result(path: Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise SourceAdditionComparisonError("training result is missing or unsafe")
    file_sha256 = sha256_file(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceAdditionComparisonError("training result is unreadable") from error
    if not isinstance(payload, dict):
        raise SourceAdditionComparisonError("training result differs")
    unsigned = dict(payload)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    try:
        config = SaiModelConfig(**payload.get("config", {}))
    except (TypeError, ValueError) as error:
        raise SourceAdditionComparisonError("training configuration differs") from error
    config_sha256 = canonical_sha256(config.as_dict())
    model_sha256 = canonical_sha256(
        {
            "config_sha256": config_sha256,
            "delta_backend": "reference",
            "initialization_policy_sha256": POLICY_SHA256,
            "initialization_seed": payload.get("initialization_seed"),
        }
    )
    specification_fields = list(RUN_SPECIFICATION_FIELDS)
    if "milestone_steps" in payload:
        specification_fields.append("milestone_steps")
    if any(field not in payload for field in specification_fields):
        raise SourceAdditionComparisonError("training specification is incomplete")
    specification = {field: payload[field] for field in specification_fields}
    if (
        payload.get("schema") != RESULT_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("delta_backend") != "reference"
        or config.mixer_family != "gated_gqa"
        or payload.get("config") != config.as_dict()
        or payload.get("config_sha256") != config_sha256
        or payload.get("model_sha256") != model_sha256
        or payload.get("initialization_policy_sha256") != POLICY_SHA256
        or payload.get("parameter_count") != parameter_ledger(config)["total"]
        or payload.get("scientific_promotion_authorized") is not False
        or payload.get("four_b_training_authorized") is not False
        or payload.get("mechanics_only") is not False
        or receipt_sha256 != canonical_sha256(unsigned)
        or payload.get("run_sha256") != canonical_sha256(specification)
    ):
        raise SourceAdditionComparisonError("training result receipt differs")
    return payload, file_sha256


def _checkpoint_bundle(result_path: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the exact checkpoint identity consumed by benchmark evaluation."""

    descriptor = result.get("checkpoint")
    if (
        not isinstance(descriptor, dict)
        or set(descriptor) != {"path", "bytes", "sha256"}
        or not isinstance(descriptor["path"], str)
        or Path(descriptor["path"]).name != descriptor["path"]
        or isinstance(descriptor["bytes"], bool)
        or not isinstance(descriptor["bytes"], int)
        or descriptor["bytes"] <= 0
        or not isinstance(descriptor["sha256"], str)
        or len(descriptor["sha256"]) != 64
    ):
        raise SourceAdditionComparisonError("checkpoint descriptor differs")
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
        raise SourceAdditionComparisonError("checkpoint artifact differs")
    try:
        manifest_payload = json.loads(manifest.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceAdditionComparisonError(
            "checkpoint manifest is unreadable"
        ) from error
    if (
        not isinstance(manifest_payload, dict)
        or manifest_payload.get("schema") != MANIFEST_SCHEMA
        or manifest_payload.get("checkpoint") != descriptor
    ):
        raise SourceAdditionComparisonError("checkpoint manifest differs")
    manifest_sha256 = sha256_file(manifest)
    rows = [
        {
            "name": checkpoint.name,
            "bytes": descriptor["bytes"],
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


def _positive_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SourceAdditionComparisonError(f"{field} differs")
    return float(value)


def _validate_nll(result: dict[str, Any]) -> dict[str, Any]:
    nll = result.get("development_nll")
    if not isinstance(nll, dict):
        raise SourceAdditionComparisonError("development NLL differs")
    sequences = nll.get("sequences")
    targets = nll.get("targets")
    byte_count = nll.get("admitted_utf8_bytes")
    if (
        sequences != result.get("development_sequences")
        or isinstance(targets, bool)
        or not isinstance(targets, int)
        or targets <= 0
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
    ):
        raise SourceAdditionComparisonError("development NLL geometry differs")
    for field in (
        "negative_log_likelihood",
        "nll_per_target",
        "perplexity",
        "nll_per_utf8_byte",
    ):
        _positive_number(nll.get(field), f"development {field}")
    strata = nll.get("strata")
    if not isinstance(strata, dict) or not strata:
        raise SourceAdditionComparisonError("development strata are required")
    expected_stratum_fields = {
        "sequences",
        "targets",
        "admitted_utf8_bytes",
        "negative_log_likelihood",
        "nll_per_target",
        "perplexity",
        "nll_per_utf8_byte",
    }
    for name, row in strata.items():
        if not isinstance(name, str) or not name or not isinstance(row, dict):
            raise SourceAdditionComparisonError("development stratum differs")
        if set(row) != expected_stratum_fields:
            raise SourceAdditionComparisonError("development stratum differs")
        if any(
            isinstance(row.get(field), bool)
            or not isinstance(row.get(field), int)
            or row[field] <= 0
            for field in ("sequences", "targets", "admitted_utf8_bytes")
        ):
            raise SourceAdditionComparisonError("development stratum geometry differs")
        for field in (
            "negative_log_likelihood",
            "nll_per_target",
            "perplexity",
            "nll_per_utf8_byte",
        ):
            _positive_number(row.get(field), f"development stratum {field}")
    if (
        sum(row["sequences"] for row in strata.values()) != sequences
        or sum(row["targets"] for row in strata.values()) != targets
        or sum(row["admitted_utf8_bytes"] for row in strata.values()) != byte_count
        or not math.isclose(
            sum(row["negative_log_likelihood"] for row in strata.values()),
            nll["negative_log_likelihood"],
            rel_tol=1e-9,
            abs_tol=1e-4,
        )
    ):
        raise SourceAdditionComparisonError("development stratum totals differ")
    return nll


def compare_payloads(
    treatment: dict[str, Any],
    control: dict[str, Any],
    *,
    treatment_stream: dict[str, Any],
    control_stream: dict[str, Any],
    development_stream: dict[str, Any],
) -> dict[str, Any]:
    """Require an equal-token comparison and compute every regression veto."""

    if any(
        treatment.get(field) != control.get(field) for field in SHARED_RESULT_FIELDS
    ):
        raise SourceAdditionComparisonError("matched training specification differs")
    if treatment.get("run_sha256") == control.get("run_sha256") or treatment.get(
        "training_stream_identity_sha256"
    ) == control.get("training_stream_identity_sha256"):
        raise SourceAdditionComparisonError("source-addition arms are duplicated")
    treatment_nll = _validate_nll(treatment)
    control_nll = _validate_nll(control)
    if set(treatment_nll["strata"]) != set(control_nll["strata"]):
        raise SourceAdditionComparisonError("development strata differ")
    for phase in treatment_nll["strata"]:
        for field in ("sequences", "targets", "admitted_utf8_bytes"):
            if (
                treatment_nll["strata"][phase][field]
                != control_nll["strata"][phase][field]
            ):
                raise SourceAdditionComparisonError(
                    "development stratum geometry differs"
                )

    stream_shared_fields = (
        "schema",
        "status",
        "tokenizer_identity_sha256",
        "sequence_length",
        "sequences",
        "valid_tokens",
        "benchmark_disjoint",
        "cross_document_targets_masked",
        "token_encoding",
        "segment_start_encoding",
        "eos_token_id",
        "vocab_size",
    )
    if any(
        treatment_stream.get(field) != control_stream.get(field)
        for field in stream_shared_fields
    ):
        raise SourceAdditionComparisonError("equal-token stream geometry differs")
    if (
        treatment_stream.get("ordered_stream_identity_sha256")
        != treatment.get("training_stream_identity_sha256")
        or control_stream.get("ordered_stream_identity_sha256")
        != control.get("training_stream_identity_sha256")
        or treatment_stream.get("source_qualification_sha256")
        == control_stream.get("source_qualification_sha256")
        or treatment_stream.get("valid_tokens")
        != treatment.get("training_sequences")
        * treatment_stream.get("sequence_length", 0)
        or development_stream.get("ordered_stream_identity_sha256")
        != treatment.get("development_stream_identity_sha256")
        or development_stream.get("ordered_stream_identity_sha256")
        != control.get("development_stream_identity_sha256")
        or development_stream.get("sequences") != treatment.get("development_sequences")
        or development_stream.get("valid_tokens")
        != treatment.get("development_sequences")
        * development_stream.get("sequence_length", 0)
    ):
        raise SourceAdditionComparisonError("stream lineage differs")
    treatment_sources = treatment_stream.get("source_receipts")
    control_sources = control_stream.get("source_receipts")
    if (
        not isinstance(treatment_sources, list)
        or not treatment_sources
        or not isinstance(control_sources, list)
        or not control_sources
        or treatment_sources == control_sources
    ):
        raise SourceAdditionComparisonError("source composition differs")

    strata: dict[str, Any] = {}
    for stratum in treatment_nll["strata"]:
        treatment_row = treatment_nll["strata"][stratum]
        control_row = control_nll["strata"][stratum]
        target_delta = treatment_row["nll_per_target"] - control_row["nll_per_target"]
        byte_delta = (
            treatment_row["nll_per_utf8_byte"] - control_row["nll_per_utf8_byte"]
        )
        strata[stratum] = {
            "treatment_nll_per_target": treatment_row["nll_per_target"],
            "control_nll_per_target": control_row["nll_per_target"],
            "treatment_minus_control_nll_per_target": target_delta,
            "treatment_nll_per_utf8_byte": treatment_row["nll_per_utf8_byte"],
            "control_nll_per_utf8_byte": control_row["nll_per_utf8_byte"],
            "treatment_minus_control_nll_per_utf8_byte": byte_delta,
            "no_target_normalized_regression": target_delta <= 0,
            "no_byte_normalized_regression": byte_delta <= 0,
        }
    aggregate_target_delta = (
        treatment_nll["nll_per_target"] - control_nll["nll_per_target"]
    )
    aggregate_byte_delta = (
        treatment_nll["nll_per_utf8_byte"] - control_nll["nll_per_utf8_byte"]
    )
    nll_supported = (
        aggregate_target_delta <= 0
        and aggregate_byte_delta <= 0
        and all(
            row["no_target_normalized_regression"]
            and row["no_byte_normalized_regression"]
            for row in strata.values()
        )
    )
    return {
        "comparison": "selected_multi_source_addition_vs_selected_web_only_control",
        "equal_training_tokens": treatment_stream["valid_tokens"],
        "equal_training_sequences": treatment_stream["sequences"],
        "treatment_admitted_utf8_bytes": treatment_stream["admitted_utf8_bytes"],
        "control_admitted_utf8_bytes": control_stream["admitted_utf8_bytes"],
        "aggregate": {
            "treatment_nll_per_target": treatment_nll["nll_per_target"],
            "control_nll_per_target": control_nll["nll_per_target"],
            "treatment_minus_control_nll_per_target": aggregate_target_delta,
            "treatment_nll_per_utf8_byte": treatment_nll["nll_per_utf8_byte"],
            "control_nll_per_utf8_byte": control_nll["nll_per_utf8_byte"],
            "treatment_minus_control_nll_per_utf8_byte": aggregate_byte_delta,
        },
        "strata": strata,
        "source_addition_supported_by_heldout_nll": nll_supported,
        "real_source_disjoint_benchmark_confirmation_required": True,
        "source_addition_retained": False,
        "data_promotion_authorized": False,
        "four_b_training_authorized": False,
    }


def compare_paths(
    treatment_result: Path,
    control_result: Path,
    *,
    treatment_stream: Path,
    control_stream: Path,
    development_stream: Path,
    treatment_training_source_sha256: str,
    control_training_source_sha256: str,
) -> dict[str, Any]:
    treatment, treatment_result_sha256 = _load_result(treatment_result)
    control, control_result_sha256 = _load_result(control_result)
    treatment_checkpoint = _checkpoint_bundle(treatment_result, treatment)
    control_checkpoint = _checkpoint_bundle(control_result, control)
    if (
        treatment_checkpoint["checkpoint_bundle_sha256"]
        == control_checkpoint["checkpoint_bundle_sha256"]
    ):
        raise SourceAdditionComparisonError(
            "source-addition checkpoints are duplicated"
        )
    treatment_report = validate_frozen_stream(treatment_stream, verify_sources=True)
    control_report = validate_frozen_stream(control_stream, verify_sources=True)
    development_report = validate_frozen_stream(development_stream, verify_sources=True)
    for value, label in (
        (treatment_training_source_sha256, "treatment training source"),
        (control_training_source_sha256, "control training source"),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise SourceAdditionComparisonError(f"{label} SHA256 differs")
    if (
        len(treatment_report.get("source_receipts", [])) != 1
        or len(control_report.get("source_receipts", [])) != 1
        or treatment_report["source_receipts"][0].get("sha256")
        != treatment_training_source_sha256
        or control_report["source_receipts"][0].get("sha256")
        != control_training_source_sha256
        or treatment_training_source_sha256 == control_training_source_sha256
    ):
        raise SourceAdditionComparisonError(
            "benchmark-ready training source lineage differs"
        )
    comparison = compare_payloads(
        treatment,
        control,
        treatment_stream=treatment_report,
        control_stream=control_report,
        development_stream=development_report,
    )
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "inputs": {
            "treatment_result": {
                "path": Path(treatment_result).name,
                "file_sha256": treatment_result_sha256,
                "receipt_sha256": treatment["receipt_sha256"],
            },
            "treatment_checkpoint": treatment_checkpoint,
            "control_result": {
                "path": Path(control_result).name,
                "file_sha256": control_result_sha256,
                "receipt_sha256": control["receipt_sha256"],
            },
            "control_checkpoint": control_checkpoint,
            "treatment_stream_identity_sha256": treatment_report[
                "ordered_stream_identity_sha256"
            ],
            "control_stream_identity_sha256": control_report[
                "ordered_stream_identity_sha256"
            ],
            "development_stream_identity_sha256": development_report[
                "ordered_stream_identity_sha256"
            ],
            "treatment_training_source_sha256": treatment_training_source_sha256,
            "control_training_source_sha256": control_training_source_sha256,
        },
        **comparison,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "training_authorized": False,
        "architecture_promotion_authorized": False,
        "four_b_training_authorized": False,
        "claim_limit": (
            "Held-out NLL evidence for one equal-token source addition only; "
            "real source-disjoint benchmark confirmation is mandatory."
        ),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment-result", type=Path, required=True)
    parser.add_argument("--control-result", type=Path, required=True)
    parser.add_argument("--treatment-stream", type=Path, required=True)
    parser.add_argument("--control-stream", type=Path, required=True)
    parser.add_argument("--development-stream", type=Path, required=True)
    parser.add_argument("--treatment-training-source-sha256", required=True)
    parser.add_argument("--control-training-source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare_paths(
        args.treatment_result,
        args.control_result,
        treatment_stream=args.treatment_stream,
        control_stream=args.control_stream,
        development_stream=args.development_stream,
        treatment_training_source_sha256=args.treatment_training_source_sha256,
        control_training_source_sha256=args.control_training_source_sha256,
    )
    _atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "receipt_sha256": payload["receipt_sha256"],
                "status": payload["status"],
                "heldout_nll_supported": payload[
                    "source_addition_supported_by_heldout_nll"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
