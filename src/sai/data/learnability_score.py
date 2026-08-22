"""Produce immutable weak/strong per-sequence learnability scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from sai.data.curriculum_control import _record_sha256, _Records
from sai.data.learnability_curriculum import SCORE_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file, validate_frozen_stream
from sai.evaluation.scale_checkpoint import (
    EVALUATION_SCALES,
    FAMILIES,
    load_evaluation_config,
)
from sai.evaluation.short_screen_mc import (
    load_validated_model_state,
    validate_short_screen_result,
)
from sai.model.reference import SaiCausalLM, exact_parameter_count
from sai.training.milestone import (
    MilestoneSnapshotError,
    load_validated_milestone_state,
    state_sha256,
)
from sai.training.runner import tensorize_stream_batch
from sai.training.stream import ReceiptBoundTokenStream

SCHEMA = "sai-model-centric-learnability-score-population-v1"
OUTPUT_NAME = "scores.jsonl"
RECEIPT_NAME = "score_receipt.json"
_MAX_RECEIPT_BYTES = 16 << 20
_TOP_KEYS = {
    "schema",
    "status",
    "training_authorized",
    "four_b_training_authorized",
    "target_stream",
    "probe_training_stream",
    "exact_record_independence",
    "probe",
    "scoring",
    "scores",
    "model_state_unchanged",
    "rng_state_unchanged",
    "limitations",
    "receipt_sha256",
}
_STREAM_KEYS = {
    "path",
    "receipt_file_sha256",
    "ordered_stream_identity_sha256",
    "source_manifest_sha256",
    "tokenizer_identity_sha256",
    "sequences",
    "sequence_length",
}
_INDEPENDENCE_KEYS = {
    "method",
    "target_unique_records",
    "probe_unique_records",
    "exact_record_overlap_count",
}
_PROBE_KEYS = {
    "family",
    "scale",
    "parameter_count",
    "config_sha256",
    "model_sha256",
    "run_sha256",
    "result",
    "weak_milestone",
    "strong_checkpoint",
}
_FILE_KEYS = {"path", "bytes", "sha256"}
_RESULT_KEYS = _FILE_KEYS | {"receipt_sha256"}
_STRONG_KEYS = {
    "checkpoint",
    "manifest",
    "final_state_sha256",
    "optimizer_step",
}
_SCORING_KEYS = {
    "method",
    "evaluator_sha256",
    "runtime_receipt",
    "execution_dtype",
    "device_type",
    "device_name",
    "batch_size_sequences",
    "sequences",
    "targets",
    "inference_mode",
    "optimizer_steps",
    "backward_calls",
}
_SCORES_KEYS = _FILE_KEYS | {"rows", "ordered_population_sha256"}
_MAX_SCORE_BYTES = 512 << 20
_LIMITATIONS = [
    "exact_record_disjointness_does_not_prove_near_duplicate_disjointness",
    "probe_preferences_may_be_checkpoint_specific",
    "score_population_does_not_authorize_training_or_4b",
]


class LearnabilityScoreError(RuntimeError):
    """A probe, target stream, score, model invariant, or receipt differs."""


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise LearnabilityScoreError(f"{label} differs")
    return value


def _positive(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LearnabilityScoreError(f"{label} differs")
    return value


def _regular_bytes(path: Path, label: str, *, maximum_bytes: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise LearnabilityScoreError(f"{label} is missing or unsafe") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise LearnabilityScoreError(f"{label} is missing or unsafe")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1 << 20, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    encoded = b"".join(chunks)

    def identity(row: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            row.st_dev,
            row.st_ino,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
        )

    if len(encoded) != before.st_size or identity(before) != identity(after):
        raise LearnabilityScoreError(f"{label} changed while reading")
    return encoded


def _file_descriptor(path: Path, encoded: bytes | None = None) -> dict[str, Any]:
    if encoded is None:
        encoded = _regular_bytes(path, "evidence artifact", maximum_bytes=1 << 34)
    return {
        "path": str(path.resolve()),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _stream_descriptor(root: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(root.resolve()),
        "receipt_file_sha256": sha256_file(root / "stream_receipt.json"),
        "ordered_stream_identity_sha256": report["ordered_stream_identity_sha256"],
        "source_manifest_sha256": report["source_manifest_sha256"],
        "tokenizer_identity_sha256": report["tokenizer_identity_sha256"],
        "sequences": report["sequences"],
        "sequence_length": report["sequence_length"],
    }


def exact_record_independence(
    target_root: Path,
    target_report: dict[str, Any],
    probe_root: Path,
    probe_report: dict[str, Any],
) -> dict[str, Any]:
    """Prove no exact packed token-and-boundary record appears in both streams."""

    with _Records(target_root, target_report) as target_records:
        target = {
            _record_sha256(*target_records.record(index))
            for index in range(target_report["sequences"])
        }
    with _Records(probe_root, probe_report) as probe_records:
        probe = {
            _record_sha256(*probe_records.record(index))
            for index in range(probe_report["sequences"])
        }
    overlap = len(target & probe)
    if overlap:
        raise LearnabilityScoreError("probe and target packed records overlap")
    return {
        "method": "exact_sha256_of_tokens_and_boundary_mask",
        "target_unique_records": len(target),
        "probe_unique_records": len(probe),
        "exact_record_overlap_count": 0,
    }


def _versions(model: nn.Module) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, tensor._version)
        for name, tensor in sorted(model.state_dict(keep_vars=True).items())
    )


def _round_microunits(value: float) -> int:
    if not math.isfinite(value) or value <= 0:
        raise LearnabilityScoreError("normalized sequence NLL is nonfinite")
    return int(math.floor(value * 1_000_000 + 0.5))


def score_model_pair(
    weak_model: SaiCausalLM,
    strong_model: SaiCausalLM,
    stream_root: Path,
    *,
    expected_stream_identity_sha256: str,
    batch_size: int,
    device: torch.device | str,
    autocast_dtype: torch.dtype | None,
) -> tuple[list[dict[str, Any]], int]:
    """Score every exact packed sequence without gradients or state mutation."""

    if (
        not isinstance(weak_model, SaiCausalLM)
        or not isinstance(strong_model, SaiCausalLM)
        or weak_model.config != strong_model.config
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise LearnabilityScoreError("learnability scorer model contract differs")
    device = torch.device(device)
    if autocast_dtype is not None and (
        device.type != "cuda" or autocast_dtype is not torch.bfloat16
    ):
        raise LearnabilityScoreError("learnability execution dtype differs")
    report = validate_frozen_stream(stream_root, verify_sources=True)
    if (
        report["ordered_stream_identity_sha256"]
        != _sha256(expected_stream_identity_sha256, "target stream identity")
        or report["vocab_size"] != weak_model.config.vocab_size
    ):
        raise LearnabilityScoreError("learnability target stream differs")
    if any(
        next(model.parameters()).device != device
        for model in (weak_model, strong_model)
    ):
        raise LearnabilityScoreError("learnability model device differs")
    versions_before = (_versions(weak_model), _versions(strong_model))
    states_before = (
        state_sha256(weak_model.state_dict()),
        state_sha256(strong_model.state_dict()),
    )
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_before = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if device.type == "cuda"
        else ()
    )
    training_modes = (weak_model.training, strong_model.training)
    weak_model.eval()
    strong_model.eval()
    stream = ReceiptBoundTokenStream(
        stream_root,
        expected_ordered_stream_identity_sha256=expected_stream_identity_sha256,
        verify_sources=True,
    )
    rows = []
    targets_total = 0
    autocast = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    try:
        with _Records(stream_root, report) as records, torch.inference_mode(), autocast:
            while stream.remaining_sequences:
                current = min(batch_size, stream.remaining_sequences)
                raw = stream.next_batch(current)
                batch = tensorize_stream_batch(raw, device=device)
                losses_by_model = []
                for model in (weak_model, strong_model):
                    logits = model(batch.input_ids, batch.segment_ids)
                    expected_shape = (
                        *batch.input_ids.shape,
                        model.config.vocab_size,
                    )
                    if logits.shape != expected_shape:
                        raise LearnabilityScoreError(
                            "learnability logit geometry differs"
                        )
                    losses = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]).float(),
                        batch.target_ids.reshape(-1),
                        reduction="none",
                        ignore_index=-100,
                    ).reshape(batch.target_ids.shape)
                    if not bool(torch.isfinite(losses).all().item()):
                        raise LearnabilityScoreError("learnability NLL is nonfinite")
                    losses_by_model.append(losses)
                    del logits
                for local_index in range(current):
                    mask = batch.target_mask[local_index]
                    target_count = int(mask.sum().item())
                    if target_count <= 0:
                        raise LearnabilityScoreError(
                            "learnability sequence has no targets"
                        )
                    weak_nll = _round_microunits(
                        float(losses_by_model[0][local_index][mask].sum())
                        / target_count
                    )
                    strong_nll = _round_microunits(
                        float(losses_by_model[1][local_index][mask].sum())
                        / target_count
                    )
                    sequence_index = raw.first_sequence + local_index
                    rows.append(
                        {
                            "schema": SCORE_SCHEMA,
                            "sequence_index": sequence_index,
                            "record_sha256": _record_sha256(
                                *records.record(sequence_index)
                            ).hex(),
                            "target_count": target_count,
                            "weak_nll_microunits_per_target": weak_nll,
                            "strong_nll_microunits_per_target": strong_nll,
                            "preference_delta_microunits": weak_nll - strong_nll,
                        }
                    )
                    targets_total += target_count
    finally:
        weak_model.train(training_modes[0])
        strong_model.train(training_modes[1])
    if len(rows) != report["sequences"] or targets_total <= 0:
        raise LearnabilityScoreError("learnability score coverage differs")
    if (
        (_versions(weak_model), _versions(strong_model)) != versions_before
        or (
            state_sha256(weak_model.state_dict()),
            state_sha256(strong_model.state_dict()),
        )
        != states_before
        or not torch.equal(torch.get_rng_state(), cpu_rng_before)
        or (
            device.type == "cuda"
            and any(
                not torch.equal(before, after)
                for before, after in zip(
                    cuda_rng_before, torch.cuda.get_rng_state_all(), strict=True
                )
            )
        )
        or any(
            parameter.grad is not None
            for model in (weak_model, strong_model)
            for parameter in model.parameters()
        )
    ):
        raise LearnabilityScoreError("learnability scoring mutated model or RNG state")
    return rows, targets_total


def _encoded_rows(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _validate_rows(encoded: bytes, expected_rows: int) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in encoded.decode("utf-8").splitlines()]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LearnabilityScoreError("learnability score JSONL differs") from error
    if len(rows) != expected_rows:
        raise LearnabilityScoreError("learnability score row count differs")
    expected_keys = {
        "schema",
        "sequence_index",
        "record_sha256",
        "target_count",
        "weak_nll_microunits_per_target",
        "strong_nll_microunits_per_target",
        "preference_delta_microunits",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise LearnabilityScoreError("learnability score fields differ")
        weak = _positive(row["weak_nll_microunits_per_target"], "weak NLL")
        strong = _positive(row["strong_nll_microunits_per_target"], "strong NLL")
        if (
            row["schema"] != SCORE_SCHEMA
            or row["sequence_index"] != index
            or _positive(row["target_count"], "target count") <= 0
            or _sha256(row["record_sha256"], "record identity") != row["record_sha256"]
            or isinstance(row["preference_delta_microunits"], bool)
            or not isinstance(row["preference_delta_microunits"], int)
            or row["preference_delta_microunits"] != weak - strong
        ):
            raise LearnabilityScoreError("learnability score values differ")
    return rows


def validate_score_population(root: Path) -> dict[str, Any]:
    """Validate the immutable two-file score population and its self-hash."""

    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise LearnabilityScoreError("learnability score root is missing or unsafe")
    observed = {entry.name for entry in os.scandir(root)}
    if observed != {OUTPUT_NAME, RECEIPT_NAME}:
        raise LearnabilityScoreError("learnability score root membership differs")
    receipt_bytes = _regular_bytes(
        root / RECEIPT_NAME,
        "learnability score receipt",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    score_bytes = _regular_bytes(
        root / OUTPUT_NAME,
        "learnability scores",
        maximum_bytes=_MAX_SCORE_BYTES,
    )
    try:
        payload = json.loads(receipt_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LearnabilityScoreError(
            "learnability score receipt JSON differs"
        ) from error
    if not isinstance(payload, dict) or set(payload) != _TOP_KEYS:
        raise LearnabilityScoreError("learnability score receipt fields differ")
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    if (
        payload["schema"] != SCHEMA
        or payload["status"] != "complete"
        or payload["training_authorized"] is not False
        or payload["four_b_training_authorized"] is not False
        or payload["receipt_sha256"] != canonical_sha256(unsigned)
        or payload["model_state_unchanged"] is not True
        or payload["rng_state_unchanged"] is not True
    ):
        raise LearnabilityScoreError("learnability score receipt identity differs")
    for name in ("target_stream", "probe_training_stream"):
        stream = payload[name]
        if not isinstance(stream, dict) or set(stream) != _STREAM_KEYS:
            raise LearnabilityScoreError("learnability stream descriptor differs")
        if not isinstance(stream["path"], str) or not stream["path"]:
            raise LearnabilityScoreError("learnability stream descriptor differs")
        for field in (
            "receipt_file_sha256",
            "ordered_stream_identity_sha256",
            "source_manifest_sha256",
            "tokenizer_identity_sha256",
        ):
            _sha256(stream[field], field)
        _positive(stream["sequences"], "stream sequences")
        _positive(stream["sequence_length"], "stream sequence length")
    if (
        payload["target_stream"]["ordered_stream_identity_sha256"]
        == payload["probe_training_stream"]["ordered_stream_identity_sha256"]
        or payload["target_stream"]["tokenizer_identity_sha256"]
        != payload["probe_training_stream"]["tokenizer_identity_sha256"]
    ):
        raise LearnabilityScoreError("learnability stream relationship differs")
    independence = payload["exact_record_independence"]
    if (
        not isinstance(independence, dict)
        or set(independence) != _INDEPENDENCE_KEYS
        or independence["method"] != "exact_sha256_of_tokens_and_boundary_mask"
        or independence["exact_record_overlap_count"] != 0
        or _positive(independence["target_unique_records"], "target records") <= 0
        or _positive(independence["probe_unique_records"], "probe records") <= 0
        or independence["target_unique_records"] > payload["target_stream"]["sequences"]
        or independence["probe_unique_records"]
        > payload["probe_training_stream"]["sequences"]
    ):
        raise LearnabilityScoreError("learnability independence receipt differs")
    probe = payload["probe"]
    if not isinstance(probe, dict) or set(probe) != _PROBE_KEYS:
        raise LearnabilityScoreError("learnability probe descriptor differs")
    if probe["scale"] not in EVALUATION_SCALES or probe["family"] not in FAMILIES:
        raise LearnabilityScoreError("learnability probe scale differs")
    _positive(probe["parameter_count"], "probe parameters")
    for field in ("config_sha256", "model_sha256", "run_sha256"):
        _sha256(probe[field], field)
    if not isinstance(probe["result"], dict) or set(probe["result"]) != _RESULT_KEYS:
        raise LearnabilityScoreError("learnability result descriptor differs")
    result_descriptor = probe["result"]
    if not isinstance(result_descriptor["path"], str) or not result_descriptor["path"]:
        raise LearnabilityScoreError("learnability result descriptor differs")
    _positive(result_descriptor["bytes"], "result bytes")
    _sha256(result_descriptor["sha256"], "result artifact")
    _sha256(result_descriptor["receipt_sha256"], "result receipt")
    descriptor = probe["weak_milestone"]
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "path",
        "bytes",
        "sha256",
        "optimizer_step",
        "sequences",
        "targets",
        "model_state_sha256",
    }:
        raise LearnabilityScoreError("learnability milestone differs")
    if not isinstance(descriptor["path"], str) or not descriptor["path"]:
        raise LearnabilityScoreError("learnability milestone differs")
    for field in ("bytes", "optimizer_step", "sequences", "targets"):
        _positive(descriptor[field], f"milestone {field}")
    for field in ("sha256", "model_state_sha256"):
        _sha256(descriptor[field], f"milestone {field}")
    strong = probe["strong_checkpoint"]
    if not isinstance(strong, dict) or set(strong) != _STRONG_KEYS:
        raise LearnabilityScoreError("learnability strong checkpoint differs")
    if not all(
        isinstance(strong[field], dict) and set(strong[field]) == _FILE_KEYS
        for field in ("checkpoint", "manifest")
    ):
        raise LearnabilityScoreError("learnability strong checkpoint differs")
    for artifact in (strong["checkpoint"], strong["manifest"]):
        if not isinstance(artifact["path"], str) or not artifact["path"]:
            raise LearnabilityScoreError("learnability strong checkpoint differs")
        _positive(artifact["bytes"], "strong checkpoint bytes")
        _sha256(artifact["sha256"], "strong checkpoint artifact")
    _sha256(strong["final_state_sha256"], "strong final state")
    _positive(strong["optimizer_step"], "strong optimizer step")
    if strong["optimizer_step"] <= descriptor["optimizer_step"]:
        raise LearnabilityScoreError("learnability checkpoint order differs")
    scoring = payload["scoring"]
    if (
        not isinstance(scoring, dict)
        or set(scoring) != _SCORING_KEYS
        or scoring["method"] != "weak_minus_strong_normalized_nll_microunits"
        or scoring["execution_dtype"] != "bfloat16"
        or scoring["device_type"] != "cuda"
        or not isinstance(scoring["device_name"], str)
        or not scoring["device_name"]
        or scoring["inference_mode"] is not True
        or scoring["optimizer_steps"] != 0
        or scoring["backward_calls"] != 0
    ):
        raise LearnabilityScoreError("learnability scoring receipt differs")
    _positive(scoring["batch_size_sequences"], "score batch size")
    _positive(scoring["sequences"], "score sequences")
    _positive(scoring["targets"], "score targets")
    _sha256(scoring["evaluator_sha256"], "evaluator")
    if (
        not isinstance(scoring["runtime_receipt"], dict)
        or set(scoring["runtime_receipt"]) != _FILE_KEYS
    ):
        raise LearnabilityScoreError("learnability runtime receipt differs")
    runtime = scoring["runtime_receipt"]
    if not isinstance(runtime["path"], str) or not runtime["path"]:
        raise LearnabilityScoreError("learnability runtime receipt differs")
    _positive(runtime["bytes"], "runtime receipt bytes")
    _sha256(runtime["sha256"], "runtime receipt artifact")
    scores = payload["scores"]
    if not isinstance(scores, dict) or set(scores) != _SCORES_KEYS:
        raise LearnabilityScoreError("learnability score output descriptor differs")
    rows = _validate_rows(score_bytes, scores["rows"])
    if (
        scores["path"] != OUTPUT_NAME
        or scores["bytes"] != len(score_bytes)
        or scores["sha256"] != hashlib.sha256(score_bytes).hexdigest()
        or scores["ordered_population_sha256"] != canonical_sha256(rows)
        or scores["rows"] != scoring["sequences"]
        or scores["rows"] != payload["target_stream"]["sequences"]
        or any(
            row["target_count"] >= payload["target_stream"]["sequence_length"]
            for row in rows
        )
        or sum(row["target_count"] for row in rows) != scoring["targets"]
        or payload["limitations"] != _LIMITATIONS
    ):
        raise LearnabilityScoreError("learnability score output differs")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate one probe trajectory and score one disjoint packed stream."""

    if args.output.exists() or args.output.is_symlink():
        raise LearnabilityScoreError("learnability score output already exists")
    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.device_count() != 1
    ):
        raise LearnabilityScoreError("exactly one CUDA BF16 GPU is required")
    geometry_sha256 = sha256_file(args.geometry)
    if geometry_sha256 != _sha256(args.geometry_sha256, "geometry"):
        raise LearnabilityScoreError("geometry bytes differ")
    config, geometry_row = load_evaluation_config(
        args.geometry, args.family, args.scale
    )
    result, bindings = validate_short_screen_result(
        args.probe_result,
        expected_sha256=args.probe_result_sha256,
        config=config,
        family=args.family,
        geometry_parameter_count=geometry_row["parameter_ledger"]["total"],
        scale=args.scale,
    )
    target_report = validate_frozen_stream(args.target_stream, verify_sources=True)
    probe_report = validate_frozen_stream(
        args.probe_training_stream, verify_sources=True
    )
    if (
        target_report["ordered_stream_identity_sha256"]
        != _sha256(args.target_stream_identity, "target stream identity")
        or probe_report["ordered_stream_identity_sha256"]
        != _sha256(args.probe_training_stream_identity, "probe stream identity")
        or probe_report["ordered_stream_identity_sha256"]
        != result["training_stream_identity_sha256"]
        or target_report["ordered_stream_identity_sha256"]
        == probe_report["ordered_stream_identity_sha256"]
        or target_report["tokenizer_identity_sha256"]
        != probe_report["tokenizer_identity_sha256"]
        or target_report["vocab_size"] != config.vocab_size
        or probe_report["vocab_size"] != config.vocab_size
    ):
        raise LearnabilityScoreError("probe or target stream identity differs")
    independence = exact_record_independence(
        args.target_stream,
        target_report,
        args.probe_training_stream,
        probe_report,
    )
    descriptors = result.get("milestone_checkpoints")
    if not isinstance(descriptors, list):
        raise LearnabilityScoreError("probe has no milestone population")
    weak_rows = [
        descriptor
        for descriptor in descriptors
        if isinstance(descriptor, dict)
        and descriptor.get("optimizer_step") == args.weak_milestone_step
    ]
    if (
        len(weak_rows) != 1
        or args.weak_milestone_step >= result["counters"]["optimizer_steps"]
    ):
        raise LearnabilityScoreError("weak milestone identity differs")
    weak_descriptor = weak_rows[0]
    expected_weak_path = (
        args.strong_checkpoint.with_name(f"{args.strong_checkpoint.name}.milestones")
        / weak_descriptor["path"]
    )
    if args.weak_milestone.resolve() != expected_weak_path.resolve():
        raise LearnabilityScoreError("weak milestone path differs")
    weak_model = SaiCausalLM(config, delta_backend=result["delta_backend"])
    strong_model = SaiCausalLM(config, delta_backend=result["delta_backend"])
    strong_observation = load_validated_model_state(
        args.strong_checkpoint,
        args.strong_checkpoint_manifest,
        model=strong_model,
        expected_bindings=bindings,
        expected_descriptor=result["checkpoint"],
        expected_counters=result["counters"],
        expected_cursor=result["stream_cursor"],
        expected_final_state_sha256=result["final_state_sha256"],
    )
    try:
        load_validated_milestone_state(
            args.weak_milestone,
            model=weak_model,
            expected_bindings=bindings,
            expected_descriptor=weak_descriptor,
        )
    except MilestoneSnapshotError as error:
        raise LearnabilityScoreError("weak milestone validation failed") from error
    if (
        exact_parameter_count(weak_model) != geometry_row["parameter_ledger"]["total"]
        or exact_parameter_count(strong_model)
        != geometry_row["parameter_ledger"]["total"]
        or weak_descriptor["model_state_sha256"] == result["final_state_sha256"]
    ):
        raise LearnabilityScoreError("probe model state differs")
    runtime_bytes = _regular_bytes(
        args.runtime_receipt,
        "runtime receipt",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    if hashlib.sha256(runtime_bytes).hexdigest() != _sha256(
        args.runtime_receipt_sha256, "runtime receipt"
    ):
        raise LearnabilityScoreError("runtime receipt bytes differ")
    device = torch.device("cuda")
    weak_model = weak_model.to(device=device)
    strong_model = strong_model.to(device=device)
    rows, targets = score_model_pair(
        weak_model,
        strong_model,
        args.target_stream,
        expected_stream_identity_sha256=args.target_stream_identity,
        batch_size=args.batch_size,
        device=device,
        autocast_dtype=torch.bfloat16,
    )
    score_bytes = _encoded_rows(rows)
    result_bytes = _regular_bytes(
        args.probe_result,
        "probe result",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    manifest_bytes = _regular_bytes(
        args.strong_checkpoint_manifest,
        "strong checkpoint manifest",
        maximum_bytes=_MAX_RECEIPT_BYTES,
    )
    evaluator_sha256 = sha256_file(Path(__file__))
    stage = args.output.parent / f".{args.output.name}.partial.{uuid.uuid4().hex}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir(mode=0o700)
    try:
        score_path = stage / OUTPUT_NAME
        with score_path.open("xb") as handle:
            handle.write(score_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        payload = {
            "schema": SCHEMA,
            "status": "complete",
            "training_authorized": False,
            "four_b_training_authorized": False,
            "target_stream": _stream_descriptor(args.target_stream, target_report),
            "probe_training_stream": _stream_descriptor(
                args.probe_training_stream, probe_report
            ),
            "exact_record_independence": independence,
            "probe": {
                "family": args.family,
                "scale": args.scale,
                "parameter_count": geometry_row["parameter_ledger"]["total"],
                "config_sha256": result["config_sha256"],
                "model_sha256": result["model_sha256"],
                "run_sha256": result["run_sha256"],
                "result": {
                    **_file_descriptor(args.probe_result, result_bytes),
                    "receipt_sha256": result["receipt_sha256"],
                },
                "weak_milestone": weak_descriptor,
                "strong_checkpoint": {
                    "checkpoint": {
                        "path": str(args.strong_checkpoint.resolve()),
                        "bytes": strong_observation["checkpoint_bytes"],
                        "sha256": strong_observation["checkpoint_sha256"],
                    },
                    "manifest": _file_descriptor(
                        args.strong_checkpoint_manifest, manifest_bytes
                    ),
                    "final_state_sha256": strong_observation["final_state_sha256"],
                    "optimizer_step": result["counters"]["optimizer_steps"],
                },
            },
            "scoring": {
                "method": "weak_minus_strong_normalized_nll_microunits",
                "evaluator_sha256": evaluator_sha256,
                "runtime_receipt": _file_descriptor(
                    args.runtime_receipt, runtime_bytes
                ),
                "execution_dtype": "bfloat16",
                "device_type": "cuda",
                "device_name": torch.cuda.get_device_name(0),
                "batch_size_sequences": args.batch_size,
                "sequences": len(rows),
                "targets": targets,
                "inference_mode": True,
                "optimizer_steps": 0,
                "backward_calls": 0,
            },
            "scores": {
                "path": OUTPUT_NAME,
                "bytes": len(score_bytes),
                "sha256": hashlib.sha256(score_bytes).hexdigest(),
                "rows": len(rows),
                "ordered_population_sha256": canonical_sha256(rows),
            },
            "model_state_unchanged": True,
            "rng_state_unchanged": True,
            "limitations": list(_LIMITATIONS),
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        receipt_path = stage / RECEIPT_NAME
        with receipt_path.open("x") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        validate_score_population(stage)
        os.replace(stage, args.output)
        return payload
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--scale", choices=EVALUATION_SCALES, default="100m")
    parser.add_argument("--probe-result", type=Path, required=True)
    parser.add_argument("--probe-result-sha256", required=True)
    parser.add_argument("--strong-checkpoint", type=Path, required=True)
    parser.add_argument("--strong-checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--weak-milestone", type=Path, required=True)
    parser.add_argument("--weak-milestone-step", type=int, required=True)
    parser.add_argument("--probe-training-stream", type=Path, required=True)
    parser.add_argument("--probe-training-stream-identity", required=True)
    parser.add_argument("--target-stream", type=Path, required=True)
    parser.add_argument("--target-stream-identity", required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--runtime-receipt-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _positive(args.batch_size, "batch size")
    _positive(args.weak_milestone_step, "weak milestone step")
    payload = run(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "rows": payload["scores"]["rows"],
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
