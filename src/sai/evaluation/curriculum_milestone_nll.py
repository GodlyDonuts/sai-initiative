"""Evaluate prospective Sai curriculum milestones on one frozen dev stream.

This evaluator is intentionally read-only.  It replays the complete terminal
training receipt, every prospectively declared model-only milestone, and the
terminal checkpoint before reporting phase-stratified held-out likelihood.
It performs no optimization and cannot authorize architecture selection or 4B
training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sai.data.token_stream import canonical_sha256, sha256_file, validate_frozen_stream
from sai.evaluation.short_screen_mc import (
    load_validated_model_state,
    validate_short_screen_result,
)
from sai.model.initialization import initialize_sai_model
from sai.model.reference import SaiCausalLM, exact_parameter_count
from sai.training.evaluate import ValidationResult, evaluate_nll
from sai.training.milestone import (
    load_validated_milestone_state,
    milestone_path,
    milestone_root,
    state_sha256,
    validate_milestone_population,
)
from sai.training.short_screen import (
    _development_batches,
    _development_strata,
    _prefix_bytes,
    load_bounded_config,
)

SCHEMA = "sai-curriculum-milestone-nll-v1"


class CurriculumMilestoneNLLError(RuntimeError):
    """A run, snapshot, stream, or phase likelihood differs."""


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CurriculumMilestoneNLLError(f"{field} must be a lowercase SHA256")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise CurriculumMilestoneNLLError("output parent or target is unsafe")
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
        raise CurriculumMilestoneNLLError("output already exists") from error
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CurriculumMilestoneNLLError(f"{field} must be a positive integer")
    return value


def _validate_curve_steps(
    steps: tuple[int, ...], *, optimizer_steps: int
) -> tuple[int, ...]:
    if (
        not steps
        or tuple(sorted(set(steps))) != steps
        or any(step <= 0 or step >= optimizer_steps for step in steps)
    ):
        raise CurriculumMilestoneNLLError("milestone steps differ")
    return (0, *steps, optimizer_steps)


def _phase_rows(result: ValidationResult) -> dict[str, dict[str, Any]]:
    payload = asdict(result)
    strata = payload.get("strata")
    if not isinstance(strata, dict) or not strata:
        raise CurriculumMilestoneNLLError("phase-stratified likelihood is required")
    return strata


def summarize_learning_curve(
    observations: list[dict[str, Any]],
    *,
    milestone_steps: tuple[int, ...],
    optimizer_steps: int,
) -> dict[str, Any]:
    """Validate and summarize one immutable acquisition/retention curve."""

    expected_steps = _validate_curve_steps(
        milestone_steps, optimizer_steps=optimizer_steps
    )
    if (
        not isinstance(observations, list)
        or len(observations) != len(expected_steps)
        or tuple(row.get("optimizer_step") for row in observations) != expected_steps
    ):
        raise CurriculumMilestoneNLLError("learning-curve steps differ")
    phase_order: tuple[str, ...] | None = None
    for row in observations:
        if not isinstance(row, dict):
            raise CurriculumMilestoneNLLError("learning-curve observation differs")
        strata = row.get("development_nll", {}).get("strata")
        if not isinstance(strata, dict) or not strata:
            raise CurriculumMilestoneNLLError("learning-curve strata differ")
        current_order = tuple(strata)
        if phase_order is None:
            phase_order = current_order
        if current_order != phase_order:
            raise CurriculumMilestoneNLLError("learning-curve strata differ")
        for phase_row in strata.values():
            value = phase_row.get("nll_per_target")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise CurriculumMilestoneNLLError("learning-curve score differs")
    if phase_order is None or len(phase_order) != len(milestone_steps) + 1:
        raise CurriculumMilestoneNLLError("phase and milestone counts differ")

    completion_steps = dict(
        zip(phase_order, (*milestone_steps, optimizer_steps), strict=True)
    )
    by_step = {row["optimizer_step"]: row for row in observations}
    phases: dict[str, Any] = {}
    for phase in phase_order:
        completion_step = completion_steps[phase]
        initial = by_step[0]["development_nll"]["strata"][phase]["nll_per_target"]
        completion = by_step[completion_step]["development_nll"]["strata"][phase][
            "nll_per_target"
        ]
        terminal = by_step[optimizer_steps]["development_nll"]["strata"][phase][
            "nll_per_target"
        ]
        phases[phase] = {
            "completion_step": completion_step,
            "initial_nll_per_target": initial,
            "completion_nll_per_target": completion,
            "terminal_nll_per_target": terminal,
            "acquisition_delta": completion - initial,
            "post_completion_forgetting_delta": terminal - completion,
            "terminal_delta_from_initialization": terminal - initial,
        }
    return {
        "observation_steps": list(expected_steps),
        "phase_order": list(phase_order),
        "phases": phases,
        "all_phases_acquired_by_completion": all(
            row["acquisition_delta"] < 0 for row in phases.values()
        ),
        "all_phases_better_than_initialization_at_terminal": all(
            row["terminal_delta_from_initialization"] < 0 for row in phases.values()
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Replay all model states and evaluate the exact same dev population."""

    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or torch.cuda.device_count() != 1
    ):
        raise CurriculumMilestoneNLLError("exactly one CUDA BF16 GPU is required")
    if args.output.exists() or args.output.is_symlink():
        raise CurriculumMilestoneNLLError("output already exists")
    if sha256_file(args.geometry) != _sha256(args.geometry_sha256, "geometry SHA256"):
        raise CurriculumMilestoneNLLError("geometry bytes differ")

    config, geometry_row = load_bounded_config(args.geometry, args.family, "100m")
    result, bindings = validate_short_screen_result(
        args.short_screen_result,
        expected_sha256=_sha256(
            args.short_screen_result_sha256, "short-screen result SHA256"
        ),
        config=config,
        family=args.family,
        geometry_parameter_count=geometry_row["parameter_ledger"]["total"],
    )
    if result.get("mechanics_only") is not False:
        raise CurriculumMilestoneNLLError("terminal run is not scientific training")
    raw_steps = result.get("milestone_steps")
    if not isinstance(raw_steps, list) or any(
        isinstance(step, bool) or not isinstance(step, int) for step in raw_steps
    ):
        raise CurriculumMilestoneNLLError("terminal milestone declaration differs")
    milestone_steps = tuple(raw_steps)
    optimizer_steps = _positive_integer(
        result.get("optimizer", {}).get("optimizer_steps"), "optimizer steps"
    )
    expected_curve_steps = _validate_curve_steps(
        milestone_steps, optimizer_steps=optimizer_steps
    )

    training = validate_frozen_stream(args.training_stream, verify_sources=True)
    development = validate_frozen_stream(args.development_stream, verify_sources=True)
    if (
        training.get("ordered_stream_identity_sha256")
        != _sha256(args.training_stream_identity, "training stream identity")
        or training["ordered_stream_identity_sha256"]
        != result.get("training_stream_identity_sha256")
        or development.get("ordered_stream_identity_sha256")
        != _sha256(args.development_stream_identity, "development stream identity")
        or development["ordered_stream_identity_sha256"]
        != result.get("development_stream_identity_sha256")
    ):
        raise CurriculumMilestoneNLLError("training/development stream differs")
    development_sequences = _positive_integer(
        result.get("development_sequences"), "development sequences"
    )
    development_batch_size = _positive_integer(
        result.get("development_batch_size_sequences"),
        "development batch size",
    )
    development_bytes = _prefix_bytes(development, development_sequences)
    development_strata = _development_strata(development, development_sequences)
    if development_strata is None:
        raise CurriculumMilestoneNLLError("development phase strata are absent")

    model = SaiCausalLM(config, delta_backend=result["delta_backend"])
    initialization = initialize_sai_model(model, seed=result["initialization_seed"])
    if initialization != result.get("initialization"):
        raise CurriculumMilestoneNLLError("initialization receipt differs")
    if exact_parameter_count(model) != geometry_row["parameter_ledger"]["total"]:
        raise CurriculumMilestoneNLLError("instantiated parameter count differs")
    model = model.to(device="cuda")

    def evaluate(step: int, checkpoint: dict[str, Any]) -> dict[str, Any]:
        score = evaluate_nll(
            model,
            _development_batches(
                args.development_stream,
                development["ordered_stream_identity_sha256"],
                sequences=development_sequences,
                batch_size=development_batch_size,
            ),
            stream_identity_sha256=development["ordered_stream_identity_sha256"],
            expected_sequences=development_sequences,
            admitted_utf8_bytes=development_bytes,
            benchmark_disjoint=True,
            autocast_dtype=torch.bfloat16,
            sequence_strata=development_strata,
        )
        _phase_rows(score)
        return {
            "optimizer_step": step,
            "model_state_sha256": state_sha256(model.state_dict()),
            "checkpoint": checkpoint,
            "development_nll": asdict(score),
        }

    observations = [
        evaluate(
            0,
            {
                "scope": "deterministically_reconstructed_initialization",
                "initialization_policy_sha256": result["initialization_policy_sha256"],
                "initialization_seed": result["initialization_seed"],
            },
        )
    ]
    descriptors = result.get("milestone_checkpoints")
    root = milestone_root(args.checkpoint)
    expected_descriptors = validate_milestone_population(
        root,
        expected_steps=milestone_steps,
        expected_bindings=bindings,
        maximum_completed_step=optimizer_steps,
    )
    if descriptors != expected_descriptors:
        raise CurriculumMilestoneNLLError("milestone descriptors differ")
    for descriptor in expected_descriptors:
        step = descriptor["optimizer_step"]
        load_validated_milestone_state(
            milestone_path(root, step),
            model=model,
            expected_bindings=bindings,
            expected_descriptor=descriptor,
        )
        observations.append(evaluate(step, descriptor))

    checkpoint_observation = load_validated_model_state(
        args.checkpoint,
        args.checkpoint_manifest,
        model=model,
        expected_bindings=bindings,
        expected_descriptor=result.get("checkpoint"),
        expected_counters=result.get("counters"),
        expected_cursor=result.get("stream_cursor"),
        expected_final_state_sha256=result.get("final_state_sha256"),
    )
    observations.append(evaluate(optimizer_steps, checkpoint_observation))
    if tuple(row["optimizer_step"] for row in observations) != expected_curve_steps:
        raise CurriculumMilestoneNLLError("evaluated milestone sequence differs")

    summary = summarize_learning_curve(
        observations,
        milestone_steps=milestone_steps,
        optimizer_steps=optimizer_steps,
    )
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "training_run": {
            "result_path": args.short_screen_result.name,
            "result_file_sha256": args.short_screen_result_sha256,
            "run_sha256": result["run_sha256"],
            "model_sha256": result["model_sha256"],
            "training_stream_identity_sha256": training[
                "ordered_stream_identity_sha256"
            ],
        },
        "development_stream": {
            "ordered_stream_identity_sha256": development[
                "ordered_stream_identity_sha256"
            ],
            "sequences": development_sequences,
            "admitted_utf8_bytes": development_bytes,
            "phase_strata": [
                {"phase": phase, "sequences": sequences, "utf8_bytes": byte_count}
                for phase, sequences, byte_count in development_strata
            ],
        },
        "milestone_steps": list(milestone_steps),
        "observations": observations,
        "learning_curve": summary,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "training_authorized": False,
        "architecture_promotion_authorized": False,
        "four_b_training_authorized": False,
        "claim_limit": (
            "Acquisition/forgetting evidence for one frozen training run only; "
            "requires a matched order control and real benchmark confirmation."
        ),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-sha256", required=True)
    parser.add_argument("--family", choices=("gated_gqa",), required=True)
    parser.add_argument("--short-screen-result", type=Path, required=True)
    parser.add_argument("--short-screen-result-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--training-stream", type=Path, required=True)
    parser.add_argument("--training-stream-identity", required=True)
    parser.add_argument("--development-stream", type=Path, required=True)
    parser.add_argument("--development-stream-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    print(
        json.dumps(
            {
                "receipt_sha256": payload["receipt_sha256"],
                "status": payload["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
