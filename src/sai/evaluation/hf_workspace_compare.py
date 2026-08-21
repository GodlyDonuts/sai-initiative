"""Compare parent, recurrent, and matched-reset Qwen development results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

import torch

from sai.data.token_stream import canonical_sha256
from sai.evaluation.development_mc import SCHEMA as RESULT_SCHEMA
from sai.training.hf_workspace_screen import SCHEMA as TRAINING_SCHEMA

SCHEMA = "sai-qwen35-0p8b-matched-workspace-comparison-v1"
BENCHMARKS = ("mmlu_pro", "musr")
BOOTSTRAP_REPLICATES = 10_000
MINIMUM_MACRO_GAIN = 0.005
MAXIMUM_BOARD_REGRESSION = -0.005
COMMON_WORKSPACE_FIELDS = (
    "parent_snapshot_tree_sha256",
    "workspace_config_sha256",
    "workspace_parameter_count",
    "workspace_initial_state_sha256",
    "training_stream_identity_sha256",
    "training_source_manifest_sha256",
    "training_sequences",
    "training_utf8_bytes",
    "optimizer",
    "code_sha256",
    "environment_sha256",
)


class HFWorkspaceComparisonError(RuntimeError):
    """One evaluation result or paired comparison differs."""


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise HFWorkspaceComparisonError("comparison input is missing or unsafe")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_result(
    path: Path, *, benchmark: str, evidence: str
) -> tuple[dict[str, Any], str]:
    file_sha256 = _sha256_file(path)
    try:
        result = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFWorkspaceComparisonError("comparison input is unreadable") from error
    if not isinstance(result, dict):
        raise HFWorkspaceComparisonError("comparison input differs")
    unsigned = dict(result)
    claimed = unsigned.pop("receipt_sha256", None)
    rows = result.get("rows")
    coverage = result.get("coverage")
    aggregate = result.get("aggregate")
    correct = (
        sum(row.get("correct") is True for row in rows)
        if isinstance(rows, list)
        else -1
    )
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "complete"
        or result.get("benchmark") != benchmark
        or result.get("development_only") is not True
        or result.get("official_benchmark_result") is not False
        or result.get("public_terminal_result") is not False
        or result.get("architecture_promotion_allowed") is not False
        or claimed != canonical_sha256(unsigned)
        or not isinstance(rows, list)
        or not rows
        or coverage != {"expected_rows": len(rows), "scored_rows": len(rows)}
        or not isinstance(aggregate, dict)
        or aggregate.get("rows") != len(rows)
        or aggregate.get("correct") != correct
        or aggregate.get("accuracy") != correct / len(rows)
    ):
        raise HFWorkspaceComparisonError("comparison result receipt differs")
    if evidence == "parent":
        parent = result.get("parent_evidence")
        if not isinstance(parent, dict) or parent.get("unchanged_parent") is not True:
            raise HFWorkspaceComparisonError("unchanged-parent evidence differs")
    else:
        workspace = result.get("workspace_evidence")
        if (
            not isinstance(workspace, dict)
            or workspace.get("state_mode") != evidence
            or workspace.get("matched_comparison") is not True
            or workspace.get("source_disjoint_from_factor_training") is not True
            or workspace.get("four_b_training_executed") is not False
        ):
            raise HFWorkspaceComparisonError("workspace evaluation evidence differs")
    return result, file_sha256


def _paired_rows(
    parent: dict[str, Any],
    recurrent: dict[str, Any],
    reset: dict[str, Any],
) -> tuple[list[float], list[float]]:
    bindings = (parent["bindings"], recurrent["bindings"], reset["bindings"])
    for field in (
        "benchmark_source_sha256",
        "training_source_sha256",
        "source_disjoint_receipt_sha256",
        "identity_order_sha256",
        "tokenizer_sha256",
        "decoding_contract_sha256",
        "scoring_contract_sha256",
    ):
        if len({value.get(field) for value in bindings}) != 1:
            raise HFWorkspaceComparisonError("benchmark execution binding differs")
    parent_rows, recurrent_rows, reset_rows = (
        parent["rows"],
        recurrent["rows"],
        reset["rows"],
    )
    if not len(parent_rows) == len(recurrent_rows) == len(reset_rows):
        raise HFWorkspaceComparisonError("paired row coverage differs")
    recurrent_parent = []
    recurrent_reset = []
    for parent_row, recurrent_row, reset_row in zip(
        parent_rows, recurrent_rows, reset_rows, strict=True
    ):
        common = ("row_id", "domain", "answer_index")
        if any(
            not (
                parent_row.get(field)
                == recurrent_row.get(field)
                == reset_row.get(field)
            )
            for field in common
        ) or any(
            not isinstance(row.get("correct"), bool)
            for row in (parent_row, recurrent_row, reset_row)
        ):
            raise HFWorkspaceComparisonError("paired row identity differs")
        recurrent_correct = float(recurrent_row["correct"])
        recurrent_parent.append(recurrent_correct - float(parent_row["correct"]))
        recurrent_reset.append(recurrent_correct - float(reset_row["correct"]))
    return recurrent_parent, recurrent_reset


def _load_training_result(
    path: Path,
    workspace_evidence: dict[str, Any],
    *,
    mode: str,
    training_schema: str = TRAINING_SCHEMA,
) -> tuple[dict[str, Any], str]:
    file_sha256 = _sha256_file(path)
    try:
        result = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFWorkspaceComparisonError("training result is unreadable") from error
    if not isinstance(result, dict):
        raise HFWorkspaceComparisonError("training result differs")
    unsigned = dict(result)
    claimed = unsigned.pop("receipt_sha256", None)
    if (
        result.get("schema") != training_schema
        or result.get("status") != "complete"
        or result.get("state_mode") != mode
        or result.get("training_sequences") != 61_035
        or result.get("parent_state_unchanged") is not True
        or result.get("four_b_training_executed") is not False
        or result.get("architecture_improvement_demonstrated") is not False
        or claimed != canonical_sha256(unsigned)
        or file_sha256 != workspace_evidence.get("training_result_file_sha256")
        or claimed != workspace_evidence.get("training_receipt_sha256")
        or result.get("run_sha256") != workspace_evidence.get("training_run_sha256")
        or result.get("workspace_final_state_sha256")
        != workspace_evidence.get("workspace_final_state_sha256")
    ):
        raise HFWorkspaceComparisonError("training result evidence differs")
    return result, file_sha256


def _bootstrap_means(values: list[float], seed: int) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    batches = []
    remaining = BOOTSTRAP_REPLICATES
    while remaining:
        count = min(128, remaining)
        indices = torch.randint(
            tensor.numel(),
            (count, tensor.numel()),
            generator=generator,
        )
        batches.append(tensor[indices].mean(dim=1))
        remaining -= count
    return torch.cat(batches)


def _interval(values: torch.Tensor) -> dict[str, float]:
    if values.shape != (BOOTSTRAP_REPLICATES,) or not bool(
        torch.isfinite(values).all().item()
    ):
        raise HFWorkspaceComparisonError("paired bootstrap differs")
    ordered = values.sort().values
    lower = ordered[math.floor(0.025 * BOOTSTRAP_REPLICATES)]
    upper = ordered[math.ceil(0.975 * BOOTSTRAP_REPLICATES) - 1]
    return {"lcb95": float(lower), "ucb95": float(upper)}


def compare(
    *,
    parent_paths: dict[str, Path],
    recurrent_paths: dict[str, Path],
    reset_paths: dict[str, Path],
    recurrent_training_result: Path,
    reset_training_result: Path,
    training_schema: str = TRAINING_SCHEMA,
    comparison_schema: str = SCHEMA,
    pass_action: str = "authorize_sub4b_confirmation",
    fail_action: str = "reject_recurrent_workspace",
    claim_limit: str = (
        "A pass authorizes only a larger sub-4B confirmation. A failure rejects "
        "this recurrent workspace factor. Neither outcome executes or authorizes "
        "4B training."
    ),
) -> dict[str, Any]:
    """Return the frozen, paired two-board development decision."""

    loaded: dict[str, dict[str, tuple[dict[str, Any], str]]] = {}
    for benchmark in BENCHMARKS:
        loaded[benchmark] = {
            "parent": _load_result(
                parent_paths[benchmark], benchmark=benchmark, evidence="parent"
            ),
            "recurrent": _load_result(
                recurrent_paths[benchmark],
                benchmark=benchmark,
                evidence="recurrent",
            ),
            "reset_average": _load_result(
                reset_paths[benchmark],
                benchmark=benchmark,
                evidence="reset_average",
            ),
        }
    recurrent_workspace = loaded[BENCHMARKS[0]]["recurrent"][0]["workspace_evidence"]
    reset_workspace = loaded[BENCHMARKS[0]]["reset_average"][0]["workspace_evidence"]
    recurrent_training, recurrent_training_file_sha256 = _load_training_result(
        recurrent_training_result,
        recurrent_workspace,
        mode="recurrent",
        training_schema=training_schema,
    )
    reset_training, reset_training_file_sha256 = _load_training_result(
        reset_training_result,
        reset_workspace,
        mode="reset_average",
        training_schema=training_schema,
    )
    if any(
        recurrent_training.get(field) != reset_training.get(field)
        for field in COMMON_WORKSPACE_FIELDS
    ):
        raise HFWorkspaceComparisonError("matched workspace training inputs differ")
    if recurrent_workspace.get("workspace_final_state_sha256") == reset_workspace.get(
        "workspace_final_state_sha256"
    ) or recurrent_workspace.get("training_run_sha256") == reset_workspace.get(
        "training_run_sha256"
    ):
        raise HFWorkspaceComparisonError("matched workspace outputs are not distinct")
    for benchmark in BENCHMARKS[1:]:
        for mode, expected in (
            ("recurrent", recurrent_workspace),
            ("reset_average", reset_workspace),
        ):
            if loaded[benchmark][mode][0].get("workspace_evidence") != expected:
                raise HFWorkspaceComparisonError(
                    "workspace identity differs across benchmarks"
                )

    seed_material = canonical_sha256(
        [
            loaded[benchmark][mode][0]["receipt_sha256"]
            for benchmark in BENCHMARKS
            for mode in ("parent", "recurrent", "reset_average")
        ]
    )
    boards = {}
    macro_parent_replicates = []
    macro_reset_replicates = []
    for index, benchmark in enumerate(BENCHMARKS):
        parent = loaded[benchmark]["parent"][0]
        recurrent = loaded[benchmark]["recurrent"][0]
        reset = loaded[benchmark]["reset_average"][0]
        recurrent_parent, recurrent_reset = _paired_rows(parent, recurrent, reset)
        seed = int(seed_material[index * 16 : index * 16 + 16], 16)
        parent_replicates = _bootstrap_means(recurrent_parent, seed)
        reset_replicates = _bootstrap_means(recurrent_reset, seed ^ 0x5A17)
        macro_parent_replicates.append(parent_replicates)
        macro_reset_replicates.append(reset_replicates)
        boards[benchmark] = {
            "rows": len(recurrent_parent),
            "parent_accuracy": parent["aggregate"]["accuracy"],
            "recurrent_accuracy": recurrent["aggregate"]["accuracy"],
            "reset_average_accuracy": reset["aggregate"]["accuracy"],
            "recurrent_minus_parent": sum(recurrent_parent) / len(recurrent_parent),
            "recurrent_minus_reset_average": sum(recurrent_reset)
            / len(recurrent_reset),
            "recurrent_minus_parent_interval": _interval(parent_replicates),
            "recurrent_minus_reset_average_interval": _interval(reset_replicates),
        }
    macro_parent = torch.stack(macro_parent_replicates).mean(dim=0)
    macro_reset = torch.stack(macro_reset_replicates).mean(dim=0)
    macro_parent_observed = sum(
        boards[benchmark]["recurrent_minus_parent"] for benchmark in BENCHMARKS
    ) / len(BENCHMARKS)
    macro_reset_observed = sum(
        boards[benchmark]["recurrent_minus_reset_average"] for benchmark in BENCHMARKS
    ) / len(BENCHMARKS)
    checks = {
        "macro_gain_vs_parent_at_least_0p5pp": (
            macro_parent_observed >= MINIMUM_MACRO_GAIN
        ),
        "macro_gain_vs_reset_at_least_0p5pp": (
            macro_reset_observed >= MINIMUM_MACRO_GAIN
        ),
        "no_board_regresses_vs_parent_more_than_0p5pp": all(
            boards[benchmark]["recurrent_minus_parent"] >= MAXIMUM_BOARD_REGRESSION
            for benchmark in BENCHMARKS
        ),
        "no_board_regresses_vs_reset_more_than_0p5pp": all(
            boards[benchmark]["recurrent_minus_reset_average"]
            >= MAXIMUM_BOARD_REGRESSION
            for benchmark in BENCHMARKS
        ),
        "at_least_one_board_paired_lcb_positive_vs_reset": any(
            boards[benchmark]["recurrent_minus_reset_average_interval"]["lcb95"] > 0
            for benchmark in BENCHMARKS
        ),
    }
    passed = all(checks.values())
    payload: dict[str, Any] = {
        "schema": comparison_schema,
        "status": "complete",
        "benchmarks": boards,
        "macro": {
            "recurrent_minus_parent": macro_parent_observed,
            "recurrent_minus_reset_average": macro_reset_observed,
            "recurrent_minus_parent_interval": _interval(macro_parent),
            "recurrent_minus_reset_average_interval": _interval(macro_reset),
        },
        "checks": checks,
        "pass": passed,
        "action": pass_action if passed else fail_action,
        "bootstrap": {
            "method": "paired_row_resampling_stratified_by_benchmark",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_material_sha256": seed_material,
            "quantiles": [0.025, 0.975],
        },
        "inputs": {
            benchmark: {
                mode: {
                    "path": str(
                        {
                            "parent": parent_paths,
                            "recurrent": recurrent_paths,
                            "reset_average": reset_paths,
                        }[mode][benchmark].resolve()
                    ),
                    "file_sha256": loaded[benchmark][mode][1],
                    "receipt_sha256": loaded[benchmark][mode][0]["receipt_sha256"],
                }
                for mode in ("parent", "recurrent", "reset_average")
            }
            for benchmark in BENCHMARKS
        },
        "training_results": {
            "recurrent": {
                "path": str(Path(recurrent_training_result).resolve()),
                "file_sha256": recurrent_training_file_sha256,
                "receipt_sha256": recurrent_training["receipt_sha256"],
                "run_sha256": recurrent_training["run_sha256"],
            },
            "reset_average": {
                "path": str(Path(reset_training_result).resolve()),
                "file_sha256": reset_training_file_sha256,
                "receipt_sha256": reset_training["receipt_sha256"],
                "run_sha256": reset_training["run_sha256"],
            },
        },
        "workspace_common_evidence": {
            field: recurrent_training[field] for field in COMMON_WORKSPACE_FIELDS
        },
        "development_only": True,
        "architecture_locked": False,
        "four_b_training_executed": False,
        "four_b_training_authorized": False,
        "claim_limit": claim_limit,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def write_comparison(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise HFWorkspaceComparisonError("comparison output target differs")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for benchmark in BENCHMARKS:
        parser.add_argument(
            f"--parent-{benchmark.replace('_', '-')}", type=Path, required=True
        )
        parser.add_argument(
            f"--recurrent-{benchmark.replace('_', '-')}", type=Path, required=True
        )
        parser.add_argument(
            f"--reset-{benchmark.replace('_', '-')}", type=Path, required=True
        )
    parser.add_argument("--recurrent-training-result", type=Path, required=True)
    parser.add_argument("--reset-training-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(
        parent_paths={
            benchmark: getattr(args, f"parent_{benchmark}") for benchmark in BENCHMARKS
        },
        recurrent_paths={
            benchmark: getattr(args, f"recurrent_{benchmark}")
            for benchmark in BENCHMARKS
        },
        reset_paths={
            benchmark: getattr(args, f"reset_{benchmark}") for benchmark in BENCHMARKS
        },
        recurrent_training_result=args.recurrent_training_result,
        reset_training_result=args.reset_training_result,
    )
    write_comparison(args.output, payload)
    print(
        json.dumps(
            {
                "pass": payload["pass"],
                "action": payload["action"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
