"""Confirm an equal-token source addition on real source-disjoint dev boards."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file
from sai.evaluation.curriculum_benchmark_compare import (
    BENCHMARKS,
    DOMAIN_REGRESSION_FLOOR,
    CurriculumBenchmarkComparisonError,
    _load_benchmark_result,
    _paired_bootstrap,
)

SCHEMA = "sai-source-addition-development-benchmark-comparison-v1"
BOOTSTRAP_SEED = 2026082203


class SourceAdditionBenchmarkComparisonError(RuntimeError):
    """A source-addition likelihood gate, row pairing, or result differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceAdditionBenchmarkComparisonError(message)


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} differs",
    )
    return value


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    file_sha256 = sha256_file(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceAdditionBenchmarkComparisonError(
            f"{label} is unreadable"
        ) from error
    _require(isinstance(payload, dict), f"{label} must be an object")
    unsigned = dict(payload)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    _require(receipt_sha256 == canonical_sha256(unsigned), f"{label} self hash differs")
    return payload, file_sha256


def _checkpoint_identity(payload: dict[str, Any], arm: str) -> str:
    inputs = payload.get("inputs")
    _require(isinstance(inputs, dict), "source-addition inputs differ")
    descriptor = inputs.get(f"{arm}_checkpoint")
    _require(
        isinstance(descriptor, dict)
        and set(descriptor)
        == {
            "checkpoint_file_sha256",
            "checkpoint_manifest_file_sha256",
            "checkpoint_bundle_sha256",
        },
        f"{arm} checkpoint binding differs",
    )
    for field in descriptor:
        _sha256(descriptor[field], f"{arm} {field}")
    return descriptor["checkpoint_bundle_sha256"]


def _load_nll_comparison(path: Path) -> tuple[dict[str, Any], str]:
    payload, file_sha256 = _load_json(path, "source-addition NLL comparison")
    _require(
        payload.get("schema") == "sai-source-addition-nll-comparison-v1"
        and payload.get("status") == "complete"
        and payload.get("source_addition_supported_by_heldout_nll") is True
        and payload.get("real_source_disjoint_benchmark_confirmation_required") is True
        and payload.get("source_addition_retained") is False
        and payload.get("data_promotion_authorized") is False
        and payload.get("architecture_promotion_authorized") is False
        and payload.get("four_b_training_authorized") is False
        and payload.get("optimizer_steps") == 0
        and payload.get("backward_calls") == 0,
        "source-addition NLL comparison does not admit benchmark confirmation",
    )
    treatment = _checkpoint_identity(payload, "treatment")
    control = _checkpoint_identity(payload, "control")
    _require(treatment != control, "source-addition checkpoints are duplicated")
    inputs = payload["inputs"]
    treatment_source = _sha256(
        inputs.get("treatment_training_source_sha256"), "treatment training source"
    )
    control_source = _sha256(
        inputs.get("control_training_source_sha256"), "control training source"
    )
    _require(
        treatment_source != control_source, "source-addition sources are duplicated"
    )
    return payload, file_sha256


def _benchmark_result(path: Path, benchmark: str) -> tuple[dict[str, Any], str]:
    try:
        return _load_benchmark_result(path, benchmark)
    except CurriculumBenchmarkComparisonError as error:
        raise SourceAdditionBenchmarkComparisonError(
            f"{benchmark} benchmark result differs"
        ) from error


def compare_source_addition_benchmarks(
    nll_comparison_path: Path,
    *,
    treatment_results: dict[str, Path],
    control_results: dict[str, Path],
) -> dict[str, Any]:
    """Apply the frozen paired real-benchmark retention gate."""

    _require(
        set(treatment_results) == set(BENCHMARKS)
        and set(control_results) == set(BENCHMARKS),
        "benchmark result set differs",
    )
    nll, nll_file_sha256 = _load_nll_comparison(nll_comparison_path)
    checkpoint_identities = {
        arm: _checkpoint_identity(nll, arm) for arm in ("treatment", "control")
    }
    loaded: dict[str, dict[str, tuple[dict[str, Any], str]]] = {
        "treatment": {},
        "control": {},
    }
    for benchmark in BENCHMARKS:
        loaded["treatment"][benchmark] = _benchmark_result(
            treatment_results[benchmark], benchmark
        )
        loaded["control"][benchmark] = _benchmark_result(
            control_results[benchmark], benchmark
        )

    shared_binding_fields = {
        "benchmark_source_sha256",
        "identity_order_sha256",
        "config_sha256",
        "tokenizer_sha256",
        "evaluator_code_sha256",
        "runtime_files_sha256",
        "runtime_sha256",
        "decoding_contract_sha256",
        "scoring_contract_sha256",
    }
    benchmark_rows: dict[str, Any] = {}
    domain_deltas: dict[str, dict[str, float]] = {}
    paired: dict[str, list[tuple[str, int]]] = {}
    for benchmark in BENCHMARKS:
        treatment, treatment_file_sha256 = loaded["treatment"][benchmark]
        control, control_file_sha256 = loaded["control"][benchmark]
        _require(
            all(
                treatment["bindings"].get(field) == control["bindings"].get(field)
                for field in shared_binding_fields
            ),
            f"{benchmark} evaluation bindings differ",
        )
        _require(
            treatment["bindings"].get("training_source_sha256")
            == nll["inputs"].get("treatment_training_source_sha256")
            and control["bindings"].get("training_source_sha256")
            == nll["inputs"].get("control_training_source_sha256")
            and treatment["bindings"].get("training_source_sha256")
            != control["bindings"].get("training_source_sha256")
            and treatment["bindings"].get("source_disjoint_receipt_sha256")
            != control["bindings"].get("source_disjoint_receipt_sha256"),
            f"{benchmark} source lineage differs",
        )
        _require(
            treatment["bindings"].get("checkpoint_sha256")
            == checkpoint_identities["treatment"]
            and control["bindings"].get("checkpoint_sha256")
            == checkpoint_identities["control"],
            f"{benchmark} checkpoint lineage differs",
        )
        treatment_rows = treatment["rows"]
        control_rows = control["rows"]
        _require(
            [(row["row_id"], row["domain"]) for row in treatment_rows]
            == [(row["row_id"], row["domain"]) for row in control_rows],
            f"{benchmark} row pairing differs",
        )
        deltas = [
            int(treatment_row["correct"]) - int(control_row["correct"])
            for treatment_row, control_row in zip(
                treatment_rows, control_rows, strict=True
            )
        ]
        paired[benchmark] = [
            (row["domain"], delta)
            for row, delta in zip(treatment_rows, deltas, strict=True)
        ]
        domains: dict[str, list[int]] = defaultdict(list)
        for row, delta in zip(treatment_rows, deltas, strict=True):
            domains[row["domain"]].append(delta)
        domain_deltas[benchmark] = {
            domain: sum(values) / len(values)
            for domain, values in sorted(domains.items())
        }
        benchmark_rows[benchmark] = {
            "rows": len(deltas),
            "treatment_accuracy": treatment["aggregate"]["accuracy"],
            "control_accuracy": control["aggregate"]["accuracy"],
            "treatment_minus_control": sum(deltas) / len(deltas),
            "paired_wins": sum(delta > 0 for delta in deltas),
            "paired_losses": sum(delta < 0 for delta in deltas),
            "paired_ties": sum(delta == 0 for delta in deltas),
            "treatment_result": {
                "path": str(treatment_results[benchmark].resolve()),
                "file_sha256": treatment_file_sha256,
                "receipt_sha256": treatment["receipt_sha256"],
            },
            "control_result": {
                "path": str(control_results[benchmark].resolve()),
                "file_sha256": control_file_sha256,
                "receipt_sha256": control["receipt_sha256"],
            },
        }

    bootstrap = _paired_bootstrap(paired, seed=BOOTSTRAP_SEED)
    deltas = [benchmark_rows[name]["treatment_minus_control"] for name in BENCHMARKS]
    macro_delta = sum(deltas) / len(deltas)
    minimum_domain_delta = min(
        delta for benchmark in domain_deltas.values() for delta in benchmark.values()
    )
    checks = {
        "heldout_nll_and_every_stratum_passed": True,
        "every_benchmark_nonnegative": all(delta >= 0 for delta in deltas),
        "at_least_one_benchmark_positive": any(delta > 0 for delta in deltas),
        "macro_accuracy_delta_positive": macro_delta > 0,
        "paired_macro_lcb_95_positive": bootstrap["macro_delta_lcb_95"] > 0,
        "every_domain_delta_at_least_minus_1pp": (
            minimum_domain_delta >= DOMAIN_REGRESSION_FLOOR
        ),
    }
    supported = all(checks.values())
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete",
        "comparison": "source_addition_vs_equal_token_selected_web_control",
        "benchmarks": benchmark_rows,
        "domain_accuracy_deltas": domain_deltas,
        "unweighted_macro_accuracy_delta": macro_delta,
        "minimum_domain_accuracy_delta": minimum_domain_delta,
        "bootstrap": bootstrap,
        "thresholds_predeclared_before_results": {
            "benchmark_delta_floor": 0.0,
            "macro_delta_strictly_positive": True,
            "macro_lcb_95_strictly_positive": True,
            "domain_delta_floor": DOMAIN_REGRESSION_FLOOR,
        },
        "checks": checks,
        "source_addition_supported_by_real_development_benchmarks": supported,
        "source_addition_retained": supported,
        "data_promotion_authorized": False,
        "architecture_promotion_authorized": False,
        "four_b_training_authorized": False,
        "official_benchmark_result": False,
        "public_terminal_result": False,
        "bindings": {
            "nll_comparison_path": str(nll_comparison_path.resolve()),
            "nll_comparison_file_sha256": nll_file_sha256,
            "nll_comparison_receipt_sha256": nll["receipt_sha256"],
            "treatment_checkpoint_sha256": checkpoint_identities["treatment"],
            "control_checkpoint_sha256": checkpoint_identities["control"],
        },
        "claim_limit": (
            "One equal-token source-addition retention decision on frozen "
            "source-disjoint development boards; no architecture or 4B claim."
        ),
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def write_comparison(path: Path, payload: dict[str, Any]) -> None:
    """Publish one immutable source-addition benchmark receipt."""

    _require(
        not path.exists() and not path.is_symlink(), "comparison output already exists"
    )
    _require(
        path.parent.is_dir() and not path.parent.is_symlink(),
        "comparison output parent is missing or unsafe",
    )
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
    parser.add_argument("--nll-comparison", type=Path, required=True)
    for arm in ("treatment", "control"):
        for benchmark in BENCHMARKS:
            parser.add_argument(
                f"--{arm}-{benchmark.replace('_', '-')}", type=Path, required=True
            )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare_source_addition_benchmarks(
        args.nll_comparison,
        treatment_results={
            benchmark: getattr(args, f"treatment_{benchmark}")
            for benchmark in BENCHMARKS
        },
        control_results={
            benchmark: getattr(args, f"control_{benchmark}") for benchmark in BENCHMARKS
        },
    )
    write_comparison(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "source_addition_retained": payload["source_addition_retained"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
