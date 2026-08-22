"""Compare curriculum ordering with its matched control on real dev boards."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-curriculum-order-development-benchmark-comparison-v1"
BENCHMARKS = ("mmlu_pro", "musr")
EXPECTED_ROWS = {"mmlu_pro": 12_032, "musr": 756}
BOOTSTRAP_SEED = 2026082202
BOOTSTRAP_REPLICATES = 10_000
DOMAIN_REGRESSION_FLOOR = -0.01


class CurriculumBenchmarkComparisonError(RuntimeError):
    """A comparison, benchmark result, pairing, or decision differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CurriculumBenchmarkComparisonError(message)


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or unsafe")
    file_sha256 = sha256_file(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CurriculumBenchmarkComparisonError(f"{label} is unreadable") from error
    _require(isinstance(payload, dict), f"{label} must be an object")
    unsigned = dict(payload)
    claimed = unsigned.pop("receipt_sha256", None)
    _require(claimed == canonical_sha256(unsigned), f"{label} self hash differs")
    return payload, file_sha256


def _load_order_comparison(path: Path) -> tuple[dict[str, Any], str]:
    payload, file_sha256 = _load_json(path, "curriculum order comparison")
    _require(
        payload.get("schema") == "sai-curriculum-order-training-comparison-v1"
        and payload.get("status") == "complete"
        and payload.get("curriculum_order_supported_by_heldout_nll") is True
        and payload.get("heldout_phase_no_regression") is True
        and payload.get("same_documents_tokens_targets_masks") is True
        and payload.get("only_training_sequence_order_changed") is True
        and payload.get("same_model_initialization_optimizer_budget_compute") is True
        and payload.get("development_population_disjoint_from_training") is True
        and payload.get("real_benchmark_gate_required") is True
        and payload.get("scientific_promotion_authorized") is False
        and payload.get("four_b_training_authorized") is False,
        "curriculum order comparison does not admit benchmark confirmation",
    )
    arms = payload.get("arms")
    _require(
        isinstance(arms, dict) and set(arms) == {"curriculum", "order_control"},
        "curriculum order arms differ",
    )
    return payload, file_sha256


def _load_benchmark_result(path: Path, benchmark: str) -> tuple[dict[str, Any], str]:
    payload, file_sha256 = _load_json(path, f"{benchmark} result")
    _require(
        payload.get("schema") == "sai-development-mc-likelihood-v1"
        and payload.get("status") == "complete"
        and payload.get("benchmark") == benchmark
        and payload.get("development_only") is True
        and payload.get("official_benchmark_result") is False
        and payload.get("public_terminal_result") is False
        and payload.get("architecture_promotion_allowed") is False,
        f"{benchmark} result contract differs",
    )
    rows = payload.get("rows")
    coverage = payload.get("coverage")
    aggregate = payload.get("aggregate")
    bindings = payload.get("bindings")
    _require(
        isinstance(rows, list)
        and len(rows) == EXPECTED_ROWS[benchmark]
        and isinstance(coverage, dict)
        and coverage == {"expected_rows": len(rows), "scored_rows": len(rows)}
        and isinstance(aggregate, dict)
        and isinstance(bindings, dict),
        f"{benchmark} coverage differs",
    )
    expected_row_keys = {
        "row_id",
        "domain",
        "answer_index",
        "predicted_index",
        "correct",
        "choice_scores",
    }
    _require(
        all(
            isinstance(row, dict)
            and set(row) == expected_row_keys
            and isinstance(row["row_id"], str)
            and row["row_id"]
            and isinstance(row["domain"], str)
            and row["domain"]
            and isinstance(row["correct"], bool)
            for row in rows
        ),
        f"{benchmark} row evidence differs",
    )
    identities = [row["row_id"] for row in rows]
    _require(len(identities) == len(set(identities)), f"{benchmark} rows repeat")
    correct = sum(row["correct"] for row in rows)
    _require(
        aggregate
        == {"correct": correct, "rows": len(rows), "accuracy": correct / len(rows)},
        f"{benchmark} aggregate differs",
    )
    return payload, file_sha256


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = math.floor(probability * (len(ordered) - 1))
    return ordered[index]


def _paired_bootstrap(
    paired: dict[str, list[tuple[str, int]]],
) -> dict[str, Any]:
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for benchmark, values in paired.items():
        for domain, delta in values:
            strata[(benchmark, domain)].append(delta)
    generator = random.Random(BOOTSTRAP_SEED)
    samples: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        benchmark_deltas = []
        for benchmark in BENCHMARKS:
            total = 0
            rows = 0
            for (candidate_benchmark, _domain), values in sorted(strata.items()):
                if candidate_benchmark != benchmark:
                    continue
                total += sum(
                    values[generator.randrange(len(values))] for _ in range(len(values))
                )
                rows += len(values)
            benchmark_deltas.append(total / rows)
        samples.append(sum(benchmark_deltas) / len(benchmark_deltas))
    return {
        "method": "paired_domain_stratified_nonparametric_bootstrap",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "macro_delta_lcb_95": _percentile(samples, 0.025),
        "macro_delta_median": _percentile(samples, 0.5),
        "macro_delta_ucb_95": _percentile(samples, 0.975),
    }


def compare_curriculum_benchmarks(
    order_comparison_path: Path,
    *,
    curriculum_results: dict[str, Path],
    control_results: dict[str, Path],
) -> dict[str, Any]:
    """Produce the predeclared benchmark confirmation decision."""

    _require(
        set(curriculum_results) == set(BENCHMARKS)
        and set(control_results) == set(BENCHMARKS),
        "benchmark result set differs",
    )
    order, order_file_sha256 = _load_order_comparison(order_comparison_path)
    loaded: dict[str, dict[str, tuple[dict[str, Any], str]]] = {
        "curriculum": {},
        "order_control": {},
    }
    for benchmark in BENCHMARKS:
        loaded["curriculum"][benchmark] = _load_benchmark_result(
            curriculum_results[benchmark], benchmark
        )
        loaded["order_control"][benchmark] = _load_benchmark_result(
            control_results[benchmark], benchmark
        )

    shared_binding_fields = {
        "benchmark_source_sha256",
        "training_source_sha256",
        "source_disjoint_receipt_sha256",
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
    paired: dict[str, list[tuple[str, int]]] = {}
    domain_deltas: dict[str, dict[str, float]] = {}
    for benchmark in BENCHMARKS:
        curriculum, curriculum_file_sha256 = loaded["curriculum"][benchmark]
        control, control_file_sha256 = loaded["order_control"][benchmark]
        _require(
            all(
                curriculum["bindings"].get(field) == control["bindings"].get(field)
                for field in shared_binding_fields
            ),
            f"{benchmark} evaluation bindings differ",
        )
        _require(
            curriculum["bindings"].get("checkpoint_sha256")
            == order["arms"]["curriculum"]["checkpoint_sha256"]
            and control["bindings"].get("checkpoint_sha256")
            == order["arms"]["order_control"]["checkpoint_sha256"],
            f"{benchmark} checkpoint lineage differs",
        )
        curriculum_rows = curriculum["rows"]
        control_rows = control["rows"]
        _require(
            [(row["row_id"], row["domain"]) for row in curriculum_rows]
            == [(row["row_id"], row["domain"]) for row in control_rows],
            f"{benchmark} row pairing differs",
        )
        deltas = [
            int(curriculum_row["correct"]) - int(control_row["correct"])
            for curriculum_row, control_row in zip(
                curriculum_rows, control_rows, strict=True
            )
        ]
        paired[benchmark] = [
            (row["domain"], delta)
            for row, delta in zip(curriculum_rows, deltas, strict=True)
        ]
        domains: dict[str, list[int]] = defaultdict(list)
        for row, delta in zip(curriculum_rows, deltas, strict=True):
            domains[row["domain"]].append(delta)
        domain_deltas[benchmark] = {
            domain: sum(values) / len(values)
            for domain, values in sorted(domains.items())
        }
        accuracy_delta = sum(deltas) / len(deltas)
        benchmark_rows[benchmark] = {
            "rows": len(deltas),
            "curriculum_accuracy": curriculum["aggregate"]["accuracy"],
            "order_control_accuracy": control["aggregate"]["accuracy"],
            "curriculum_minus_control": accuracy_delta,
            "paired_wins": sum(delta > 0 for delta in deltas),
            "paired_losses": sum(delta < 0 for delta in deltas),
            "paired_ties": sum(delta == 0 for delta in deltas),
            "curriculum_result": {
                "path": str(curriculum_results[benchmark].resolve()),
                "file_sha256": curriculum_file_sha256,
                "receipt_sha256": curriculum["receipt_sha256"],
            },
            "order_control_result": {
                "path": str(control_results[benchmark].resolve()),
                "file_sha256": control_file_sha256,
                "receipt_sha256": control["receipt_sha256"],
            },
        }

    bootstrap = _paired_bootstrap(paired)
    deltas = [benchmark_rows[name]["curriculum_minus_control"] for name in BENCHMARKS]
    macro_delta = sum(deltas) / len(deltas)
    minimum_domain_delta = min(
        delta for benchmark in domain_deltas.values() for delta in benchmark.values()
    )
    checks = {
        "heldout_nll_and_every_phase_passed": True,
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
        "comparison": "curriculum_order_vs_exact_sequence_order_control",
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
        "curriculum_order_supported_by_real_development_benchmarks": supported,
        "data_order_retention_authorized": supported,
        "architecture_promotion_authorized": False,
        "four_b_training_authorized": False,
        "official_benchmark_result": False,
        "public_terminal_result": False,
        "bindings": {
            "order_comparison_path": str(order_comparison_path.resolve()),
            "order_comparison_file_sha256": order_file_sha256,
            "order_comparison_receipt_sha256": order["receipt_sha256"],
        },
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def write_comparison(path: Path, payload: dict[str, Any]) -> None:
    """Publish one immutable comparison receipt."""

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
    parser.add_argument("--order-comparison", type=Path, required=True)
    for arm in ("curriculum", "control"):
        for benchmark in BENCHMARKS:
            parser.add_argument(
                f"--{arm}-{benchmark.replace('_', '-')}", type=Path, required=True
            )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare_curriculum_benchmarks(
        args.order_comparison,
        curriculum_results={
            benchmark: getattr(args, f"curriculum_{benchmark}")
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
                "supported": payload[
                    "curriculum_order_supported_by_real_development_benchmarks"
                ],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
