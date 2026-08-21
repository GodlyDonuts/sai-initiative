"""Compare exact row-aligned 100M short-screen development results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "sai-short-screen-family-comparison-v1"
FAMILIES = ("gated_gqa", "gdn_hybrid", "kda_mla_hybrid")
BENCHMARKS = ("mmlu_pro", "musr")
SHARED_BINDINGS = (
    "benchmark_source_sha256",
    "training_source_sha256",
    "source_disjoint_receipt_sha256",
    "identity_order_sha256",
    "tokenizer_sha256",
    "evaluator_code_sha256",
    "runtime_files_sha256",
    "runtime_sha256",
    "decoding_contract_sha256",
    "scoring_contract_sha256",
)


class ShortScreenComparisonError(RuntimeError):
    """An input result or paired comparison differs from the contract."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ShortScreenComparisonError("result artifact is missing or unsafe")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_result(path: Path, benchmark: str) -> tuple[dict[str, Any], str]:
    file_sha256 = _sha256_file(path)
    try:
        result = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShortScreenComparisonError("result artifact is unreadable") from error
    if not isinstance(result, dict):
        raise ShortScreenComparisonError("result payload differs")
    unsigned = dict(result)
    receipt_sha256 = unsigned.pop("receipt_sha256", None)
    coverage = result.get("coverage")
    rows = result.get("rows")
    aggregate = result.get("aggregate")
    if (
        result.get("schema") != "sai-development-mc-likelihood-v1"
        or result.get("status") != "complete"
        or result.get("benchmark") != benchmark
        or result.get("development_only") is not True
        or result.get("official_benchmark_result") is not False
        or result.get("public_terminal_result") is not False
        or result.get("architecture_promotion_allowed") is not False
        or receipt_sha256 != _canonical_sha256(unsigned)
        or not isinstance(coverage, dict)
        or set(coverage) != {"expected_rows", "scored_rows"}
        or coverage["expected_rows"] != coverage["scored_rows"]
        or not isinstance(rows, list)
        or len(rows) != coverage["scored_rows"]
        or not isinstance(aggregate, dict)
        or aggregate.get("rows") != len(rows)
        or aggregate.get("correct") != sum(row.get("correct") is True for row in rows)
        or aggregate.get("accuracy") != aggregate.get("correct") / len(rows)
    ):
        raise ShortScreenComparisonError("result receipt or coverage differs")
    identities = [row.get("row_id") for row in rows]
    if any(not isinstance(identity, str) or not identity for identity in identities):
        raise ShortScreenComparisonError("row identity differs")
    if len(set(identities)) != len(identities):
        raise ShortScreenComparisonError("row identities are duplicated")
    return result, file_sha256


def _paired_interval(values: list[int]) -> dict[str, Any]:
    count = len(values)
    mean = sum(values) / count
    variance = (
        sum((value - mean) ** 2 for value in values) / (count - 1) if count > 1 else 0.0
    )
    half_width = 1.959963984540054 * math.sqrt(variance / count)
    return {
        "method": "paired_normal_95ci",
        "delta_percentage_points": 100.0 * mean,
        "lower_percentage_points": 100.0 * (mean - half_width),
        "upper_percentage_points": 100.0 * (mean + half_width),
    }


def compare(
    result_paths: Mapping[str, Mapping[str, Path]],
) -> dict[str, Any]:
    """Validate and compare all three families on both exact populations."""

    if set(result_paths) != set(FAMILIES) or any(
        set(result_paths[family]) != set(BENCHMARKS) for family in FAMILIES
    ):
        raise ShortScreenComparisonError("family/benchmark result matrix differs")
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    file_hashes: dict[str, dict[str, str]] = {}
    for family in FAMILIES:
        loaded[family] = {}
        file_hashes[family] = {}
        for benchmark in BENCHMARKS:
            loaded[family][benchmark], file_hashes[family][benchmark] = _load_result(
                result_paths[family][benchmark], benchmark
            )

    benchmarks: dict[str, Any] = {}
    for benchmark in BENCHMARKS:
        reference = loaded[FAMILIES[0]][benchmark]
        reference_rows = reference["rows"]
        reference_bindings = reference.get("bindings")
        if not isinstance(reference_bindings, dict):
            raise ShortScreenComparisonError("result bindings differ")
        checkpoint_hashes = set()
        for family in FAMILIES:
            result = loaded[family][benchmark]
            bindings = result.get("bindings")
            if not isinstance(bindings, dict) or any(
                bindings.get(field) != reference_bindings.get(field)
                for field in SHARED_BINDINGS
            ):
                raise ShortScreenComparisonError("shared evaluation binding differs")
            checkpoint_sha256 = bindings.get("checkpoint_sha256")
            if (
                not isinstance(checkpoint_sha256, str)
                or len(checkpoint_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in checkpoint_sha256
                )
            ):
                raise ShortScreenComparisonError("checkpoint identity differs")
            checkpoint_hashes.add(checkpoint_sha256)
            for reference_row, row in zip(reference_rows, result["rows"], strict=True):
                if any(
                    row.get(field) != reference_row.get(field)
                    for field in ("row_id", "domain", "answer_index")
                ):
                    raise ShortScreenComparisonError("paired row identity differs")
                if not isinstance(row.get("correct"), bool):
                    raise ShortScreenComparisonError("paired row score differs")
        if len(checkpoint_hashes) != len(FAMILIES):
            raise ShortScreenComparisonError(
                "family checkpoint identities are duplicated"
            )

        pairwise = {}
        for left_index, left in enumerate(FAMILIES):
            for right in FAMILIES[left_index + 1 :]:
                values = [
                    int(left_row["correct"]) - int(right_row["correct"])
                    for left_row, right_row in zip(
                        loaded[left][benchmark]["rows"],
                        loaded[right][benchmark]["rows"],
                        strict=True,
                    )
                ]
                pairwise[f"{left}_minus_{right}"] = {
                    "left_only_correct": values.count(1),
                    "right_only_correct": values.count(-1),
                    "same_outcome": values.count(0),
                    "paired_interval": _paired_interval(values),
                }
        benchmarks[benchmark] = {
            "rows": len(reference_rows),
            "families": {
                family: {
                    "accuracy": loaded[family][benchmark]["aggregate"]["accuracy"],
                    "correct": loaded[family][benchmark]["aggregate"]["correct"],
                    "result_file_sha256": file_hashes[family][benchmark],
                    "result_receipt_sha256": loaded[family][benchmark][
                        "receipt_sha256"
                    ],
                    "checkpoint_sha256": loaded[family][benchmark]["bindings"][
                        "checkpoint_sha256"
                    ],
                }
                for family in FAMILIES
            },
            "pairwise": pairwise,
        }

    result = {
        "schema": SCHEMA,
        "status": "complete",
        "development_only": True,
        "iso_data_comparison": True,
        "iso_flop_comparison": False,
        "scientific_promotion_allowed": False,
        "four_b_training_authorized": False,
        "benchmarks": benchmarks,
    }
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


def write_comparison(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise ShortScreenComparisonError("comparison output path differs")
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
    parser = argparse.ArgumentParser()
    for family in FAMILIES:
        for benchmark in BENCHMARKS:
            parser.add_argument(
                f"--{family.replace('_', '-')}-{benchmark.replace('_', '-')}",
                type=Path,
                required=True,
            )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        family: {
            benchmark: getattr(args, f"{family}_{benchmark}")
            for benchmark in BENCHMARKS
        }
        for family in FAMILIES
    }
    write_comparison(args.output, compare(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
