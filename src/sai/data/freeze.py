"""Freeze exact, verified, benchmark-disjoint Sai training populations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sai.readiness import BENCHMARK_ROWS, DATA_ROLES

SCHEMA = "sai-4b-training-data-freeze-v1"
ROW_SCHEMA = "sai-4b-training-example-v1"
ROLE_MODES = {
    "skill_direct": "direct",
    "skill_deliberate": "deliberate",
    "behavior_replay": "replay",
    "rl_prompts": "rl_prompt",
}
VERIFIERS = {
    "exact_answer",
    "execution",
    "unit_tests",
    "symbolic",
    "parent_behavior",
    "rule_verifier",
    "human_review",
}
WORD = re.compile(r"\w+")
PROMPT_FIELDS = ("prompt", "question", "problem", "instruction", "text")


class DataFreezeError(RuntimeError):
    """A training row, benchmark boundary, or output contract is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def normalized_words(text: str) -> str:
    return " ".join(WORD.findall(text.casefold()))


def ngrams(text: str, width: int) -> Iterable[str]:
    words = WORD.findall(text.casefold())
    if not words:
        return
    if len(words) < width:
        yield " ".join(words)
        return
    for index in range(len(words) - width + 1):
        yield " ".join(words[index : index + width])


def _prompt(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    for field in PROMPT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_benchmark_boundary(
    paths: dict[str, Path],
    *,
    ngram_width: int,
    expected_rows: dict[str, int],
) -> tuple[set[str], set[str], dict[str, Any]]:
    if set(paths) != set(expected_rows):
        raise DataFreezeError("exact benchmark boundary is required")
    exact: set[str] = set()
    grams: set[str] = set()
    receipts: dict[str, Any] = {}
    for benchmark in sorted(paths):
        path = paths[benchmark]
        if not path.is_file() or path.is_symlink():
            raise DataFreezeError(f"benchmark source is missing or unsafe: {path}")
        rows = 0
        identities: list[str] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DataFreezeError(
                        f"benchmark row is malformed: {path}"
                    ) from error
                prompt = _prompt(row)
                normalized = normalized_words(prompt)
                if not normalized:
                    raise DataFreezeError("benchmark prompt is empty")
                identity = row.get("id")
                if not isinstance(identity, str) or not identity:
                    identity = canonical_sha256(
                        {"benchmark": benchmark, "prompt": prompt}
                    )
                rows += 1
                identities.append(identity)
                exact.add(normalized)
                grams.update(ngrams(prompt, ngram_width))
        if rows != expected_rows[benchmark] or len(identities) != len(set(identities)):
            raise DataFreezeError(f"{benchmark} row geometry differs")
        receipts[benchmark] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "rows": rows,
            "identity_sha256": canonical_sha256(identities),
        }
    return exact, grams, receipts


def _validate_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DataFreezeError(f"{field} differs")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise DataFreezeError(f"{field} differs") from error
    return value


def normalize_training_row(row: Any, role: str) -> dict[str, Any]:
    if not isinstance(row, dict) or row.get("schema") != ROW_SCHEMA:
        raise DataFreezeError("training row schema differs")
    if row.get("role") != role or row.get("mode") != ROLE_MODES[role]:
        raise DataFreezeError("training row role/mode differs")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise DataFreezeError("training prompt is empty")
    prompt = prompt.strip()
    response = row.get("response")
    if role == "rl_prompts":
        if response not in (None, ""):
            raise DataFreezeError("RL prompt must not contain a training response")
        response = None
    elif not isinstance(response, str) or not response.strip():
        raise DataFreezeError("supervised/replay response is empty")
    else:
        response = response.strip()
    source = row.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("dataset"), str)
        or not source["dataset"]
        or not isinstance(source.get("row_id"), str)
        or not source["row_id"]
        or not isinstance(source.get("license"), str)
        or not source["license"]
    ):
        raise DataFreezeError("training source provenance differs")
    verification = row.get("verification")
    if (
        not isinstance(verification, dict)
        or verification.get("passed") is not True
        or verification.get("kind") not in VERIFIERS
    ):
        raise DataFreezeError("training verification differs")
    evidence = _validate_sha256(
        verification.get("evidence_sha256"), "verification evidence sha256"
    )
    normalized = {
        "schema": ROW_SCHEMA,
        "role": role,
        "mode": ROLE_MODES[role],
        "prompt": prompt,
        "response": response,
        "source": {
            "dataset": source["dataset"],
            "row_id": source["row_id"],
            "license": source["license"],
        },
        "verification": {
            "passed": True,
            "kind": verification["kind"],
            "evidence_sha256": evidence,
        },
    }
    identity = canonical_sha256(normalized)
    declared = row.get("identity_sha256")
    if declared is not None and declared != identity:
        raise DataFreezeError("declared training identity differs")
    return {**normalized, "identity_sha256": identity}


def overlap_kind(
    text: str,
    exact: set[str],
    benchmark_ngrams: set[str],
    width: int,
) -> str | None:
    normalized = normalized_words(text)
    if normalized in exact:
        return "exact"
    return (
        "ngram"
        if any(gram in benchmark_ngrams for gram in ngrams(text, width))
        else None
    )


def freeze(
    sources: dict[str, Path],
    benchmarks: dict[str, Path],
    output: Path,
    *,
    ngram_width: int = 13,
    expected_benchmark_rows: dict[str, int] = BENCHMARK_ROWS,
) -> dict[str, Any]:
    """Write deterministic role files plus a complete source/overlap receipt."""

    if set(sources) != DATA_ROLES or ngram_width <= 0:
        raise DataFreezeError("exact data roles and positive ngram width are required")
    if output.exists():
        raise DataFreezeError("data freeze output already exists")
    for path in sources.values():
        if not path.is_file() or path.is_symlink():
            raise DataFreezeError(f"training source is missing or unsafe: {path}")
    exact, benchmark_ngrams, benchmark_receipts = load_benchmark_boundary(
        benchmarks,
        ngram_width=ngram_width,
        expected_rows=expected_benchmark_rows,
    )

    stage = output.with_name(f".{output.name}.partial.{os.getpid()}")
    if stage.exists():
        raise DataFreezeError("data freeze staging path already exists")
    stage.mkdir(parents=True)
    seen_prompts: set[str] = set()
    role_receipts: dict[str, Any] = {}
    try:
        for role in sorted(DATA_ROLES):
            source_path = sources[role]
            accepted: list[dict[str, Any]] = []
            input_rows = malformed = duplicates = 0
            overlap_drops: Counter[str] = Counter()
            with source_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    input_rows += 1
                    try:
                        row = json.loads(line)
                        normalized = normalize_training_row(row, role)
                    except (json.JSONDecodeError, DataFreezeError):
                        malformed += 1
                        continue
                    prompt_key = normalized_words(normalized["prompt"])
                    if prompt_key in seen_prompts:
                        duplicates += 1
                        continue
                    overlap = overlap_kind(
                        normalized["prompt"],
                        exact,
                        benchmark_ngrams,
                        ngram_width,
                    )
                    if overlap is not None:
                        overlap_drops[f"{overlap}:prompt"] += 1
                        continue
                    response = normalized["response"]
                    if response is not None:
                        overlap = overlap_kind(
                            response,
                            exact,
                            benchmark_ngrams,
                            ngram_width,
                        )
                        if overlap is not None:
                            overlap_drops[f"{overlap}:response"] += 1
                            continue
                    seen_prompts.add(prompt_key)
                    accepted.append(normalized)
            if not accepted:
                raise DataFreezeError(f"{role} has no accepted rows")
            accepted.sort(key=lambda row: row["identity_sha256"])
            role_path = stage / f"{role}.jsonl"
            with role_path.open("w", encoding="utf-8") as handle:
                for row in accepted:
                    handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
            role_receipts[role] = {
                "source": str(source_path.resolve()),
                "source_sha256": sha256_file(source_path),
                "path": str((output / role_path.name).resolve()),
                "sha256": sha256_file(role_path),
                "rows": len(accepted),
                "identity_sha256": canonical_sha256(
                    [row["identity_sha256"] for row in accepted]
                ),
                "input_rows": input_rows,
                "malformed_or_unverified_rows": malformed,
                "duplicate_prompt_rows": duplicates,
                "benchmark_overlap_drops": dict(sorted(overlap_drops.items())),
            }
        report = {
            "schema": SCHEMA,
            "status": "complete",
            "output": str(output.resolve()),
            "ngram_width": ngram_width,
            "benchmarks": benchmark_receipts,
            "roles": role_receipts,
            "checks": {
                "all_roles_nonempty": True,
                "global_prompt_deduplication": True,
                "model_visible_prompt_and_response_filtering": True,
                "benchmark_rows_complete": True,
                "training_not_authorized": True,
            },
            "training_authorized": False,
        }
        report_path = stage / "freeze_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _mapping(values: list[str], expected: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or key not in expected or key in result or not path:
            raise DataFreezeError(f"invalid mapping: {value}")
        result[key] = Path(path)
    if set(result) != expected:
        raise DataFreezeError(f"mapping keys differ: {sorted(result)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ngram-width", type=int, default=13)
    args = parser.parse_args()
    report = freeze(
        _mapping(args.source, DATA_ROLES),
        _mapping(args.benchmark, set(BENCHMARK_ROWS)),
        args.output,
        ngram_width=args.ngram_width,
    )
    print(
        json.dumps(
            {role: receipt["rows"] for role, receipt in report["roles"].items()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
