"""Join complete Hermès source audits into one source-safe teacher census."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.institutional_books_compiler_aggregate import (
    SCHEMA as BOOK_AGGREGATE_SCHEMA,
)
from sai.data.reservoir_audit_aggregate import SCHEMA as RESERVOIR_AGGREGATE_SCHEMA
from sai.data.source_quality_gate_publication import SCHEMA as QUALITY_GATE_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-hermes-teacher-census-v1"
MAXIMUM_INPUT_BYTES = 64 << 20


class HermesTeacherCensusError(RuntimeError):
    """A source population, teacher aggregate, or global count differs."""


def _load_signed(path: Path, label: str) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or not 0 < path.stat().st_size <= MAXIMUM_INPUT_BYTES
    ):
        raise HermesTeacherCensusError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise HermesTeacherCensusError(f"{label} cannot be decoded") from error
    if not isinstance(value, dict):
        raise HermesTeacherCensusError(f"{label} differs")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if value.get("receipt_sha256") != canonical_sha256(unsigned):
        raise HermesTeacherCensusError(f"{label} receipt differs")
    return value


def _counts(value: Any, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise HermesTeacherCensusError(f"{label} counts differ")
    result: dict[str, int] = {}
    for key, count in value.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise HermesTeacherCensusError(f"{label} counts differ")
        result[key] = count
    return result


def _require_partition(value: dict[str, int], rows: int, label: str) -> None:
    if sum(value.values()) != rows:
        raise HermesTeacherCensusError(f"{label} coverage differs")


def _usage(value: Any) -> dict[str, int]:
    result = _counts(value, "teacher usage")
    if any(
        key not in {"prompt_tokens", "completion_tokens", "total_tokens"}
        for key in result
    ):
        raise HermesTeacherCensusError("teacher usage fields differ")
    return result


def _extract_reservoir(
    aggregate: dict[str, Any],
) -> tuple[str, int, dict[str, dict[str, int]], dict[str, int]]:
    summary = aggregate.get("summary")
    if (
        aggregate.get("schema") != RESERVOIR_AGGREGATE_SCHEMA
        or aggregate.get("status") != "complete"
        or aggregate.get("training_ready") is not False
        or aggregate.get("independent_factual_verification_complete") is not False
        or aggregate.get("cross_source_deduplication_complete") is not False
        or aggregate.get("benchmark_decontamination_complete") is not False
        or not isinstance(aggregate.get("population_file_sha256"), str)
        or not isinstance(summary, dict)
    ):
        raise HermesTeacherCensusError("reservoir teacher aggregate differs")
    rows = summary.get("rows")
    raw_counts = summary.get("counts")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
        or not isinstance(raw_counts, dict)
        or summary.get("model_judgments_are_verified_admissions") is not False
        or summary.get("representation_verification_is_training_admission") is not False
    ):
        raise HermesTeacherCensusError("reservoir teacher summary differs")
    fields = {
        "verdict": "verdict",
        "domains": "domains",
        "languages": "source_language",
        "styles": "style",
        "risks": "risks",
        "triage_routes": None,
        "recommended_representations": "recommended_representations",
        "epistemic_functions": "epistemic_functions",
        "likely_origins": "likely_origin",
        "grounding_types": "grounding_type",
        "preservation_policies": "preservation_policy",
        "translation_dispositions": "translation_disposition",
        "reservoir_curriculum_phases": "curriculum_phase",
        "difficulty": "difficulty",
        "prerequisite_burden": "prerequisite_burden",
    }
    normalized = {}
    for output_field, source_field in fields.items():
        value = (
            summary.get("conservative_triage_routes")
            if source_field is None
            else raw_counts.get(source_field)
        )
        normalized[output_field] = _counts(value, output_field)
    for field in (
        "verdict",
        "languages",
        "styles",
        "triage_routes",
        "likely_origins",
        "grounding_types",
        "preservation_policies",
        "translation_dispositions",
        "reservoir_curriculum_phases",
        "difficulty",
        "prerequisite_burden",
    ):
        _require_partition(normalized[field], rows, field)
    return (
        aggregate["population_file_sha256"],
        rows,
        normalized,
        _usage(summary.get("usage")),
    )


def _extract_books(
    aggregate: dict[str, Any],
) -> tuple[str, int, dict[str, dict[str, int]], dict[str, int]]:
    population = aggregate.get("population")
    raw_counts = aggregate.get("counts")
    if (
        aggregate.get("schema") != BOOK_AGGREGATE_SCHEMA
        or aggregate.get("status") != "complete_nontraining_book_compiler_aggregate"
        or aggregate.get("source_text_persisted") is not False
        or aggregate.get("evidence_quotes_persisted") is not False
        or aggregate.get("model_judgments_are_verified_admissions") is not False
        or aggregate.get("training_ready") is not False
        or not isinstance(population, dict)
        or not isinstance(raw_counts, dict)
    ):
        raise HermesTeacherCensusError("book teacher aggregate differs")
    rows = population.get("rows")
    source_sha256 = population.get("candidate_file_sha256")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows <= 0
        or not isinstance(source_sha256, str)
    ):
        raise HermesTeacherCensusError("book teacher population differs")
    fields = {
        "verdict": "verdict",
        "domains": "domain",
        "languages": "current_language",
        "styles": "style",
        "risks": "risk",
        "triage_routes": "triage_route",
        "recommended_representations": "recommended_representation",
        "book_curriculum_bands": "curriculum_band",
        "book_genres": "genre",
        "book_translation_types": "translation_type",
        "book_rights_statuses": "rights_status",
    }
    normalized = {
        output_field: _counts(raw_counts.get(source_field), output_field)
        for output_field, source_field in fields.items()
    }
    for field in (
        "verdict",
        "languages",
        "styles",
        "triage_routes",
        "book_curriculum_bands",
        "book_genres",
        "book_translation_types",
        "book_rights_statuses",
    ):
        _require_partition(normalized[field], rows, field)
    return source_sha256, rows, normalized, _usage(aggregate.get("usage"))


def build_census(
    quality_publication_path: Path,
    aggregate_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Require one complete teacher aggregate for every quality-gated source."""

    if (
        not aggregate_paths
        or len(aggregate_paths) != len(set(aggregate_paths))
        or output_path.exists()
        or output_path.is_symlink()
    ):
        raise HermesTeacherCensusError("teacher census geometry differs")
    quality = _load_signed(quality_publication_path, "quality-gate publication")
    populations = quality.get("populations")
    if (
        quality.get("schema") != QUALITY_GATE_SCHEMA
        or quality.get("status")
        != "complete_source_safe_mechanical_quality_gate_publication"
        or quality.get("training_ready") is not False
        or quality.get("cross_population_duplicate_identity_rows") != 0
        or quality.get("cross_population_duplicate_assignments") != 0
        or quality.get("cross_population_duplicate_content_rows") != 0
        or quality.get("cross_population_duplicate_content_assignments") != 0
        or not isinstance(populations, list)
        or not populations
    ):
        raise HermesTeacherCensusError("quality-gate publication differs")
    expected: dict[str, dict[str, Any]] = {}
    for order, population in enumerate(populations):
        if (
            not isinstance(population, dict)
            or population.get("order") != order
            or not isinstance(population.get("source_sha256"), str)
            or len(population["source_sha256"]) != 64
            or isinstance(population.get("source_rows"), bool)
            or not isinstance(population.get("source_rows"), int)
            or population["source_rows"] <= 0
            or population["source_sha256"] in expected
        ):
            raise HermesTeacherCensusError("quality-gate population differs")
        expected[population["source_sha256"]] = population
    if (
        quality.get("population_assignment_rows")
        != sum(row["source_rows"] for row in populations)
        or quality.get("unique_candidate_rows") != quality["population_assignment_rows"]
        or quality.get("unique_source_content_rows")
        != quality["population_assignment_rows"]
    ):
        raise HermesTeacherCensusError("quality-gate population coverage differs")

    merged: dict[str, Counter[str]] = {}
    usage = Counter()
    observed: dict[str, dict[str, Any]] = {}
    for path in aggregate_paths:
        aggregate = _load_signed(path, "teacher aggregate")
        if aggregate.get("schema") == RESERVOIR_AGGREGATE_SCHEMA:
            source_sha256, rows, counts, aggregate_usage = _extract_reservoir(aggregate)
        elif aggregate.get("schema") == BOOK_AGGREGATE_SCHEMA:
            source_sha256, rows, counts, aggregate_usage = _extract_books(aggregate)
        else:
            raise HermesTeacherCensusError("teacher aggregate schema differs")
        source = expected.get(source_sha256)
        if source is None or source_sha256 in observed or rows != source["source_rows"]:
            raise HermesTeacherCensusError("teacher aggregate population differs")
        for field, values in counts.items():
            merged.setdefault(field, Counter()).update(values)
        usage.update(aggregate_usage)
        observed[source_sha256] = {
            "order": source["order"],
            "source_sha256": source_sha256,
            "rows": rows,
            "aggregate_file": path.name,
            "aggregate_file_sha256": sha256_file(path),
            "aggregate_schema": aggregate["schema"],
            "aggregate_receipt_sha256": aggregate["receipt_sha256"],
        }
    if set(observed) != set(expected):
        raise HermesTeacherCensusError("teacher aggregate coverage differs")
    ordered = sorted(observed.values(), key=lambda row: row["order"])
    total_rows = sum(row["rows"] for row in ordered)
    for field in ("verdict", "languages", "styles", "triage_routes"):
        if sum(merged.get(field, {}).values()) != total_rows:
            raise HermesTeacherCensusError(f"global {field} coverage differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete_source_safe_nontraining_hermes_teacher_census",
        "quality_gate_publication": {
            "path": quality_publication_path.name,
            "file_sha256": sha256_file(quality_publication_path),
            "receipt_sha256": quality["receipt_sha256"],
        },
        "population_rows": total_rows,
        "population_count": len(ordered),
        "populations": ordered,
        "ordered_aggregate_receipts_sha256": canonical_sha256(
            [row["aggregate_receipt_sha256"] for row in ordered]
        ),
        "counts": {
            field: dict(sorted(counter.items()))
            for field, counter in sorted(merged.items())
        },
        "usage": dict(sorted(usage.items())),
        "publication_contains_source_text": False,
        "teacher_judgments_are_verified_admissions": False,
        "rights_admission_complete": False,
        "independent_factual_verification_complete": False,
        "representation_verification_complete": False,
        "global_near_deduplication_complete": False,
        "benchmark_decontamination_complete": False,
        "curriculum_assignment_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality-publication", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_census(args.quality_publication, args.aggregate, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "population_rows": result["population_rows"],
                "population_count": result["population_count"],
                "receipt_sha256": result["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
