"""Promote strong bounded Common Pile evidence to full candidate materialization."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_compiler_aggregate import (
    load_population,
    validate_shard_summaries,
)
from sai.data.bounded_pilot_work_queue import _load_aggregate
from sai.data.common_pile_streaming_pilot import SCHEMA as PILOT_SCHEMA
from sai.data.data_yield_ledger import _load_receipt
from sai.data.nous_compiler_worker import COMPILER_REASONING_EFFORT
from sai.data.reservoir_audit_aggregate import (
    _triage_route,
    _validate_compiler_receipt,
    summarize,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-common-pile-full-source-candidate-promotion-v1"
METHOD = {
    "minimum_bounded_rows": 1_024,
    "minimum_retain_ppm": 850_000,
    "maximum_reject_ppm": 50_000,
    "maximum_quarantine_ppm": 150_000,
    "maximum_rights_hold_ppm": 25_000,
    "minimum_educational_value_milli": 2_500,
    "minimum_source_reliability_milli": 3_000,
    "minimum_coherence_milli": 3_000,
}


class CommonPileFullSourcePromotionError(RuntimeError):
    """The bounded evidence, source coverage, or promotion output differs."""


def _ppm(value: int, total: int) -> int:
    return (value * 1_000_000) // total


def _load_pilot(path: Path) -> dict[str, Any]:
    try:
        payload = _load_receipt(path)
    except Exception as error:  # noqa: BLE001 - normalize receipt loader failures
        raise CommonPileFullSourcePromotionError("pilot receipt differs") from error
    decontamination = payload.get("decontamination")
    near_duplicate = payload.get("near_duplicate_filter")
    scan = payload.get("scan")
    if (
        payload.get("schema") != PILOT_SCHEMA
        or payload.get("status") != "complete_nontraining_pilot"
        or payload.get("full_source_ingestion_authorized") is not False
        or payload.get("global_cross_source_near_duplicate_filter_complete")
        is not False
        or payload.get("training_ready") is not False
        or payload.get("four_b_training_authorized") is not False
        or not isinstance(decontamination, dict)
        or not isinstance(near_duplicate, dict)
        or not isinstance(scan, dict)
        or scan.get("selected_rows") != decontamination.get("scanned")
        or decontamination.get("accepted")
        != near_duplicate.get("input_documents")
        or near_duplicate.get("input_documents")
        != near_duplicate.get("output_documents")
        + near_duplicate.get("documents_dropped")
    ):
        raise CommonPileFullSourcePromotionError("pilot receipt differs")
    return payload


def _source_evidence(
    source_id: str,
    lineage: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = [
        receipt
        for source, receipt in zip(lineage, receipts, strict=True)
        if source.get("source_id") == source_id
    ]
    if not selected:
        raise CommonPileFullSourcePromotionError("promotion source is absent")
    verdicts = Counter()
    routes = Counter()
    score_sums = Counter()
    for receipt in selected:
        judgment = receipt["judgment"]
        verdicts[judgment["verdict"]] += 1
        routes[_triage_route(judgment)] += 1
        score_sums.update(judgment["scores"])
    rows = len(selected)
    means = {
        key: (score_sums[key] * 1_000) // rows
        for key in (
            "educational_value",
            "source_reliability",
            "coherence",
        )
    }
    metrics = {
        "bounded_rows": rows,
        "verdict_counts": dict(sorted(verdicts.items())),
        "triage_counts": dict(sorted(routes.items())),
        "retain_ppm": _ppm(verdicts["retain"], rows),
        "reject_ppm": _ppm(verdicts["reject"], rows),
        "quarantine_ppm": _ppm(routes["quarantine"], rows),
        "rights_hold_ppm": _ppm(routes["rights_hold"], rows),
        "mean_scores_milli": means,
    }
    checks = {
        "minimum_bounded_rows": rows >= METHOD["minimum_bounded_rows"],
        "minimum_retain_ppm": metrics["retain_ppm"]
        >= METHOD["minimum_retain_ppm"],
        "maximum_reject_ppm": metrics["reject_ppm"]
        <= METHOD["maximum_reject_ppm"],
        "maximum_quarantine_ppm": metrics["quarantine_ppm"]
        <= METHOD["maximum_quarantine_ppm"],
        "maximum_rights_hold_ppm": metrics["rights_hold_ppm"]
        <= METHOD["maximum_rights_hold_ppm"],
        "minimum_educational_value_milli": means["educational_value"]
        >= METHOD["minimum_educational_value_milli"],
        "minimum_source_reliability_milli": means["source_reliability"]
        >= METHOD["minimum_source_reliability_milli"],
        "minimum_coherence_milli": means["coherence"]
        >= METHOD["minimum_coherence_milli"],
    }
    return {
        **metrics,
        "checks": checks,
        "failed_checks": sorted(key for key, passed in checks.items() if not passed),
        "full_source_candidate_materialization_authorized": all(checks.values()),
    }


def build_promotion(
    population_root: Path,
    judgments_root: Path,
    aggregate_path: Path,
    pilot_receipt_paths: list[Path],
    output_path: Path,
) -> dict[str, Any]:
    """Replay final bounded evidence and seal candidate-only source decisions."""

    if (
        output_path.exists()
        or output_path.is_symlink()
        or not pilot_receipt_paths
        or len(pilot_receipt_paths) != len(set(pilot_receipt_paths))
    ):
        raise CommonPileFullSourcePromotionError("promotion output boundary differs")
    aggregate = _load_aggregate(aggregate_path)
    candidates, lineage, population = load_population(population_root)
    receipts = []
    expected_paths = set()
    for candidate in candidates:
        identity = candidate["candidate_identity_sha256"]
        path = judgments_root / f"{identity}.compiler.json"
        expected_paths.add(path)
        receipt = _validate_compiler_receipt(_load_receipt(path), candidate)
        if receipt.get("request_reasoning_effort") != COMPILER_REASONING_EFFORT:
            raise CommonPileFullSourcePromotionError("reasoning effort differs")
        receipts.append(receipt)
    if set(judgments_root.glob("*.compiler.json")) != expected_paths:
        raise CommonPileFullSourcePromotionError("judgment population differs")
    shard_hashes = validate_shard_summaries(candidates, judgments_root)
    summary_lineage = [
        {"source_id": row["source_id"], "stratum": row["source_type"]}
        for row in lineage
    ]
    compiler_summary = summarize(summary_lineage, receipts)
    if (
        aggregate.get("population", {}).get("receipt_sha256")
        != population.get("receipt_sha256")
        or aggregate.get("population", {}).get("rows") != len(candidates)
        or aggregate.get("compiler_summary") != compiler_summary
        or aggregate.get("ordered_shard_summaries_sha256")
        != canonical_sha256(shard_hashes)
        or population.get("full_bounded_cross_source_survivor_coverage") is not True
        or population.get("cross_source_filter", {}).get("documents_dropped") != 0
    ):
        raise CommonPileFullSourcePromotionError("aggregate evidence differs")

    pilots = [_load_pilot(path) for path in pilot_receipt_paths]
    by_source = {pilot.get("source_id"): pilot for pilot in pilots}
    if len(by_source) != len(pilots) or set(by_source) != set(
        population.get("by_source", {})
    ):
        raise CommonPileFullSourcePromotionError("pilot source coverage differs")
    decisions = []
    for source_id in sorted(by_source):
        pilot = by_source[source_id]
        evidence = _source_evidence(source_id, lineage, receipts)
        pilot_rows = pilot["near_duplicate_filter"]["output_documents"]
        if pilot_rows != population["by_source"][source_id] or pilot_rows != evidence[
            "bounded_rows"
        ]:
            raise CommonPileFullSourcePromotionError("pilot row coverage differs")
        decisions.append(
            {
                "source_id": source_id,
                "pilot_receipt": {
                    "path": str(pilot_receipt_paths[pilots.index(pilot)]),
                    "file_sha256": sha256_file(
                        pilot_receipt_paths[pilots.index(pilot)]
                    ),
                    "receipt_sha256": pilot["receipt_sha256"],
                },
                "parent": pilot["parent"],
                "benchmark_decontamination_dropped_rows": pilot[
                    "decontamination"
                ]["dropped"],
                "within_pilot_near_duplicate_dropped_rows": pilot[
                    "near_duplicate_filter"
                ]["documents_dropped"],
                **evidence,
                "bulk_training_admission": False,
                "training_ready": False,
            }
        )
    authorized = [
        row["source_id"]
        for row in decisions
        if row["full_source_candidate_materialization_authorized"]
    ]
    payload = {
        "schema": SCHEMA,
        "status": "complete_candidate_only_source_decision",
        "method": METHOD,
        "aggregate": {
            "path": str(aggregate_path),
            "file_sha256": sha256_file(aggregate_path),
            "receipt_sha256": aggregate["receipt_sha256"],
        },
        "population_receipt_sha256": population["receipt_sha256"],
        "bounded_rows": len(candidates),
        "sources": decisions,
        "authorized_source_ids": authorized,
        "full_source_candidate_materialization_authorized": bool(authorized),
        "authorization_is_per_source": True,
        "raw_source_files_are_training_data": False,
        "full_source_materialization_is_training_admission": False,
        "rights_provenance_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--pilot-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_promotion(
        args.population_root,
        args.judgments_root,
        args.aggregate,
        args.pilot_receipt,
        args.output,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
