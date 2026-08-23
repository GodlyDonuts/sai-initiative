"""Route every bounded pilot identity after the external rights-page probe."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.external_rights_page_probe import (
    PDR_POLICY_URL,
    ExternalRightsPageProbeError,
    _validate_result,
    build_targets,
)
from sai.data.external_rights_page_probe import (
    SCHEMA as PAGE_PROBE_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-external-rights-adjudication-queue-v1"
RECORD_SCHEMA = "sai-external-rights-adjudication-record-v1"
ROUTES = {
    "pressbooks_observed": "adjudicate_book_and_section_license_scope",
    "pdr_essay_observed": "adjudicate_essay_editorial_license_and_exceptions",
    "pdr_policy_observed": "adjudicate_policy_scope_and_embedded_third_party_material",
    "response_without_expected_evidence": (
        "review_noncanonical_or_conflicting_declaration"
    ),
    "access_blocked": "obtain_source_owned_rights_statement_without_access_bypass",
    "source_missing": "resolve_moved_withdrawn_or_replaced_source",
    "transport_unresolved": "manual_transport_and_source_existence_review",
    "unexpected_response": "manual_unexpected_response_review",
}


class ExternalRightsAdjudicationQueueError(RuntimeError):
    """A page-probe binding, identity mapping, or routing result differs."""


def _route(result: dict[str, Any]) -> str:
    if result["expected_license_evidence_observed"]:
        if result["scope"] == "pressbooks_work_page":
            return ROUTES["pressbooks_observed"]
        if result["scope"] == "public_domain_review_essay_page":
            return ROUTES["pdr_essay_observed"]
        if result["scope"] == "public_domain_review_collection_conjecture_policy":
            return ROUTES["pdr_policy_observed"]
        raise ExternalRightsAdjudicationQueueError("observed rights scope differs")
    status = result["http_status"]
    if status == 200:
        return ROUTES["response_without_expected_evidence"]
    if status in {401, 403}:
        return ROUTES["access_blocked"]
    if status in {404, 410}:
        return ROUTES["source_missing"]
    if status is None:
        return ROUTES["transport_unresolved"]
    return ROUTES["unexpected_response"]


def _record_target_key(record: dict[str, Any]) -> tuple[str, str, str]:
    source_id = record["source_id"]
    metadata = record["source_metadata"]
    if source_id == "common_pile_pressbooks":
        scope = "pressbooks_work_page"
        url = metadata.get("metadata.book_url")
    elif source_id == "common_pile_public_domain_review":
        if metadata.get("type") == "essay":
            scope = "public_domain_review_essay_page"
            url = metadata.get("metadata.url")
        elif metadata.get("type") in {"collection", "conjecture"}:
            scope = "public_domain_review_collection_conjecture_policy"
            url = PDR_POLICY_URL
        else:
            raise ExternalRightsAdjudicationQueueError("PDR queue type differs")
    else:
        raise ExternalRightsAdjudicationQueueError("queue source differs")
    if not isinstance(url, str) or not url:
        raise ExternalRightsAdjudicationQueueError("queue source URL differs")
    return source_id, scope, url


def build_queue(
    provenance_roots: list[Path],
    page_probe_root: Path,
    output_root: Path,
    *,
    target_builder: Callable[
        [list[Path]],
        tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]],
    ] = build_targets,
) -> dict[str, Any]:
    """Join exact page outcomes back to all records and seal review routes."""

    if output_root.exists() or output_root.is_symlink():
        raise ExternalRightsAdjudicationQueueError("queue output differs")
    probe_path = page_probe_root / "receipt.json"
    probe = _load_receipt(probe_path)
    results_descriptor = probe.get("results")
    if (
        probe.get("schema") != PAGE_PROBE_SCHEMA
        or probe.get("status") != "complete_text_free_page_evidence_probe"
        or probe.get("source_page_text_persisted") is not False
        or probe.get("rights_provenance_verified") is not False
        or probe.get("legal_clearance_established") is not False
        or probe.get("training_ready") is not False
        or not isinstance(results_descriptor, dict)
    ):
        raise ExternalRightsAdjudicationQueueError("page probe receipt differs")
    targets, provenance, records = target_builder(provenance_roots)
    if (
        probe.get("provenance_bindings") != provenance
        or probe.get("targets", {}).get("count") != len(targets)
        or probe.get("population_records") != len(records)
        or probe.get("records_covered_by_targets") != len(records)
    ):
        raise ExternalRightsAdjudicationQueueError("page probe population differs")
    result_path = _bound_file(page_probe_root, results_descriptor)
    targets_by_sha = {target["target_sha256"]: target for target in targets}
    targets_by_key = {
        (target["source_id"], target["scope"], target["url"]): target
        for target in targets
    }
    if len(targets_by_sha) != len(targets) or len(targets_by_key) != len(targets):
        raise ExternalRightsAdjudicationQueueError("queue target index differs")
    results_by_sha = {}
    with result_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                result = json.loads(line)
                target = targets_by_sha[result["target_sha256"]]
                result = _validate_result(result, target)
            except (
                KeyError,
                json.JSONDecodeError,
                ExternalRightsPageProbeError,
            ) as error:
                raise ExternalRightsAdjudicationQueueError(
                    f"page result {line_number} differs"
                ) from error
            if result["target_sha256"] in results_by_sha:
                raise ExternalRightsAdjudicationQueueError("queue result repeats")
            results_by_sha[result["target_sha256"]] = result
    if set(results_by_sha) != set(targets_by_sha):
        raise ExternalRightsAdjudicationQueueError("queue result coverage differs")

    grouped_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        try:
            target = targets_by_key[_record_target_key(record)]
        except KeyError as error:
            raise ExternalRightsAdjudicationQueueError(
                "queue record target differs"
            ) from error
        grouped_records[target["target_sha256"]].append(record)
    for target_sha256, grouped in grouped_records.items():
        target = targets_by_sha[target_sha256]
        if target["record_count"] != len(grouped) or target[
            "ordered_identity_sha256"
        ] != canonical_sha256([row["identity_sha256"] for row in grouped]):
            raise ExternalRightsAdjudicationQueueError(
                "queue grouped identity binding differs"
            )
    if set(grouped_records) != set(targets_by_sha):
        raise ExternalRightsAdjudicationQueueError("queue grouped coverage differs")

    queue = []
    for target_sha256, grouped in grouped_records.items():
        result = results_by_sha[target_sha256]
        route = _route(result)
        for source in grouped:
            row = {
                "schema": RECORD_SCHEMA,
                "identity_sha256": source["identity_sha256"],
                "source_id": source["source_id"],
                "declared_license": source["declared_license"],
                "source_provenance_record_sha256": source["record_sha256"],
                "page_probe_target_sha256": target_sha256,
                "page_probe_result_sha256": result["result_sha256"],
                "scope": result["scope"],
                "http_status": result["http_status"],
                "response_sha256": result["response_sha256"],
                "expected_license_evidence_observed": result[
                    "expected_license_evidence_observed"
                ],
                "observed_pattern": result["observed_pattern"],
                "adjudication_route": route,
                "rights_provenance_verified": False,
                "legal_clearance_established": False,
                "training_ready": False,
            }
            row["record_sha256"] = canonical_sha256(row)
            queue.append(row)
    queue.sort(key=lambda row: row["identity_sha256"])
    if len(queue) != len(records) or len(
        {row["identity_sha256"] for row in queue}
    ) != len(records):
        raise ExternalRightsAdjudicationQueueError("queue identity coverage differs")

    output_root.mkdir(parents=True)
    try:
        output_path = output_root / "adjudication_queue.jsonl"
        with output_path.open("x") as handle:
            for row in queue:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
        route_counts = Counter(row["adjudication_route"] for row in queue)
        source_counts = Counter(row["source_id"] for row in queue)
        payload = {
            "schema": SCHEMA,
            "status": "complete_text_free_fail_closed_queue",
            "page_probe": {
                "root_name": page_probe_root.name,
                "receipt_file_sha256": sha256_file(probe_path),
                "receipt_sha256": probe["receipt_sha256"],
                "results_sha256": results_descriptor["sha256"],
            },
            "provenance_bindings": provenance,
            "population_records": len(records),
            "queue": {
                "path": output_path.name,
                "rows": len(queue),
                "bytes": output_path.stat().st_size,
                "sha256": sha256_file(output_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in queue]
                ),
            },
            "records_by_source": dict(sorted(source_counts.items())),
            "records_by_adjudication_route": dict(sorted(route_counts.items())),
            "records_with_observed_license_evidence": sum(
                row["expected_license_evidence_observed"] for row in queue
            ),
            "exact_identity_coverage": True,
            "source_text_persisted": False,
            "source_page_text_persisted": False,
            "automated_legal_decision_made": False,
            "access_control_bypassed": False,
            "rights_provenance_verified": False,
            "legal_clearance_established": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance-root", type=Path, action="append", required=True)
    parser.add_argument("--page-probe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = build_queue(
        args.provenance_root,
        args.page_probe_root,
        args.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
