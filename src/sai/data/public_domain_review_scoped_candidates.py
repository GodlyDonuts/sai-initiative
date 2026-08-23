"""Materialize quote-excluded Public Domain Review candidate text exactly."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.public_domain_review_scope_audit import (
    MAXIMUM_RESPONSE_BYTES,
    SOURCE_ID,
    UPSTREAM_COLLECTOR,
    USER_AGENT,
    _load_inputs,
    reconstruct_page,
)
from sai.data.public_domain_review_scope_audit import (
    RESULT_SCHEMA as SCOPE_RESULT_SCHEMA,
)
from sai.data.public_domain_review_scope_audit import (
    SCHEMA as SCOPE_AUDIT_SCHEMA,
)
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-public-domain-review-scoped-candidates-v1"
CANDIDATE_SCHEMA = "sai-public-domain-review-scoped-candidate-v1"
RESULT_SCHEMA = "sai-public-domain-review-materialization-result-v1"


class PublicDomainReviewMaterializationError(RuntimeError):
    """The frozen scope evidence, live page, or materialized output differs."""


def _load_scope_audit(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipt = _load_receipt(root / "receipt.json")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    descriptor = receipt.get("results")
    method = receipt.get("method")
    if (
        receipt.get("schema") != SCOPE_AUDIT_SCHEMA
        or receipt.get("status") != "complete_text_free_scope_audit"
        or receipt.get("receipt_sha256") != canonical_sha256(unsigned)
        or receipt.get("upstream_collector") != UPSTREAM_COLLECTOR
        or receipt.get("source_page_text_persisted") is not False
        or receipt.get("scoped_text_persisted") is not False
        or receipt.get("legal_clearance_established") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(descriptor, dict)
        or not isinstance(method, dict)
        or method.get("source_page_bodies_persisted") is not False
        or method.get("scoped_text_persisted") is not False
    ):
        raise PublicDomainReviewMaterializationError("scope audit receipt differs")
    path = _bound_file(root, descriptor)
    rows = []
    by_identity = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            identity = row.get("candidate_identity_sha256")
            row_unsigned = {
                key: value for key, value in row.items() if key != "result_sha256"
            }
            if (
                row.get("schema") != SCOPE_RESULT_SCHEMA
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in by_identity
                or row.get("result_sha256") != canonical_sha256(row_unsigned)
                or row.get("source_page_text_persisted") is not False
                or row.get("scoped_text_persisted") is not False
                or row.get("legal_clearance_established") is not False
                or row.get("training_ready") is not False
            ):
                raise PublicDomainReviewMaterializationError("scope audit row differs")
            rows.append(row)
            by_identity[identity] = row
    if (
        len(rows) != descriptor.get("rows")
        or len(rows) != 1_342
        or descriptor.get("ordered_results_sha256")
        != canonical_sha256([row["result_sha256"] for row in rows])
    ):
        raise PublicDomainReviewMaterializationError("scope audit coverage differs")
    return by_identity, receipt


def fetch_page(
    url: str, *, timeout_seconds: float, maximum_attempts: int
) -> dict[str, Any]:
    """Fetch one bounded source page without writing its response body to disk."""

    if (
        not isinstance(url, str)
        or not url.startswith("https://publicdomainreview.org/")
        or not 1 <= timeout_seconds <= 120
        or not 1 <= maximum_attempts <= 5
    ):
        raise PublicDomainReviewMaterializationError("page request differs")
    attempts = []
    response_bytes = b""
    http_status = None
    final_url = None
    error_type = None
    for attempt in range(1, maximum_attempts + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Encoding": "identity",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                http_status = response.status
                final_url = response.geturl()
                response_bytes = response.read(MAXIMUM_RESPONSE_BYTES + 1)
            attempts.append(
                {"attempt": attempt, "outcome": "response", "status": http_status}
            )
            break
        except urllib.error.HTTPError as error:
            http_status = error.code
            error_type = "http_error"
            attempts.append(
                {"attempt": attempt, "outcome": "http_error", "status": error.code}
            )
            if error.code not in {408, 425, 429, 500, 502, 503, 504}:
                break
        except (TimeoutError, urllib.error.URLError, OSError) as error:
            error_type = type(error).__name__
            attempts.append(
                {"attempt": attempt, "outcome": "transport_error", "status": None}
            )
        if attempt < maximum_attempts:
            time.sleep(float(2 ** (attempt - 1)))
    return {
        "attempts": attempts,
        "http_status": http_status,
        "final_url": final_url,
        "error_type": error_type,
        "response_bytes": response_bytes,
    }


def materialize_response(
    candidate: dict[str, Any],
    provenance: dict[str, Any],
    rights: dict[str, Any],
    scope: dict[str, Any],
    fetched: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Bind one live response to the active scope hash or route it to review."""

    identity = candidate.get("identity_sha256")
    source = candidate.get("source")
    metadata = provenance.get("source_metadata")
    response_bytes = fetched.get("response_bytes")
    final_url = fetched.get("final_url")
    if (
        scope.get("candidate_identity_sha256") != identity
        or scope.get("status") != "scope_reconstruction_complete"
        or scope.get("rights_scope_reconstruction_complete") is not True
        or provenance.get("record_sha256")
        != scope.get("source_provenance_record_sha256")
        or rights.get("record_sha256") != scope.get("rights_record_sha256")
        or rights.get("identity_sha256") != identity
        or rights.get("source_id") != SOURCE_ID
        or not isinstance(source, dict)
        or source.get("license") != "CC-BY-SA-4.0"
        or not isinstance(metadata, dict)
        or metadata.get("metadata.url") != scope.get("source_url")
        or metadata.get("type") != scope.get("source_type")
        or not isinstance(response_bytes, bytes)
    ):
        raise PublicDomainReviewMaterializationError("materialization binding differs")
    final_url_on_source_host = bool(
        isinstance(final_url, str)
        and urlsplit(final_url).scheme == "https"
        and urlsplit(final_url).netloc.lower() == "publicdomainreview.org"
    )
    response_truncated = len(response_bytes) > MAXIMUM_RESPONSE_BYTES
    response_bytes = response_bytes[:MAXIMUM_RESPONSE_BYTES]
    base_result = {
        "schema": RESULT_SCHEMA,
        "candidate_identity_sha256": identity,
        "scope_audit_result_sha256": scope["result_sha256"],
        "source_provenance_record_sha256": provenance["record_sha256"],
        "rights_record_sha256": rights["record_sha256"],
        "source_type": scope["source_type"],
        "source_url": scope["source_url"],
        "attempts": fetched.get("attempts"),
        "http_status": fetched.get("http_status"),
        "final_url": final_url,
        "final_url_on_source_host": final_url_on_source_host,
        "error_type": fetched.get("error_type"),
        "response_bytes_inspected": len(response_bytes),
        "response_sha256": (
            hashlib.sha256(response_bytes).hexdigest() if response_bytes else None
        ),
        "response_truncated": response_truncated,
        "source_page_html_persisted": False,
        "source_text_persisted_in_result": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }
    if (
        fetched.get("http_status") != 200
        or not response_bytes
        or response_truncated
        or not final_url_on_source_host
    ):
        result = {
            **base_result,
            "status": "materialization_transport_or_response_review",
            "scoped_candidate_record_sha256": None,
        }
        result["result_sha256"] = canonical_sha256(result)
        return result, None
    try:
        replay = reconstruct_page(response_bytes, scope["source_type"])
    except Exception as error:  # noqa: BLE001 - fail one remote page closed
        result = {
            **base_result,
            "status": "materialization_page_parse_review",
            "parse_error_type": type(error).__name__,
            "scoped_candidate_record_sha256": None,
        }
        result["result_sha256"] = canonical_sha256(result)
        return result, None
    scoped_text = replay["scoped_text"]
    license_scope_observed = (
        scope["source_type"] == "collection"
        or replay["page_specific_cc_by_sa_observed"]
    )
    exact = (
        replay["frozen_geometry_text"] == candidate["text"]
        and hashlib.sha256(scoped_text.encode()).hexdigest()
        == scope.get("scoped_text_sha256")
        and len(scoped_text.encode()) == scope.get("scoped_text_bytes")
        and replay["excluded_quote_elements"] == scope.get("excluded_quote_elements")
        and replay["excluded_quote_codepoints"]
        == scope.get("excluded_quote_codepoints")
        and license_scope_observed
    )
    if not exact:
        result = {
            **base_result,
            "status": "materialization_source_or_scope_drift_review",
            "scoped_candidate_record_sha256": None,
        }
        result["result_sha256"] = canonical_sha256(result)
        return result, None
    scoped_candidate = {
        "schema": CANDIDATE_SCHEMA,
        "text": scoped_text,
        "source": source,
        "original_candidate_identity_sha256": identity,
        "source_provenance_record_sha256": provenance["record_sha256"],
        "rights_record_sha256": rights["record_sha256"],
        "scope_audit_result_sha256": scope["result_sha256"],
        "source_url": scope["source_url"],
        "source_type": scope["source_type"],
        "scoped_text_bytes": len(scoped_text.encode()),
        "scoped_text_sha256": hashlib.sha256(scoped_text.encode()).hexdigest(),
        "excluded_quote_elements": replay["excluded_quote_elements"],
        "excluded_quote_codepoints": replay["excluded_quote_codepoints"],
        "page_response_sha256": base_result["response_sha256"],
        "rights_scope_evidence_observed": True,
        "attribution_required": True,
        "share_alike_required": True,
        "source_page_replay_complete": True,
        "content_quality_verified": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }
    scoped_candidate["record_sha256"] = canonical_sha256(scoped_candidate)
    result = {
        **base_result,
        "status": "materialized_exact_scoped_candidate",
        "scoped_candidate_record_sha256": scoped_candidate["record_sha256"],
    }
    result["result_sha256"] = canonical_sha256(result)
    return result, scoped_candidate


def _ineligible_result(scope: dict[str, Any]) -> dict[str, Any]:
    result = {
        "schema": RESULT_SCHEMA,
        "candidate_identity_sha256": scope["candidate_identity_sha256"],
        "scope_audit_result_sha256": scope["result_sha256"],
        "source_provenance_record_sha256": scope["source_provenance_record_sha256"],
        "rights_record_sha256": scope["rights_record_sha256"],
        "source_type": scope["source_type"],
        "source_url": scope["source_url"],
        "status": "scope_audit_not_eligible",
        "scope_audit_status": scope["status"],
        "scoped_candidate_record_sha256": None,
        "source_page_html_persisted": False,
        "source_text_persisted_in_result": False,
        "legal_clearance_established": False,
        "training_ready": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def build_candidates(
    pilot_root: Path,
    provenance_root: Path,
    rights_root: Path,
    scope_audit_root: Path,
    output_root: Path,
    *,
    concurrency: int,
    timeout_seconds: float,
    maximum_attempts: int,
    fetch_function: Callable[..., dict[str, Any]] = fetch_page,
) -> dict[str, Any]:
    """Replay the active audit and persist only exact scoped candidates."""

    if output_root.exists() or output_root.is_symlink() or not 1 <= concurrency <= 2:
        raise PublicDomainReviewMaterializationError(
            "materialization output or concurrency differs"
        )
    candidates, provenance_by_identity, inputs = _load_inputs(
        pilot_root, provenance_root, rights_root
    )
    rights_by_identity = inputs["rights_by_identity"]
    scope_by_identity, scope_receipt = _load_scope_audit(scope_audit_root)
    candidate_by_identity = {row["identity_sha256"]: row for row in candidates}
    identities = set(candidate_by_identity)
    if (
        identities != set(scope_by_identity)
        or scope_receipt.get("pilot", {}).get("receipt_sha256")
        != inputs["pilot"]["receipt_sha256"]
        or scope_receipt.get("provenance", {}).get("receipt_sha256")
        != inputs["provenance"]["receipt_sha256"]
        or scope_receipt.get("rights_receipt_sha256")
        != inputs["rights"]["receipt_sha256"]
    ):
        raise PublicDomainReviewMaterializationError("scope input binding differs")
    eligible = [
        identity
        for identity, scope in scope_by_identity.items()
        if scope["status"] == "scope_reconstruction_complete"
    ]
    results = [
        _ineligible_result(scope)
        for scope in scope_by_identity.values()
        if scope["status"] != "scope_reconstruction_complete"
    ]
    materialized = []

    def work(identity: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        scope = scope_by_identity[identity]
        fetched = fetch_function(
            scope["source_url"],
            timeout_seconds=timeout_seconds,
            maximum_attempts=maximum_attempts,
        )
        return materialize_response(
            candidate_by_identity[identity],
            provenance_by_identity[identity],
            rights_by_identity[identity],
            scope,
            fetched,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(work, identity): identity for identity in eligible}
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            result, candidate = future.result()
            if result["candidate_identity_sha256"] != futures[future]:
                raise PublicDomainReviewMaterializationError(
                    "materialization identity differs"
                )
            results.append(result)
            if candidate is not None:
                materialized.append(candidate)
            if completed % 100 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "event": "pdr_materialization_progress",
                            "complete": completed,
                            "remaining": len(futures) - completed,
                            "materialized": len(materialized),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    results.sort(key=lambda row: row["candidate_identity_sha256"])
    materialized.sort(key=lambda row: row["original_candidate_identity_sha256"])
    if (
        len(results) != len(candidates)
        or len({row["candidate_identity_sha256"] for row in results}) != len(candidates)
        or any(
            row["result_sha256"]
            != canonical_sha256(
                {key: value for key, value in row.items() if key != "result_sha256"}
            )
            for row in results
        )
        or any(
            row["record_sha256"]
            != canonical_sha256(
                {key: value for key, value in row.items() if key != "record_sha256"}
            )
            for row in materialized
        )
    ):
        raise PublicDomainReviewMaterializationError(
            "materialization result coverage differs"
        )
    status_counts = Counter(row["status"] for row in results)
    output_root.mkdir(parents=True)
    try:
        candidate_path = output_root / "scoped_candidates.jsonl"
        result_path = output_root / "materialization_results.jsonl"
        _atomic_jsonl(candidate_path, materialized)
        _atomic_jsonl(result_path, results)
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_scoped_candidates",
            "pilot_receipt_sha256": inputs["pilot"]["receipt_sha256"],
            "provenance_receipt_sha256": inputs["provenance"]["receipt_sha256"],
            "rights_receipt_sha256": inputs["rights"]["receipt_sha256"],
            "scope_audit": {
                "root_name": scope_audit_root.name,
                "receipt_file_sha256": sha256_file(scope_audit_root / "receipt.json"),
                "receipt_sha256": scope_receipt["receipt_sha256"],
                "results_sha256": scope_receipt["results"]["sha256"],
            },
            "eligible_scope_audit_rows": len(eligible),
            "population_rows": len(candidates),
            "results": {
                "path": result_path.name,
                "rows": len(results),
                "bytes": result_path.stat().st_size,
                "sha256": sha256_file(result_path),
                "ordered_results_sha256": canonical_sha256(
                    [row["result_sha256"] for row in results]
                ),
            },
            "scoped_candidates": {
                "path": candidate_path.name,
                "rows": len(materialized),
                "bytes": candidate_path.stat().st_size,
                "sha256": sha256_file(candidate_path),
                "ordered_records_sha256": canonical_sha256(
                    [row["record_sha256"] for row in materialized]
                ),
                "text_bytes": sum(row["scoped_text_bytes"] for row in materialized),
            },
            "records_by_materialization_status": dict(sorted(status_counts.items())),
            "source_page_html_persisted": False,
            "source_text_persisted_in_candidate_file": True,
            "source_text_persisted_in_result_file": False,
            "rights_scope_evidence_observed_for_every_candidate": True,
            "content_quality_verified": False,
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
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--provenance-root", type=Path, required=True)
    parser.add_argument("--rights-root", type=Path, required=True)
    parser.add_argument("--scope-audit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    args = parser.parse_args()
    result = build_candidates(
        args.pilot_root,
        args.provenance_root,
        args.rights_root,
        args.scope_audit_root,
        args.output_root,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        maximum_attempts=args.maximum_attempts,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
