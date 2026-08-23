"""Probe source-owned pages for declared-license evidence without storing HTML."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sai.data.agent_labeling import _atomic_create
from sai.data.common_pile_external_provenance import (
    RECORD_SCHEMA as PROVENANCE_RECORD_SCHEMA,
)
from sai.data.common_pile_external_provenance import SCHEMA as PROVENANCE_SCHEMA
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-external-rights-page-probe-v1"
PDR_POLICY_URL = "https://publicdomainreview.org/reusing-material/"
MAXIMUM_RESPONSE_BYTES = 2 << 20
USER_AGENT = "SaiDataRightsAudit/1.0 (+https://github.com/GodlyDonuts/sai-initiative)"
LICENSE_PATTERNS = {
    "CC-BY-4.0": (b"creativecommons.org/licenses/by/4.0",),
    "CC-BY-SA-4.0": (b"creativecommons.org/licenses/by-sa/4.0",),
    "CC0-1.0": (b"creativecommons.org/publicdomain/zero/1.0", b"cc0 1.0"),
    "PUBLIC-DOMAIN": (
        b"creativecommons.org/publicdomain/mark/1.0",
        b"public domain",
    ),
}
DECLARATION_LICENSES = {
    (
        "Creative Commons - Attribution - "
        "https://creativecommons.org/licenses/by/4.0/"
    ): "CC-BY-4.0",
    (
        "Creative Commons - Attribution Share-Alike - "
        "https://creativecommons.org/licenses/by-sa/4.0/"
    ): "CC-BY-SA-4.0",
    (
        "Creative Commons Zero - Public Domain - "
        "https://creativecommons.org/publicdomain/zero/1.0/"
    ): "CC0-1.0",
    "Public Domain": "PUBLIC-DOMAIN",
}
LICENSE_PATTERN_MANIFEST = {
    key: [pattern.decode() for pattern in patterns]
    for key, patterns in LICENSE_PATTERNS.items()
}
EXPECTED_SOURCE_IDS = {
    "common_pile_pressbooks",
    "common_pile_public_domain_review",
}
RESULT_FIELDS = {
    "attempts",
    "content_type",
    "error_type",
    "expected_license",
    "expected_license_evidence_observed",
    "final_url",
    "http_status",
    "observed_pattern",
    "ordered_identity_sha256",
    "record_count",
    "response_bytes_inspected",
    "response_sha256",
    "response_truncated",
    "result_sha256",
    "scope",
    "source_id",
    "source_page_text_persisted",
    "target_sha256",
    "url",
}


class ExternalRightsPageProbeError(RuntimeError):
    """A provenance population, rights target, or probe result differs."""


def _load_records(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = _load_receipt(root / "receipt.json")
    output = receipt.get("output")
    if (
        receipt.get("schema") != PROVENANCE_SCHEMA
        or receipt.get("status") != "complete_text_free_source_metadata_replay"
        or receipt.get("source_text_persisted") is not False
        or receipt.get("training_ready") is not False
        or not isinstance(output, dict)
    ):
        raise ExternalRightsPageProbeError("provenance receipt differs")
    path = _bound_file(root, output)
    records = []
    identities = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ExternalRightsPageProbeError(
                    f"provenance row {line_number} cannot be decoded"
                ) from error
            if not isinstance(row, dict):
                raise ExternalRightsPageProbeError(
                    f"provenance row {line_number} differs"
                )
            unsigned = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != PROVENANCE_RECORD_SCHEMA
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in identities
                or row.get("record_sha256") != canonical_sha256(unsigned)
                or not isinstance(row.get("source_metadata"), dict)
                or row.get("declared_license") not in DECLARATION_LICENSES
            ):
                raise ExternalRightsPageProbeError(
                    f"provenance row {line_number} differs"
                )
            identities.add(identity)
            records.append(row)
    if len(records) != output.get("rows"):
        raise ExternalRightsPageProbeError("provenance row coverage differs")
    return receipt, records


def build_targets(
    provenance_roots: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Group book rows, select essay pages, and bind PDR policy-scoped rows."""

    if len(provenance_roots) != 2 or len(provenance_roots) != len(
        set(provenance_roots)
    ):
        raise ExternalRightsPageProbeError("rights probe provenance roots differ")
    provenance_bindings = []
    all_records = []
    for root in provenance_roots:
        receipt, records = _load_records(root)
        provenance_bindings.append(
            {
                "root_name": root.name,
                "source_id": receipt["source_id"],
                "receipt_file_sha256": sha256_file(root / "receipt.json"),
                "receipt_sha256": receipt["receipt_sha256"],
                "manifest_sha256": receipt["output"]["sha256"],
                "records": len(records),
            }
        )
        all_records.extend(records)
    if {row["source_id"] for row in provenance_bindings} != EXPECTED_SOURCE_IDS:
        raise ExternalRightsPageProbeError("rights probe source roots differ")
    identities = [row["identity_sha256"] for row in all_records]
    if len(identities) != len(set(identities)):
        raise ExternalRightsPageProbeError("rights probe repeats a source identity")
    pressbooks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pdr_policy = []
    targets = []
    for row in all_records:
        source_id = row["source_id"]
        metadata = row["source_metadata"]
        expected = DECLARATION_LICENSES[row["declared_license"]]
        if source_id == "common_pile_pressbooks":
            url = metadata.get("metadata.book_url")
            if not isinstance(url, str) or not url:
                raise ExternalRightsPageProbeError("Pressbooks work URL differs")
            pressbooks[url].append(row)
        elif source_id == "common_pile_public_domain_review":
            source_type = metadata.get("type")
            if source_type in {"collection", "conjecture"}:
                pdr_policy.append(row)
            elif source_type == "essay":
                url = metadata.get("metadata.url")
                if not isinstance(url, str) or not url:
                    raise ExternalRightsPageProbeError("PDR essay URL differs")
                targets.append(
                    {
                        "scope": "public_domain_review_essay_page",
                        "source_id": source_id,
                        "url": url,
                        "expected_license": expected,
                        "record_count": 1,
                        "ordered_identity_sha256": canonical_sha256(
                            [row["identity_sha256"]]
                        ),
                    }
                )
            else:
                raise ExternalRightsPageProbeError("PDR source type differs")
        else:
            raise ExternalRightsPageProbeError("rights probe source differs")
    for url, rows in pressbooks.items():
        licenses = {DECLARATION_LICENSES[row["declared_license"]] for row in rows}
        if len(licenses) != 1:
            raise ExternalRightsPageProbeError(
                "Pressbooks work has conflicting declarations"
            )
        targets.append(
            {
                "scope": "pressbooks_work_page",
                "source_id": "common_pile_pressbooks",
                "url": url,
                "expected_license": next(iter(licenses)),
                "record_count": len(rows),
                "ordered_identity_sha256": canonical_sha256(
                    [row["identity_sha256"] for row in rows]
                ),
            }
        )
    if not pdr_policy:
        raise ExternalRightsPageProbeError("PDR policy scope is empty")
    policy_licenses = {
        DECLARATION_LICENSES[row["declared_license"]] for row in pdr_policy
    }
    if policy_licenses != {"CC-BY-SA-4.0"}:
        raise ExternalRightsPageProbeError("PDR policy declarations differ")
    targets.append(
        {
            "scope": "public_domain_review_collection_conjecture_policy",
            "source_id": "common_pile_public_domain_review",
            "url": PDR_POLICY_URL,
            "expected_license": "CC-BY-SA-4.0",
            "record_count": len(pdr_policy),
            "ordered_identity_sha256": canonical_sha256(
                [row["identity_sha256"] for row in pdr_policy]
            ),
        }
    )
    for target in targets:
        target["target_sha256"] = canonical_sha256(target)
    targets.sort(key=lambda row: (row["source_id"], row["scope"], row["url"]))
    if len(targets) != len({row["target_sha256"] for row in targets}):
        raise ExternalRightsPageProbeError("rights probe targets repeat")
    return (
        targets,
        sorted(provenance_bindings, key=lambda row: row["source_id"]),
        all_records,
    )


_HOST_SEMAPHORES: dict[str, threading.Semaphore] = {}
_HOST_LOCK = threading.Lock()


def _host_semaphore(url: str) -> threading.Semaphore:
    host = urlsplit(url).netloc.lower()
    with _HOST_LOCK:
        return _HOST_SEMAPHORES.setdefault(host, threading.Semaphore(2))


def fetch_target(
    target: dict[str, Any], *, timeout_seconds: float, maximum_attempts: int
) -> dict[str, Any]:
    """Fetch one bounded response, hash it in memory, and retain no page text."""

    if not 1 <= maximum_attempts <= 5 or not 1 <= timeout_seconds <= 120:
        raise ExternalRightsPageProbeError("rights probe network bounds differ")
    attempts = []
    response_bytes = b""
    final_url = None
    status = None
    content_type = None
    truncated = False
    error_type = None
    for attempt in range(1, maximum_attempts + 1):
        try:
            request = urllib.request.Request(
                target["url"],
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Encoding": "identity",
                },
            )
            with _host_semaphore(target["url"]):
                with urllib.request.urlopen(
                    request, timeout=timeout_seconds
                ) as response:
                    status = response.status
                    final_url = response.geturl()
                    content_type = response.headers.get_content_type()
                    response_bytes = response.read(MAXIMUM_RESPONSE_BYTES + 1)
            truncated = len(response_bytes) > MAXIMUM_RESPONSE_BYTES
            response_bytes = response_bytes[:MAXIMUM_RESPONSE_BYTES]
            attempts.append(
                {"attempt": attempt, "outcome": "response", "status": status}
            )
            break
        except urllib.error.HTTPError as error:
            status = error.code
            attempts.append(
                {"attempt": attempt, "outcome": "http_error", "status": error.code}
            )
            error_type = "http_error"
            if error.code not in {408, 425, 429, 500, 502, 503, 504}:
                break
        except (TimeoutError, urllib.error.URLError, OSError) as error:
            attempts.append(
                {
                    "attempt": attempt,
                    "outcome": "transport_error",
                    "status": None,
                }
            )
            error_type = type(error).__name__
        if attempt < maximum_attempts:
            time.sleep(float(2 ** (attempt - 1)))
    normalized = response_bytes.lower().replace(b"http://", b"https://")
    observed_pattern = next(
        (
            pattern.decode()
            for pattern in LICENSE_PATTERNS[target["expected_license"]]
            if pattern in normalized
        ),
        None,
    )
    result = {
        **target,
        "attempts": attempts,
        "http_status": status,
        "final_url": final_url,
        "content_type": content_type,
        "response_bytes_inspected": len(response_bytes),
        "response_sha256": (
            hashlib.sha256(response_bytes).hexdigest() if response_bytes else None
        ),
        "response_truncated": truncated,
        "expected_license_evidence_observed": observed_pattern is not None,
        "observed_pattern": observed_pattern,
        "error_type": error_type,
        "source_page_text_persisted": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def _validate_result(result: Any, target: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict) or set(result) != RESULT_FIELDS:
        raise ExternalRightsPageProbeError("rights probe result fields differ")
    for key, value in target.items():
        if result.get(key) != value:
            raise ExternalRightsPageProbeError("rights probe result target differs")
    unsigned = {key: value for key, value in result.items() if key != "result_sha256"}
    observed = result.get("observed_pattern")
    response_sha256 = result.get("response_sha256")
    expected_patterns = {
        pattern.decode() for pattern in LICENSE_PATTERNS[target["expected_license"]]
    }
    if (
        result.get("result_sha256") != canonical_sha256(unsigned)
        or result.get("source_page_text_persisted") is not False
        or not isinstance(result.get("attempts"), list)
        or not isinstance(result.get("response_bytes_inspected"), int)
        or not 0 <= result["response_bytes_inspected"] <= MAXIMUM_RESPONSE_BYTES
        or (
            response_sha256 is not None
            and (
                not isinstance(response_sha256, str)
                or len(response_sha256) != 64
                or any(
                    character not in "0123456789abcdef" for character in response_sha256
                )
            )
        )
        or (result["response_bytes_inspected"] == 0) != (response_sha256 is None)
        or not isinstance(result.get("response_truncated"), bool)
        or not isinstance(result.get("expected_license_evidence_observed"), bool)
        or (observed is not None and observed not in expected_patterns)
        or result["expected_license_evidence_observed"] != (observed is not None)
    ):
        raise ExternalRightsPageProbeError("rights probe result differs")
    return result


def build_probe(
    provenance_roots: list[Path],
    output_root: Path,
    *,
    concurrency: int,
    timeout_seconds: float,
    maximum_attempts: int,
    fetch_function: Callable[..., dict[str, Any]] = fetch_target,
) -> dict[str, Any]:
    """Probe every frozen target and seal result/accounting receipts."""

    if output_root.exists() or output_root.is_symlink() or not 1 <= concurrency <= 32:
        raise ExternalRightsPageProbeError("rights probe output or concurrency differs")
    targets, provenance, records = build_targets(provenance_roots)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                fetch_function,
                target,
                timeout_seconds=timeout_seconds,
                maximum_attempts=maximum_attempts,
            ): target
            for target in targets
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            results.append(_validate_result(future.result(), futures[future]))
            if completed % 100 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "event": "rights_page_probe_progress",
                            "complete": completed,
                            "remaining": len(futures) - completed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    results.sort(key=lambda row: row["target_sha256"])
    output_root.mkdir(parents=True)
    try:
        result_path = output_root / "results.jsonl"
        with result_path.open("x") as handle:
            for row in results:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
        match_records = sum(
            row["record_count"]
            for row in results
            if row["expected_license_evidence_observed"]
        )
        source_target_counts = Counter(row["source_id"] for row in results)
        scope_target_counts = Counter(row["scope"] for row in results)
        status_counts = Counter(str(row["http_status"]) for row in results)
        all_targets_matched = all(
            row["expected_license_evidence_observed"] for row in results
        )
        payload = {
            "schema": SCHEMA,
            "status": "complete_text_free_page_evidence_probe",
            "provenance_bindings": provenance,
            "method": {
                "pressbooks_scope": "one_page_per_unique_book_url",
                "public_domain_review_essay_scope": "one_page_per_retained_essay",
                "public_domain_review_collection_conjecture_scope": (
                    "official_reusing_material_policy_page"
                ),
                "maximum_response_bytes": MAXIMUM_RESPONSE_BYTES,
                "maximum_concurrent_requests_per_host": 2,
                "user_agent": USER_AGENT,
                "license_patterns_sha256": canonical_sha256(LICENSE_PATTERN_MANIFEST),
            },
            "targets": {
                "count": len(targets),
                "ordered_targets_sha256": canonical_sha256(
                    [row["target_sha256"] for row in targets]
                ),
                "by_source": dict(sorted(source_target_counts.items())),
                "by_scope": dict(sorted(scope_target_counts.items())),
            },
            "population_records": len(records),
            "records_covered_by_targets": sum(row["record_count"] for row in targets),
            "records_with_observed_license_evidence": match_records,
            "results": {
                "path": result_path.name,
                "rows": len(results),
                "bytes": result_path.stat().st_size,
                "sha256": sha256_file(result_path),
                "ordered_results_sha256": canonical_sha256(
                    [row["result_sha256"] for row in results]
                ),
                "http_status_counts": dict(sorted(status_counts.items())),
            },
            "all_targets_observed_expected_license_evidence": all_targets_matched,
            "source_page_text_persisted": False,
            "external_source_page_probe_complete": True,
            "rights_provenance_verified": False,
            "legal_clearance_established": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        if payload["records_covered_by_targets"] != len(records):
            raise ExternalRightsPageProbeError("rights target record coverage differs")
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    args = parser.parse_args()
    result = build_probe(
        args.provenance_root,
        args.output_root,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        maximum_attempts=args.maximum_attempts,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
