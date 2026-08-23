"""Replay Public Domain Review pages and measure a quote-excluded rights scope."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import textwrap
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from bs4 import BeautifulSoup

from sai.data.agent_labeling import _atomic_create
from sai.data.bounded_pilot_compiler_aggregate import load_rights_queue
from sai.data.common_pile_external_provenance import (
    RECORD_SCHEMA as PROVENANCE_RECORD_SCHEMA,
)
from sai.data.common_pile_external_provenance import (
    SCHEMA as PROVENANCE_SCHEMA,
)
from sai.data.common_pile_streaming_pilot import SCHEMA as PILOT_SCHEMA
from sai.data.data_yield_ledger import _bound_file, _load_receipt
from sai.data.external_rights_adjudication_queue import (
    ROUTES as RIGHTS_ROUTES,
)
from sai.data.token_stream import canonical_sha256, normalize_document, sha256_file

SCHEMA = "sai-public-domain-review-scope-audit-v1"
RESULT_SCHEMA = "sai-public-domain-review-scope-result-v1"
SOURCE_ID = "common_pile_public_domain_review"
CC_BY_SA_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
USER_AGENT = "SaiDataRightsAudit/1.0 (+https://github.com/GodlyDonuts/sai-initiative)"
MAXIMUM_RESPONSE_BYTES = 2 << 20
UPSTREAM_COLLECTOR = {
    "repository": "https://github.com/r-three/common-pile",
    "commit": "9457f04a14cb2355ab00023420369d46ffd4a395",
    "scrape_py_sha256": (
        "2755775b281abafa102370e68a377eeacd66b31db774b01e812737d9c6eb591e"
    ),
    "utils_py_sha256": (
        "4a781b4293bcbf7bc24f7618e130dd944d4944654778525ab799c6060b73d305"
    ),
}


class PublicDomainReviewScopeAuditError(RuntimeError):
    """The frozen source, page replay, license scope, or result differs."""


def _elements_text(document: Any, element: str, class_name: str | None) -> list[str]:
    values = [
        item.get_text().strip()
        for item in document.find_all(element, class_=class_name)
    ]
    return values or [""]


def _parse_source_text(document: Any, source_type: str) -> str:
    """Reproduce the pinned upstream collector's exact text geometry."""

    if source_type == "collection":
        headers = document.find_all("div", class_="collection-header")
        if not headers:
            raise PublicDomainReviewScopeAuditError("collection header differs")
        title = _elements_text(headers[0], "h1", None)[0]
        byline = _elements_text(document, "div", "attribution")[0]
        intro = _elements_text(document, "p", "intro")[0]
        date = _elements_text(document, "p", "date")[0]
        text_blocks = "\n".join(_elements_text(document, "div", "essay__text-block"))
        return (
            textwrap.dedent("""
                {title}
                {byline}
                {date}

                {intro}

                {text_blocks}
                """)
            .strip()
            .format(
                title=title,
                byline=byline,
                date=date,
                intro=intro,
                text_blocks=text_blocks,
            )
        )
    if source_type not in {"essay", "conjecture"}:
        raise PublicDomainReviewScopeAuditError("source type differs")
    essays = document.find_all("div", class_="essay-view")
    if not essays:
        raise PublicDomainReviewScopeAuditError("essay view differs")
    essay = essays[0]
    title = _elements_text(essay, "span", "title")[0]
    subtitle = _elements_text(essay, "span", "subtitle")[0]
    byline = _elements_text(essay, "p", "byline")[0]
    intro = _elements_text(essay, "p", "intro")[0]
    date = _elements_text(essay, "p", "date")[0]
    text_blocks = "\n".join(_elements_text(essay, "div", "essay__text-block"))
    return (
        textwrap.dedent("""
            {title}
            {subtitle}
            {byline}
            {date}

            {intro}

            {text_blocks}
            """)
        .strip()
        .format(
            title=title,
            subtitle=subtitle,
            byline=byline,
            date=date,
            intro=intro,
            text_blocks=text_blocks,
        )
    )


def reconstruct_page(response_bytes: bytes, source_type: str) -> dict[str, Any]:
    """Rebuild frozen and conservative unquoted views without retaining HTML."""

    if not response_bytes or len(response_bytes) > MAXIMUM_RESPONSE_BYTES:
        raise PublicDomainReviewScopeAuditError("page response size differs")
    document = BeautifulSoup(response_bytes, "html.parser")
    license_blocks = document.select("div.essay-license.essay__content")
    page_license_observed = any(
        "CC BY-SA" in block.get_text(" ", strip=True)
        and any(anchor.get("href") == CC_BY_SA_URL for anchor in block.find_all("a"))
        for block in license_blocks
    )
    for block in license_blocks:
        block.decompose()
    frozen_geometry_text = _parse_source_text(document, source_type)
    scoped_document = BeautifulSoup(response_bytes, "html.parser")
    for block in scoped_document.select("div.essay-license.essay__content"):
        block.decompose()
    selected_quote_nodes = scoped_document.select(
        "div.essay__text-block blockquote, div.essay__text-block q, "
        "p.intro blockquote, p.intro q"
    )
    quote_nodes = [
        node
        for node in selected_quote_nodes
        if not any(parent.name in {"blockquote", "q"} for parent in node.parents)
    ]
    excluded_quote_codepoints = sum(len(node.get_text()) for node in quote_nodes)
    for node in quote_nodes:
        node.decompose()
    scoped_text = _parse_source_text(scoped_document, source_type)
    if not scoped_text:
        raise PublicDomainReviewScopeAuditError("scoped page text is empty")
    return {
        "frozen_geometry_text": frozen_geometry_text,
        "scoped_text": scoped_text,
        "page_specific_cc_by_sa_observed": page_license_observed,
        "excluded_quote_elements": len(quote_nodes),
        "excluded_quote_codepoints": excluded_quote_codepoints,
    }


def _load_inputs(
    pilot_root: Path, provenance_root: Path, rights_root: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    pilot = _load_receipt(pilot_root / "receipt.json")
    near = pilot.get("near_duplicate_filter")
    if (
        pilot.get("schema") != PILOT_SCHEMA
        or pilot.get("status") != "complete_nontraining_pilot"
        or pilot.get("source_id") != SOURCE_ID
        or pilot.get("rights_verification_complete") is not False
        or pilot.get("training_ready") is not False
        or not isinstance(near, dict)
    ):
        raise PublicDomainReviewScopeAuditError("pilot receipt differs")
    candidate_path = _bound_file(
        pilot_root,
        {
            "path": near.get("output_path"),
            "bytes": near.get("output_bytes"),
            "sha256": near.get("output_sha256"),
        },
    )
    near_receipt_path = pilot_root / str(near.get("receipt_path"))
    near_receipt = _load_receipt(near_receipt_path)
    near_output = near_receipt.get("output")
    if (
        near.get("receipt_file_sha256") != sha256_file(near_receipt_path)
        or near.get("receipt_sha256") != near_receipt.get("receipt_sha256")
        or not isinstance(near_output, dict)
    ):
        raise PublicDomainReviewScopeAuditError("near-duplicate receipt differs")
    candidates = []
    with candidate_path.open() as handle:
        for line in handle:
            candidates.append(normalize_document(json.loads(line)))
    identities = [row["identity_sha256"] for row in candidates]
    if (
        len(candidates) != near.get("output_documents")
        or len(identities) != len(set(identities))
        or near_output.get("ordered_identity_sha256")
        != hashlib.sha256(
            b"".join(bytes.fromhex(identity) for identity in identities)
        ).hexdigest()
    ):
        raise PublicDomainReviewScopeAuditError("pilot candidate coverage differs")

    provenance = _load_receipt(provenance_root / "receipt.json")
    output = provenance.get("output")
    if (
        provenance.get("schema") != PROVENANCE_SCHEMA
        or provenance.get("status") != "complete_text_free_source_metadata_replay"
        or provenance.get("source_id") != SOURCE_ID
        or provenance.get("source_text_persisted") is not False
        or provenance.get("training_ready") is not False
        or not isinstance(output, dict)
    ):
        raise PublicDomainReviewScopeAuditError("provenance receipt differs")
    provenance_path = _bound_file(provenance_root, output)
    by_identity = {}
    with provenance_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            unsigned = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != PROVENANCE_RECORD_SCHEMA
                or identity in by_identity
                or row.get("record_sha256") != canonical_sha256(unsigned)
            ):
                raise PublicDomainReviewScopeAuditError("provenance row differs")
            by_identity[identity] = row
    if set(identities) != set(by_identity):
        raise PublicDomainReviewScopeAuditError("provenance coverage differs")

    rights_by_identity, rights = load_rights_queue(rights_root)
    pdr_rights = {
        identity: row
        for identity, row in rights_by_identity.items()
        if row["source_id"] == SOURCE_ID
    }
    if set(identities) != set(pdr_rights):
        raise PublicDomainReviewScopeAuditError("rights coverage differs")
    return (
        candidates,
        by_identity,
        {
            "pilot": pilot,
            "provenance": provenance,
            "rights": rights,
            "rights_by_identity": pdr_rights,
            "candidate_path": candidate_path,
            "provenance_path": provenance_path,
        },
    )


def fetch_and_replay(
    candidate: dict[str, Any],
    provenance: dict[str, Any],
    rights: dict[str, Any],
    *,
    timeout_seconds: float,
    maximum_attempts: int,
) -> dict[str, Any]:
    """Fetch one page and retain only hashes and scope accounting."""

    if not 1 <= timeout_seconds <= 120 or not 1 <= maximum_attempts <= 5:
        raise PublicDomainReviewScopeAuditError("network bounds differ")
    metadata = provenance["source_metadata"]
    source_type = metadata.get("type")
    url = metadata.get("metadata.url")
    if (
        source_type not in {"collection", "conjecture", "essay"}
        or not isinstance(url, str)
        or not url.startswith("https://publicdomainreview.org/")
        or rights.get("expected_license_evidence_observed") is not True
    ):
        raise PublicDomainReviewScopeAuditError("source metadata scope differs")
    expected_route = (
        RIGHTS_ROUTES["pdr_essay_observed"]
        if source_type == "essay"
        else RIGHTS_ROUTES["pdr_policy_observed"]
    )
    if rights.get("adjudication_route") != expected_route:
        raise PublicDomainReviewScopeAuditError("rights route differs")

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

    truncated = len(response_bytes) > MAXIMUM_RESPONSE_BYTES
    response_bytes = response_bytes[:MAXIMUM_RESPONSE_BYTES]
    replay = None
    parse_error = None
    if http_status == 200 and response_bytes and not truncated:
        try:
            replay = reconstruct_page(response_bytes, source_type)
        except (PublicDomainReviewScopeAuditError, IndexError, ValueError) as error:
            parse_error = type(error).__name__
    frozen_match = bool(
        replay is not None and replay["frozen_geometry_text"] == candidate["text"]
    )
    page_license = bool(
        replay is not None and replay["page_specific_cc_by_sa_observed"]
    )
    license_scope_observed = source_type == "collection" or page_license
    final_url_on_source_host = bool(
        final_url is not None
        and urlsplit(final_url).scheme == "https"
        and urlsplit(final_url).netloc.lower() == "publicdomainreview.org"
    )
    status = "scope_reconstruction_complete"
    if (
        http_status != 200
        or not response_bytes
        or truncated
        or not final_url_on_source_host
    ):
        status = "transport_or_response_review"
    elif replay is None:
        status = "page_parse_review"
    elif not frozen_match:
        status = "source_page_drift_review"
    elif not license_scope_observed:
        status = "page_specific_license_review"
    scoped_text = replay["scoped_text"] if replay is not None else None
    result = {
        "schema": RESULT_SCHEMA,
        "candidate_identity_sha256": candidate["identity_sha256"],
        "source_provenance_record_sha256": provenance["record_sha256"],
        "rights_record_sha256": rights["record_sha256"],
        "source_type": source_type,
        "source_url": url,
        "attempts": attempts,
        "http_status": http_status,
        "final_url": final_url,
        "final_url_on_source_host": final_url_on_source_host,
        "error_type": error_type,
        "parse_error_type": parse_error,
        "response_bytes_inspected": len(response_bytes),
        "response_sha256": (
            hashlib.sha256(response_bytes).hexdigest() if response_bytes else None
        ),
        "response_truncated": truncated,
        "page_replays_frozen_candidate": frozen_match,
        "page_specific_cc_by_sa_observed": page_license,
        "policy_scope_evidence_observed": rights["expected_license_evidence_observed"],
        "license_scope_evidence_observed": license_scope_observed,
        "excluded_quote_elements": (
            replay["excluded_quote_elements"] if replay is not None else None
        ),
        "excluded_quote_codepoints": (
            replay["excluded_quote_codepoints"] if replay is not None else None
        ),
        "scoped_text_bytes": (
            len(scoped_text.encode()) if scoped_text is not None else None
        ),
        "scoped_text_sha256": (
            hashlib.sha256(scoped_text.encode()).hexdigest()
            if scoped_text is not None
            else None
        ),
        "scope_reconstruction_changed_text": (
            scoped_text != candidate["text"] if scoped_text is not None else None
        ),
        "status": status,
        "source_page_text_persisted": False,
        "scoped_text_persisted": False,
        "rights_scope_reconstruction_complete": (
            status == "scope_reconstruction_complete"
        ),
        "legal_clearance_established": False,
        "training_ready": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def build_audit(
    pilot_root: Path,
    provenance_root: Path,
    rights_root: Path,
    output_root: Path,
    *,
    concurrency: int,
    timeout_seconds: float,
    maximum_attempts: int,
    fetch_function: Callable[..., dict[str, Any]] = fetch_and_replay,
) -> dict[str, Any]:
    """Measure exact source drift and quote-excluded scope for every pilot row."""

    if output_root.exists() or output_root.is_symlink() or not 1 <= concurrency <= 2:
        raise PublicDomainReviewScopeAuditError("audit output or concurrency differs")
    candidates, provenance, inputs = _load_inputs(
        pilot_root, provenance_root, rights_root
    )
    rights_by_identity = inputs["rights_by_identity"]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                fetch_function,
                candidate,
                provenance[candidate["identity_sha256"]],
                rights_by_identity[candidate["identity_sha256"]],
                timeout_seconds=timeout_seconds,
                maximum_attempts=maximum_attempts,
            ): candidate["identity_sha256"]
            for candidate in candidates
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            result = future.result()
            unsigned = {
                key: value for key, value in result.items() if key != "result_sha256"
            }
            if (
                result.get("schema") != RESULT_SCHEMA
                or result.get("candidate_identity_sha256") != futures[future]
                or result.get("result_sha256") != canonical_sha256(unsigned)
                or result.get("source_page_text_persisted") is not False
                or result.get("scoped_text_persisted") is not False
                or result.get("legal_clearance_established") is not False
                or result.get("training_ready") is not False
            ):
                raise PublicDomainReviewScopeAuditError("scope result differs")
            results.append(result)
            if completed % 100 == 0 or completed == len(futures):
                print(
                    json.dumps(
                        {
                            "event": "pdr_scope_audit_progress",
                            "complete": completed,
                            "remaining": len(futures) - completed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    results.sort(key=lambda row: row["candidate_identity_sha256"])
    if len(results) != len(candidates) or len(
        {row["candidate_identity_sha256"] for row in results}
    ) != len(candidates):
        raise PublicDomainReviewScopeAuditError("scope result coverage differs")
    status_counts = Counter(row["status"] for row in results)
    type_counts = Counter(row["source_type"] for row in results)

    output_root.mkdir(parents=True)
    try:
        results_path = output_root / "scope_results.jsonl"
        with results_path.open("x") as handle:
            for row in results:
                handle.write(
                    json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                )
        payload = {
            "schema": SCHEMA,
            "status": "complete_text_free_scope_audit",
            "pilot": {
                "root_name": pilot_root.name,
                "receipt_file_sha256": sha256_file(pilot_root / "receipt.json"),
                "receipt_sha256": inputs["pilot"]["receipt_sha256"],
                "candidate_file_sha256": sha256_file(inputs["candidate_path"]),
            },
            "provenance": {
                "root_name": provenance_root.name,
                "receipt_file_sha256": sha256_file(provenance_root / "receipt.json"),
                "receipt_sha256": inputs["provenance"]["receipt_sha256"],
                "manifest_sha256": sha256_file(inputs["provenance_path"]),
            },
            "rights_receipt_sha256": inputs["rights"]["receipt_sha256"],
            "upstream_collector": UPSTREAM_COLLECTOR,
            "method": {
                "maximum_response_bytes": MAXIMUM_RESPONSE_BYTES,
                "maximum_concurrent_requests_per_host": 2,
                "user_agent": USER_AGENT,
                "quote_elements_excluded": ["blockquote", "q"],
                "collection_license_scope": "official_policy_plus_exact_page_replay",
                "essay_and_conjecture_license_scope": (
                    "page_specific_cc_by_sa_plus_exact_page_replay"
                ),
                "source_page_bodies_persisted": False,
                "scoped_text_persisted": False,
            },
            "results": {
                "path": results_path.name,
                "rows": len(results),
                "bytes": results_path.stat().st_size,
                "sha256": sha256_file(results_path),
                "ordered_results_sha256": canonical_sha256(
                    [row["result_sha256"] for row in results]
                ),
            },
            "records_by_source_type": dict(sorted(type_counts.items())),
            "records_by_status": dict(sorted(status_counts.items())),
            "records_with_exact_live_page_replay": sum(
                row["page_replays_frozen_candidate"] for row in results
            ),
            "records_with_complete_scope_reconstruction": status_counts[
                "scope_reconstruction_complete"
            ],
            "records_with_quote_exclusions": sum(
                (row["excluded_quote_elements"] or 0) > 0 for row in results
            ),
            "excluded_quote_elements": sum(
                row["excluded_quote_elements"] or 0 for row in results
            ),
            "excluded_quote_codepoints": sum(
                row["excluded_quote_codepoints"] or 0 for row in results
            ),
            "source_page_text_persisted": False,
            "scoped_text_persisted": False,
            "automated_legal_decision_made": False,
            "rights_scope_evidence_measured": True,
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    args = parser.parse_args()
    result = build_audit(
        args.pilot_root,
        args.provenance_root,
        args.rights_root,
        args.output_root,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        maximum_attempts=args.maximum_attempts,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
