from __future__ import annotations

import hashlib

from sai.data.public_domain_review_scope_audit import reconstruct_page
from sai.data.public_domain_review_scoped_candidates import (
    _ineligible_result,
    materialize_response,
)
from sai.data.token_stream import canonical_sha256


def _page(text: str = "Original synthesis.") -> bytes:
    return f"""
    <html><body>
      <div class="essay-view">
        <span class="title">Bridges</span><span class="subtitle">Across fields</span>
        <p class="byline">By Grace</p><p class="date">January 2, 1900</p>
        <p class="intro">An introduction.</p>
        <div class="essay__text-block"><p>{text}</p></div>
        <div class="essay__text-block"><blockquote>Quoted material.</blockquote></div>
      </div>
      <div class="essay-license essay__content essay__text-block">
        The text is published under a
        <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA</a>
        license.
      </div>
    </body></html>
    """.encode()


def _inputs() -> tuple[dict, dict, dict, dict]:
    body = _page()
    replay = reconstruct_page(body, "essay")
    identity = "1" * 64
    candidate = {
        "identity_sha256": identity,
        "text": replay["frozen_geometry_text"],
        "source": {
            "dataset": "common-pile/public_domain_review_filtered",
            "domain": "english",
            "license": "CC-BY-SA-4.0",
            "row_id": "row-1",
        },
    }
    provenance = {
        "record_sha256": "2" * 64,
        "source_metadata": {
            "metadata.url": "https://publicdomainreview.org/essay/bridges/",
            "type": "essay",
        },
    }
    rights = {
        "identity_sha256": identity,
        "source_id": "common_pile_public_domain_review",
        "record_sha256": "3" * 64,
    }
    scope = {
        "candidate_identity_sha256": identity,
        "source_provenance_record_sha256": provenance["record_sha256"],
        "rights_record_sha256": rights["record_sha256"],
        "source_type": "essay",
        "source_url": provenance["source_metadata"]["metadata.url"],
        "status": "scope_reconstruction_complete",
        "rights_scope_reconstruction_complete": True,
        "scoped_text_sha256": hashlib.sha256(
            replay["scoped_text"].encode()
        ).hexdigest(),
        "scoped_text_bytes": len(replay["scoped_text"].encode()),
        "excluded_quote_elements": replay["excluded_quote_elements"],
        "excluded_quote_codepoints": replay["excluded_quote_codepoints"],
        "result_sha256": "4" * 64,
    }
    return candidate, provenance, rights, scope


def _fetched(body: bytes) -> dict:
    return {
        "attempts": [{"attempt": 1, "outcome": "response", "status": 200}],
        "http_status": 200,
        "final_url": "https://publicdomainreview.org/essay/bridges/",
        "error_type": None,
        "response_bytes": body,
    }


def test_exact_page_materializes_quote_excluded_candidate() -> None:
    candidate, provenance, rights, scope = _inputs()
    result, materialized = materialize_response(
        candidate, provenance, rights, scope, _fetched(_page())
    )
    assert result["status"] == "materialized_exact_scoped_candidate"
    assert materialized is not None
    assert "Quoted material." not in materialized["text"]
    assert materialized["excluded_quote_elements"] == 1
    assert materialized["attribution_required"] is True
    assert materialized["share_alike_required"] is True
    assert materialized["content_quality_verified"] is False
    assert materialized["training_ready"] is False
    assert materialized["record_sha256"] == canonical_sha256(
        {key: value for key, value in materialized.items() if key != "record_sha256"}
    )
    assert result["scoped_candidate_record_sha256"] == materialized["record_sha256"]


def test_changed_page_routes_to_drift_without_persisting_text() -> None:
    candidate, provenance, rights, scope = _inputs()
    result, materialized = materialize_response(
        candidate, provenance, rights, scope, _fetched(_page("Changed synthesis."))
    )
    assert result["status"] == "materialization_source_or_scope_drift_review"
    assert result["source_text_persisted_in_result"] is False
    assert result["scoped_candidate_record_sha256"] is None
    assert materialized is None


def test_ineligible_audit_row_is_text_free_and_signed() -> None:
    _candidate, provenance, rights, scope = _inputs()
    scope["status"] = "source_page_drift_review"
    scope["source_provenance_record_sha256"] = provenance["record_sha256"]
    scope["rights_record_sha256"] = rights["record_sha256"]
    result = _ineligible_result(scope)
    assert result["status"] == "scope_audit_not_eligible"
    assert result["scope_audit_status"] == "source_page_drift_review"
    assert result["source_text_persisted_in_result"] is False
    assert result["training_ready"] is False
    assert result["result_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )
