from __future__ import annotations

from sai.data.nous_label_worker import _retryable_http_status


def test_cloudflare_524_is_retryable_for_both_endpoints() -> None:
    assert _retryable_http_status(524, "https://inference-api.nousresearch.com/v1")
    assert _retryable_http_status(524, "http://127.0.0.1:8645/v1")


def test_credential_refresh_401_is_retryable_only_through_loopback_proxy() -> None:
    assert _retryable_http_status(401, "http://127.0.0.1:8645/v1")
    assert not _retryable_http_status(401, "https://inference-api.nousresearch.com/v1")


def test_nontransient_client_error_remains_terminal() -> None:
    assert not _retryable_http_status(403, "http://127.0.0.1:8645/v1")
