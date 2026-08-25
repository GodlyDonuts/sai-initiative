from __future__ import annotations

import hashlib

import httpx

from sai.data.pleias_practical_locator_scan import (
    _download_pinned_parent,
    _hash_selected,
    _ordered_parents,
    _route,
)


def _row(text: str, **updates):
    row = {
        "identifier": "doc-1",
        "collection": "open_books",
        "open_type": "open",
        "license": "public domain",
        "language": "English",
        "word_count": 100,
        "token_count": 150,
        "text": text,
    }
    row.update(updates)
    return row


def test_practical_route_accepts_plain_english_public_domain_text():
    text = "A coherent paragraph explains a useful historical subject in context. " * 12
    route, content, digest = _route(_row(text))
    assert route == "pass_mechanical_gate"
    assert content == text.encode()
    assert len(digest) == 64


def test_practical_route_rejects_nonenglish_and_unclear_rights():
    text = "A complete useful paragraph with enough context and length. " * 12
    assert _route(_row(text, language="French"))[0] == "hold_nonenglish"
    assert _route(_row(text, license="unknown"))[0] == "hold_rights"


def test_practical_route_rejects_contextless_answer_key():
    text = "\n".join(f"{number}. A" for number in range(1, 90))
    route, content, digest = _route(_row(text, word_count=178))
    assert route != "pass_mechanical_gate"
    assert content is None
    assert digest is None


def test_hash_selection_is_deterministic_and_full_rate_accepts():
    identity = "0" * 64
    assert _hash_selected(identity, 1_000_000)
    assert _hash_selected(identity, 1) == _hash_selected(identity, 1)


def test_early_stop_parent_order_is_deterministic_and_not_path_ordered():
    parents = [
        {
            "source_path": f"common_corpus_{index}/part.parquet",
            "sha256": f"{index:x}" * 64,
        }
        for index in range(1, 9)
    ]
    ordered, method = _ordered_parents(parents, True)
    repeated, repeated_method = _ordered_parents(list(reversed(parents)), True)
    assert method == repeated_method == "canonical_source_identity_sha256"
    assert ordered == repeated
    assert ordered != parents


def test_bounded_parent_download_retries_and_closes_response(tmp_path, monkeypatch):
    content = b"exact pinned parquet payload"
    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ReadTimeout("transient closed response", request=request)
        assert request.headers["Authorization"] == "Bearer token"
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(200, content=content, request=request)

    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def client(**kwargs):
        return real_client(
            transport=transport,
            follow_redirects=kwargs["follow_redirects"],
            timeout=kwargs["timeout"],
        )

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(
        "sai.data.pleias_practical_locator_scan.time.sleep", lambda _: None
    )
    row = {
        "source_repository": "example/source",
        "source_path": "data/parent.parquet",
        "source_revision": "1" * 40,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    path = _download_pinned_parent(row, "token", tmp_path, attempts=2)
    assert path.read_bytes() == content
    assert len(requests) == 2
