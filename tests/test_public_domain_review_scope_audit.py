from __future__ import annotations

from sai.data.public_domain_review_scope_audit import reconstruct_page


def test_reconstructs_collection_and_excludes_embedded_quote() -> None:
    body = b"""
    <html><body>
      <div class="collection-header"><h1>Light</h1></div>
      <div class="attribution">Text by Ada</div>
      <p class="date">January 1, 1900</p>
      <p class="intro">An introduction.</p>
      <div class="essay__text-block"><p>Original analysis.</p></div>
      <div class="essay__text-block"><blockquote>Quoted material.</blockquote></div>
    </body></html>
    """
    result = reconstruct_page(body, "collection")
    assert result["frozen_geometry_text"] == (
        "Light\nText by Ada\nJanuary 1, 1900\n\nAn introduction.\n\n"
        "Original analysis.\nQuoted material."
    )
    assert result["scoped_text"] == (
        "Light\nText by Ada\nJanuary 1, 1900\n\nAn introduction.\n\n"
        "Original analysis.\n"
    )
    assert result["excluded_quote_elements"] == 1
    assert result["excluded_quote_codepoints"] == len("Quoted material.")
    assert result["page_specific_cc_by_sa_observed"] is False


def test_reconstructs_essay_and_requires_exact_license_link() -> None:
    body = b"""
    <html><body>
      <div class="essay-view">
        <span class="title">Bridges</span><span class="subtitle">Across fields</span>
        <p class="byline">By Grace</p><p class="date">January 2, 1900</p>
        <p class="intro">An introduction.</p>
        <div class="essay__text-block"><p>Original synthesis.</p></div>
      </div>
      <div class="essay-license essay__content">
        The text of this essay is published under a
        <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA</a>
        license.
      </div>
    </body></html>
    """
    result = reconstruct_page(body, "essay")
    assert result["frozen_geometry_text"] == (
        "Bridges\nAcross fields\nBy Grace\nJanuary 2, 1900\n\n"
        "An introduction.\n\nOriginal synthesis."
    )
    assert result["scoped_text"] == result["frozen_geometry_text"]
    assert result["page_specific_cc_by_sa_observed"] is True


def test_nearby_license_text_does_not_substitute_for_exact_link() -> None:
    body = b"""
    <html><body>
      <div class="essay-view">
        <span class="title">Title</span><span class="subtitle"></span>
        <p class="byline">By Grace</p><p class="date">January 2, 1900</p>
        <p class="intro">Intro.</p>
        <div class="essay__text-block"><p>Text.</p></div>
      </div>
      <div class="essay-license essay__content">
        CC BY-SA <a href="https://example.com/">not the license</a>
      </div>
    </body></html>
    """
    result = reconstruct_page(body, "essay")
    assert result["page_specific_cc_by_sa_observed"] is False
