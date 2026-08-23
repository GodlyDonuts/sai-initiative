from __future__ import annotations

from sai.data.common_pile_confirmation_population import (
    build_confirmation_parent_plan,
)


def _row(source_id: str, path: str, size: int) -> dict:
    return {
        "source_id": source_id,
        "epistemic_function": "reference",
        "repository": f"common-pile/{source_id}",
        "revision": "a" * 40,
        "license": "CC0-1.0",
        "access": "public",
        "path": path,
        "physical_bytes": size,
        "sha256": ("1" if path == "old.json.gz" else "2") * 64,
        "text_column": "text",
    }


def test_confirmation_prefers_new_parent_and_reuses_only_when_necessary() -> None:
    arxiv = "common_pile_arxiv_abstracts"
    libretexts = "common_pile_libretexts"
    rows = [
        _row(arxiv, "old.json.gz", 10),
        _row(arxiv, "new.json.gz", 20),
        _row(libretexts, "old.json.gz", 30),
    ]
    plan = build_confirmation_parent_plan(
        rows,
        [arxiv, libretexts],
        {arxiv: {"old.json.gz"}, libretexts: {"old.json.gz"}},
        plan_sha256="f" * 64,
    )
    by_source = {row["source_id"]: row for row in plan}
    assert by_source[arxiv]["path"] == "new.json.gz"
    assert by_source[arxiv]["parent_disjoint_from_discovery"] is True
    assert by_source[libretexts]["path"] == "old.json.gz"
    assert by_source[libretexts]["parent_disjoint_from_discovery"] is False
