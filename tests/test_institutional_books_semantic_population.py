from __future__ import annotations

import copy

import pytest

from sai.data.institutional_books_semantic_population import (
    InstitutionalBooksSemanticPopulationError,
    _label,
    select_diverse_barcodes,
)


def _row(
    barcode: str, topic: str, genre: str, tokens: int
) -> dict[str, object]:
    return {
        "barcode_src": barcode,
        "topic_or_subject_gen": topic,
        "genre_or_form_src": genre,
        "token_count_o200k_base_gen": tokens,
    }


def test_diverse_selection_is_deterministic_and_round_robins_strata() -> None:
    rows = [
        _row("physics-a", "Physics -- mechanics", "textbook", 50_000),
        _row("physics-b", "Physics -- mechanics", "textbook", 55_000),
        _row("physics-c", "Physics -- mechanics", "textbook", 60_000),
        _row("poetry-a", "Persian poetry", "poetry", 10_000),
        _row("history-a", "Chinese history", "monograph", 100_000),
    ]
    selected, stats = select_diverse_barcodes(rows, 3, "seed")
    replay, replay_stats = select_diverse_barcodes(
        list(reversed(copy.deepcopy(rows))), 3, "seed"
    )
    assert selected == replay
    assert stats == replay_stats
    assert len(selected) == 3
    assert stats["eligible_strata"] == 3
    assert stats["selected_strata"] == 3


def test_diverse_selection_revisits_strata_until_cap() -> None:
    rows = [
        _row("a1", "math", "textbook", 10_000),
        _row("a2", "math", "textbook", 11_000),
        _row("a3", "math", "textbook", 12_000),
        _row("b1", "literature", "novel", 100_000),
    ]
    selected, stats = select_diverse_barcodes(rows, 4, "seed")
    assert set(selected) == {"a1", "a2", "a3", "b1"}
    assert stats["selected_rows"] == 4
    assert stats["selected_strata"] == 2


def test_selection_rejects_duplicate_barcodes_and_bad_geometry() -> None:
    duplicate = [
        _row("same", "math", "textbook", 10_000),
        _row("same", "history", "essay", 20_000),
    ]
    with pytest.raises(
        InstitutionalBooksSemanticPopulationError,
        match="selection candidate differs",
    ):
        select_diverse_barcodes(duplicate, 2, "seed")
    with pytest.raises(
        InstitutionalBooksSemanticPopulationError,
        match="selection geometry differs",
    ):
        select_diverse_barcodes([_row("a", "math", "book", 10_000)], 0, "seed")


def test_stratum_labels_are_language_agnostic_but_stable() -> None:
    assert _label("  Physics -- Mechanics / Dynamics ") == "physics"
    assert _label("日本文学") == "unknown"
    assert _label(None) == "unknown"
