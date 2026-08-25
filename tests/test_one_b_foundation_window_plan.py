from __future__ import annotations

from sai.data.one_b_foundation_window_plan import WINDOW_SEQUENCES, WINDOW_WEIGHTS


def test_foundation_window_is_exact_102_4b_tokens() -> None:
    assert WINDOW_SEQUENCES == 25_000_000
    assert WINDOW_SEQUENCES * 4_096 == 102_400_000_000
    assert WINDOW_WEIGHTS == (65, 25, 8, 2)
