from __future__ import annotations

from sai.training.one_b_production_contract import OLMO_COMMIT, OLMO_CORE_COMMIT


def test_runtime_pins_match_the_compatible_legacy_olmo_pair() -> None:
    assert OLMO_COMMIT == "090253dac6688f2532509daa7aa2eb5fae50e956"
    assert OLMO_CORE_COMMIT == "7899e7cefaae44e30766ee654bd177f1e1474bc7"
