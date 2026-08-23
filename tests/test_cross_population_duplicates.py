import pytest

from sai.data.cross_population_duplicates import (
    CrossPopulationDuplicateError,
    find_exact_pairs,
)
from sai.data.token_stream import canonical_sha256


def _candidate(identity: str, text: str) -> dict:
    import hashlib

    return {
        "candidate_identity_sha256": identity,
        "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
    }


def test_exact_pairs_cross_populations_without_quadratic_similarity() -> None:
    text = "Substantive source material " * 20
    populations = [
        (
            "left",
            [_candidate("1" * 64, text), _candidate("2" * 64, "unique " * 40)],
            [{"source_id": "source_a"}, {"source_id": "source_a"}],
        ),
        (
            "right",
            [
                _candidate("3" * 64, text),
                _candidate("4" * 64, text.upper()),
            ],
            [{"source_id": "source_b"}, {"source_id": "source_c"}],
        ),
    ]
    pairs = find_exact_pairs(populations)
    assert len(pairs) == 3
    assert sum("byte_exact" in pair["reasons"] for pair in pairs) == 1
    assert sum("normalized_token_exact" in pair["reasons"] for pair in pairs) == 3
    assert sum(pair["cross_population"] for pair in pairs) == 2
    assert all(pair["cross_source"] is True for pair in pairs)
    for pair in pairs:
        unsigned = {key: value for key, value in pair.items() if key != "pair_sha256"}
        assert pair["pair_sha256"] == canonical_sha256(unsigned)


def test_duplicate_candidate_identity_is_a_custody_error() -> None:
    identity = "1" * 64
    candidate = _candidate(identity, "source text " * 40)
    populations = [
        ("left", [candidate], [{"source_id": "a"}]),
        ("right", [candidate], [{"source_id": "b"}]),
    ]
    with pytest.raises(CrossPopulationDuplicateError, match="custody"):
        find_exact_pairs(populations)
