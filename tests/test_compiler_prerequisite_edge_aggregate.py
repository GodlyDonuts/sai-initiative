from __future__ import annotations

from copy import deepcopy

from sai.data.compiler_prerequisite_edge_aggregate import (
    HELPFUL_SCHEMA,
    NONEDGE_SCHEMA,
    STRICT_SCHEMA,
    UNSUPPORTED_SCHEMA,
    route_candidate,
)
from tests.test_compiler_prerequisite_edge_labeling import _candidate, _strict


def _receipt(verdict: str) -> dict:
    judgment = deepcopy(_strict(_candidate()))
    judgment["verdict"] = verdict
    judgment["judgment_sha256"] = "8" * 64
    if verdict == "co_taught_not_prerequisite":
        judgment["direction_supported"] = False
        judgment["defects"] = ["cooccurrence_only"]
    elif verdict == "unsupported":
        judgment["direction_supported"] = False
        judgment["defects"] = ["insufficient_directional_evidence"]
    return {"receipt_sha256": "7" * 64, "judgment": judgment}


def test_strict_edge_route_is_source_text_free_and_nontraining() -> None:
    candidate = _candidate()
    route, row = route_candidate(candidate, _receipt("strict_prerequisite"))
    assert route == "strict"
    assert row["schema"] == STRICT_SCHEMA
    for anchor in candidate["supporting_anchors"]:
        assert anchor["text"] not in str(row)
    assert row["source_text_persisted"] is False
    assert row["directional_prerequisite_verified"] is False
    assert row["training_ready"] is False


def test_helpful_foundation_route_remains_a_graph_candidate() -> None:
    route, row = route_candidate(_candidate(), _receipt("helpful_foundation"))
    assert route == "helpful"
    assert row["schema"] == HELPFUL_SCHEMA
    assert row["acyclic_graph_construction_complete"] is False


def test_co_taught_and_unsupported_routes_are_explicit_nonedges() -> None:
    route, row = route_candidate(_candidate(), _receipt("co_taught_not_prerequisite"))
    assert route == "co_taught"
    assert row["schema"] == NONEDGE_SCHEMA
    assert row["defects"] == ["cooccurrence_only"]

    route, row = route_candidate(_candidate(), _receipt("unsupported"))
    assert route == "unsupported"
    assert row["schema"] == UNSUPPORTED_SCHEMA
    assert row["defects"] == ["insufficient_directional_evidence"]
