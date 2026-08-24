import json
from pathlib import Path

from sai.data.grounded_bridge_curriculum_candidates import (
    DOCUMENT_TYPES,
    RECEIPT_SCHEMA,
    ROW_SCHEMA,
    SPLIT_POLICY_SHA256,
    _split,
    build_candidates,
    compile_bridge,
)
from sai.data.grounded_bridge_decontamination import (
    CLEAN_SCHEMA,
)
from sai.data.grounded_bridge_decontamination import (
    SCHEMA as DECONTAMINATION_SCHEMA,
)
from sai.data.grounded_bridge_foundation_query import build_query
from sai.data.grounded_bridge_population import ROW_SCHEMA as ANCHOR_ROW_SCHEMA
from sai.data.grounded_bridge_population import SCHEMA as ANCHOR_POPULATION_SCHEMA
from sai.data.token_stream import canonical_sha256, sha256_file


def _clean_bridge(pair: str = "0" * 64) -> dict:
    row = {
        "schema": CLEAN_SCHEMA,
        "pair_identity_sha256": pair,
        "bridge_label": "mathematics::music",
        "anchor_a_candidate_identity_sha256": "a" * 64,
        "anchor_a_source_content_sha256": "b" * 64,
        "anchor_b_candidate_identity_sha256": "c" * 64,
        "anchor_b_source_content_sha256": "d" * 64,
        "bridge_thesis": "Both structures use ratios to organize relationships.",
        "shared_structure": "Intervals and fractions compare quantities relationally.",
        "prerequisite_map": ["Understand ratios", "Recognize musical intervals"],
        "analogy_failure_modes": [
            "Musical consonance is perceptual; arithmetic equality is exact."
        ],
        "representations": [
            {
                "title": "Ratios across two domains",
                "type": "conceptual_explanation",
                "text": (
                    "A ratio compares magnitudes, while an interval compares "
                    "frequencies."
                ),
            },
            {
                "title": "A bounded transfer",
                "type": "worked_transfer_problem",
                "text": (
                    "Use the shared ratio structure, then check where perception "
                    "intervenes."
                ),
            },
        ],
        "verification_questions": [
            {
                "anchor_side": "both",
                "question": "What structure is shared?",
                "expected_answer": "A comparison expressed as a ratio.",
            }
        ],
        "verification_confidence_ppm": 950_000,
        "generator_receipt_sha256": "1" * 64,
        "verification_receipt_sha256": "2" * 64,
        "same_family_route": "retain",
        "independent_family_route": "retain",
        "same_family_retention_passed": True,
        "independent_family_retention_passed": True,
        "source_disjoint": True,
        "independent_model_family_verification_complete": True,
        "benchmark_decontamination_complete": True,
        "global_deduplication_complete": False,
        "transfer_ablation_complete": False,
        "bridge_verified": False,
        "training_ready": False,
    }
    row["record_sha256"] = canonical_sha256(row)
    return row


def _anchors() -> list[dict]:
    return [
        {
            "candidate_identity_sha256": "a" * 64,
            "source_content_sha256": "b" * 64,
            "source": {
                "dataset": "source/a",
                "revision": "revision-a",
                "row_id": "row-a",
            },
        },
        {
            "candidate_identity_sha256": "c" * 64,
            "source_content_sha256": "d" * 64,
            "source": {
                "dataset": "source/b",
                "revision": "revision-b",
                "row_id": "row-b",
            },
        },
    ]


def test_bridge_compiler_preserves_open_gates_and_pair_disjoint_split() -> None:
    rows = compile_bridge(_clean_bridge(), "3" * 64, _anchors(), "4" * 64)
    assert {row["document_type"] for row in rows} == set(DOCUMENT_TYPES)
    assert len(rows) == 5
    assert all(row["schema"] == ROW_SCHEMA for row in rows)
    assert all(row["corpus_split"] == "development" for row in rows)
    assert len({row["source_group_sha256"] for row in rows}) == 1
    assert len({row["content_sha256"] for row in rows}) == len(rows)
    assert len({row["normalized_content_sha256"] for row in rows}) == len(rows)
    assert all(
        row["anchor_candidate_identity_sha256s"] == ["a" * 64, "c" * 64] for row in rows
    )
    assert all(
        row["anchor_source_content_sha256s"] == ["b" * 64, "d" * 64] for row in rows
    )
    assert all(
        row["anchor_sources"] == [value["source"] for value in _anchors()]
        for row in rows
    )
    assert all(row["bridge_verified"] is False for row in rows)
    assert all(
        row["source_disjoint_against_foundation_complete"] is False for row in rows
    )
    assert all(row["training_ready"] is False for row in rows)
    assert _split("0" * 64) == (0, "development")
    assert _split("f" * 64)[1] == "train"


def test_candidate_builder_binds_clean_population_and_durable_receipt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "decontamination"
    source.mkdir()
    clean_path = source / "benchmark_disjoint_bridges.jsonl"
    clean = _clean_bridge()
    clean_path.write_text(json.dumps(clean, sort_keys=True) + "\n")
    source_receipt = {
        "schema": DECONTAMINATION_SCHEMA,
        "status": "complete_post_generation_bridge_benchmark_screen",
        "post_generation_benchmark_screen_complete": True,
        "independent_model_family_verification_complete": True,
        "global_deduplication_complete": False,
        "transfer_ablation_complete": False,
        "training_ready": False,
        "benchmark_disjoint_bridges": {
            "path": clean_path.name,
            "rows": 1,
            "bytes": clean_path.stat().st_size,
            "sha256": sha256_file(clean_path),
            "ordered_records_sha256": canonical_sha256([clean["record_sha256"]]),
        },
    }
    source_receipt["receipt_sha256"] = canonical_sha256(source_receipt)
    (source / "receipt.json").write_text(
        json.dumps(source_receipt, sort_keys=True) + "\n"
    )
    anchor_root = tmp_path / "anchors"
    anchor_root.mkdir()
    anchor_path = anchor_root / "candidates.jsonl"
    anchor_row = {
        "schema": ANCHOR_ROW_SCHEMA,
        "pair_identity_sha256": "0" * 64,
        "candidate_identity_sha256": "0" * 64,
        "anchor_a": _anchors()[0],
        "anchor_b": _anchors()[1],
        "source_disjoint": True,
        "training_ready": False,
    }
    anchor_path.write_text(json.dumps(anchor_row, sort_keys=True) + "\n")
    anchor_receipt = {
        "schema": ANCHOR_POPULATION_SCHEMA,
        "population": {
            "path": str(anchor_path),
            "rows": 1,
            "bytes": anchor_path.stat().st_size,
            "sha256": sha256_file(anchor_path),
        },
        "selection": {"selected_pairs": 1},
        "source_disjoint_pairs": True,
        "training_ready": False,
    }
    anchor_receipt["receipt_sha256"] = canonical_sha256(anchor_receipt)
    anchor_receipt_path = anchor_root / "receipt.json"
    anchor_receipt_path.write_text(json.dumps(anchor_receipt, sort_keys=True) + "\n")
    output = tmp_path / "candidates"
    durable = tmp_path / "evidence" / "receipt.json"
    result = build_candidates(source, anchor_path, anchor_receipt_path, output, durable)
    assert result["schema"] == RECEIPT_SCHEMA
    assert result["split_policy_sha256"] == SPLIT_POLICY_SHA256
    assert result["clean_bridges"] == 1
    assert result["counts"]["documents"] == 5
    assert result["global_deduplication_against_foundation_complete"] is False
    assert result["source_disjoint_split_complete"] is False
    assert result["transfer_ablation_complete"] is False
    assert result["training_ready"] is False
    assert json.loads(durable.read_text()) == result
    assert "Bridge thesis" in (output / "curriculum_candidates.jsonl").read_text()
    query = tmp_path / "query"
    query_result = build_query(output, query)
    assert query_result["counts"]["documents"] == 5
    assert query_result["counts"]["anchors"] == 2


def test_candidate_finalizer_waits_for_complete_decontamination() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        root / "scripts" / "finalize_grounded_bridge_curriculum_local.sh"
    ).read_text()
    assert "while [[ ! -f" in script
    assert "grounded_bridge_curriculum_candidates" in script
    assert "--durable-receipt" in script
    assert "rm " not in script
