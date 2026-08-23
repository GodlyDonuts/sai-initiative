import json
from pathlib import Path

from sai.data.token_stream import canonical_sha256, sha256_file

ROOT = Path(__file__).parents[1]
PUBLICATION = (
    ROOT
    / "artifacts"
    / "sai_source_mechanical_quality_gate_publication_20260826_r1.json"
)
RECEIPTS = ROOT / "artifacts" / "sai_source_mechanical_quality_gate_20260826_r3"


def test_source_safe_quality_gate_publication_is_exact() -> None:
    payload = json.loads(PUBLICATION.read_text())
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    assert payload["receipt_sha256"] == canonical_sha256(unsigned)
    assert payload["receipt_sha256"] == (
        "c25127e13c579bb066b887d264da1905bd78f2f3d24c183bba547ea019a2bf66"
    )
    assert payload["policy_sha256"] == (
        "f85ae862121974b48210964b9a81abd55ae4a6a35cf7e7758840ba854f9faf0f"
    )
    assert payload["population_assignment_rows"] == 8_323
    assert payload["unique_candidate_rows"] == 8_323
    assert payload["cross_population_duplicate_identity_rows"] == 0
    assert payload["decision_counts"] == {
        "cleanup_review": 9,
        "hard_reject": 1,
        "pass_mechanical_gate": 8_313,
    }
    assert payload["reason_counts"] == {
        "contextless_scored_answer_sheet": 1,
        "control_character_corruption": 1,
        "duplicated_boilerplate": 9,
    }
    assert payload["publication_contains_source_text"] is False
    assert payload["mechanical_pass_is_semantic_admission"] is False
    assert payload["training_ready"] is False
    assert len(payload["populations"]) == 12
    for population in payload["populations"]:
        path = RECEIPTS / population["receipt_file"]
        assert sha256_file(path) == population["receipt_file_sha256"]
        receipt = json.loads(path.read_text())
        receipt_unsigned = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        assert receipt["receipt_sha256"] == canonical_sha256(receipt_unsigned)
        assert receipt["training_ready"] is False
