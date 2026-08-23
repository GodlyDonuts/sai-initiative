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
PUBLICATION_R2 = (
    ROOT
    / "artifacts"
    / "sai_source_mechanical_quality_gate_publication_20260826_r2.json"
)
RECEIPTS_R2 = ROOT / "artifacts" / "sai_source_mechanical_quality_gate_20260826_r4"
PUBLICATION_R3 = (
    ROOT
    / "artifacts"
    / "sai_source_mechanical_quality_gate_publication_20260826_r3.json"
)
CODE_RECEIPTS = ROOT / "artifacts" / "sai_source_mechanical_quality_gate_20260826_r5"


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


def test_revised_source_safe_quality_gate_publication_is_exact() -> None:
    payload = json.loads(PUBLICATION_R2.read_text())
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    assert payload["receipt_sha256"] == canonical_sha256(unsigned)
    assert payload["receipt_sha256"] == (
        "50a641ecb9f5570235fc2bd50f33cf41c1fcbaa4ff03d71dcfbfbea8e9b71a82"
    )
    assert payload["policy_sha256"] == (
        "436ea538156447a7188a15404764302c7b3290b3a06c12677d316f265ccc6c80"
    )
    assert payload["population_assignment_rows"] == 8_323
    assert payload["unique_candidate_rows"] == 8_323
    assert payload["cross_population_duplicate_identity_rows"] == 0
    assert payload["decision_counts"] == {
        "cleanup_review": 9,
        "context_review": 1,
        "hard_reject": 1,
        "pass_mechanical_gate": 8_312,
    }
    assert payload["reason_counts"] == {
        "contextless_metadata_form": 1,
        "contextless_scored_answer_sheet": 1,
        "control_character_corruption": 1,
        "duplicated_boilerplate": 9,
    }
    assert payload["publication_contains_source_text"] is False
    assert payload["mechanical_pass_is_semantic_admission"] is False
    assert payload["training_ready"] is False
    assert len(payload["populations"]) == 12
    for population in payload["populations"]:
        path = RECEIPTS_R2 / population["receipt_file"]
        assert sha256_file(path) == population["receipt_file_sha256"]
        receipt = json.loads(path.read_text())
        receipt_unsigned = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        assert receipt["receipt_sha256"] == canonical_sha256(receipt_unsigned)
        assert receipt["training_ready"] is False


def test_code_expanded_source_safe_quality_gate_publication_is_exact() -> None:
    payload = json.loads(PUBLICATION_R3.read_text())
    unsigned = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    assert payload["receipt_sha256"] == canonical_sha256(unsigned)
    assert payload["receipt_sha256"] == (
        "df34d6507032269351df3d841032e068de5ff986dcbcb7d5f92f212e98e82385"
    )
    assert payload["policy_sha256"] == (
        "436ea538156447a7188a15404764302c7b3290b3a06c12677d316f265ccc6c80"
    )
    assert payload["population_assignment_rows"] == 10_371
    assert payload["unique_candidate_rows"] == 10_371
    assert payload["cross_population_duplicate_identity_rows"] == 0
    assert payload["cross_population_duplicate_assignments"] == 0
    assert payload["unique_source_content_rows"] == 10_371
    assert payload["cross_population_duplicate_content_rows"] == 0
    assert payload["cross_population_duplicate_content_assignments"] == 0
    assert payload["decision_counts"] == {
        "cleanup_review": 9,
        "context_review": 1,
        "hard_reject": 1,
        "pass_mechanical_gate": 10_360,
    }
    assert payload["publication_contains_source_text"] is False
    assert payload["mechanical_pass_is_semantic_admission"] is False
    assert payload["training_ready"] is False
    assert len(payload["populations"]) == 13
    for population in payload["populations"]:
        receipt_root = (
            CODE_RECEIPTS
            if population["receipt_file"] == "opencoder_code.receipt.json"
            else RECEIPTS_R2
        )
        path = receipt_root / population["receipt_file"]
        assert sha256_file(path) == population["receipt_file_sha256"]
        receipt = json.loads(path.read_text())
        receipt_unsigned = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        assert receipt["receipt_sha256"] == canonical_sha256(receipt_unsigned)
        assert receipt["training_ready"] is False
