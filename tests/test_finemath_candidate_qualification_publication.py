import json
from pathlib import Path

from sai.data.token_stream import canonical_sha256, sha256_file


def test_source_safe_finemath_candidate_publication_replays() -> None:
    root = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "sai_finemath_candidate_qualification_20260825_r1"
    )
    publication = json.loads((root / "publication.json").read_text())
    unsigned = {
        key: value for key, value in publication.items() if key != "receipt_sha256"
    }
    assert publication["receipt_sha256"] == canonical_sha256(unsigned)
    descriptors = (
        (
            root / "extraction" / "aggregate-receipt.json",
            publication["mechanical_candidate"]["aggregate_receipt_file_sha256"],
        ),
        (
            root / "decontamination" / "receipt.json",
            publication["official_boundary"]["receipt_file_sha256"],
        ),
        (
            root / "answer-key-filter-receipt.json",
            publication["contextless_answer_key_filter"]["receipt_file_sha256"],
        ),
        (
            root / "audit" / "filtered-population-receipt.json",
            publication["semantic_audit_population"]["receipt_file_sha256"],
        ),
        (
            root / "accounting" / "validation.out",
            publication["official_boundary"]["validation_output_file_sha256"],
        ),
    )
    for path, expected in descriptors:
        assert sha256_file(path) == expected
    assert publication["official_boundary"]["accepted_rows"] == 52_277
    assert publication["official_boundary"]["dropped_rows"] == 4_377
    assert publication["contextless_answer_key_filter"]["dropped_rows"] == 0
    assert publication["training_ready"] is False
    assert publication["four_b_training_authorized"] is False
