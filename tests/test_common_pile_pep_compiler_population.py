from sai.data.common_pile_pep_compiler_population import (
    SOURCE_ID,
    STRATUM,
    _candidate_and_lineage,
)
from sai.data.token_stream import canonical_sha256, normalize_document


def test_pep_candidate_is_excerpted_and_source_bound() -> None:
    raw_document = {
        "schema": "sai-pretraining-document-v1",
        "text": "Python enhancement proposal body. " * 2_000,
        "source": {
            "dataset": "common-pile/python_enhancement_proposals_filtered",
            "row_id": "a" * 64,
            "license": "LicenseRef-Public-Domain",
            "domain": "code",
        },
        "verification": {
            "benchmark_disjoint": True,
            "evidence_sha256": "b" * 64,
        },
    }
    raw_document["identity_sha256"] = canonical_sha256(raw_document)
    document = normalize_document(raw_document)
    attribution = {
        "source": {
            "dataset": raw_document["source"]["dataset"],
            "revision": "c" * 40,
            "source_file": "peps.json.gz",
            "row_index": 7,
            "domain": "code",
        },
        "rights_declaration": {
            "canonical_license": "LicenseRef-Public-Domain",
            "rights_hold": False,
        },
        "row_id": "a" * 64,
        "record_sha256": "d" * 64,
    }
    candidate, lineage = _candidate_and_lineage(
        document, attribution, "e" * 64, 0
    )
    assert len(candidate["text"].encode()) <= 32 * 1024
    assert candidate["source"]["source_type"] == "documentation"
    assert lineage["source_id"] == SOURCE_ID
    assert lineage["stratum"] == STRATUM
    assert lineage["full_text_bytes"] > lineage["excerpt_bytes"]
    assert lineage["raw_source_is_training_ready"] is False
