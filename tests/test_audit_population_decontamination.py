from sai.data.audit_population_decontamination import (
    promote_lineage,
    screen_candidate,
)
from sai.data.token_stream import canonical_sha256


def _candidate(text: str) -> dict:
    return {"candidate_identity_sha256": "a" * 64, "text": text}


def test_screen_candidate_emits_text_free_exact_decision() -> None:
    words = {b"a" * 32}
    code: set[bytes] = set()
    decision = screen_candidate(_candidate("ordinary disjoint prose"), words, code)
    assert decision["contaminated"] is False
    assert decision["source_text_persisted"] is False
    assert "text" not in decision


def test_promote_lineage_rebinds_ordinal_and_preserves_custody() -> None:
    source = {
        "schema": "sai-reservoir-audit-lineage-v1",
        "ordinal": 9,
        "candidate_identity_sha256": "a" * 64,
        "source_id": "source",
        "stratum": "tier",
        "repository": "org/repo",
        "revision": "b" * 40,
        "license": "license",
        "excerpt_sha256": "c" * 64,
        "excerpt_bytes": 100,
        "raw_source_is_training_ready": False,
    }
    source["lineage_sha256"] = canonical_sha256(source)
    result = promote_lineage(source, 2)
    assert result["ordinal"] == 2
    assert result["pre_decontamination_ordinal"] == 9
    assert result["pre_decontamination_lineage_sha256"] == source["lineage_sha256"]
    assert result["benchmark_decontamination_complete"] is True
    assert result["lineage_sha256"] == canonical_sha256(
        {key: value for key, value in result.items() if key != "lineage_sha256"}
    )
