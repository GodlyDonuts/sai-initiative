from sai.data.grounded_bridge_population import (
    QUALIFICATION_SHA256,
    build_pair_plan,
    judgment_qualifies,
)


def judgment(domain: str, bridge: str) -> dict:
    return {
        "source_language": "english",
        "confidence_ppm": 900_000,
        "scores": {"source_reliability": 4, "educational_value": 4},
        "preservation_policy": "preserve_plus_derivatives",
        "risks": {},
        "cross_domain_bridges": [bridge],
        "domains": [domain],
        "subdomains": [domain],
        "concepts_taught": [f"{domain} concept"],
        "prerequisites_assumed": [],
        "evidence_quotes": [domain],
        "judgment_sha256": ("a" if domain == "mathematics" else "b") * 64,
    }


def anchor(index: int, domain: str, bridge: str) -> dict:
    identity = f"{index:064x}"
    content = f"{index + 1000:064x}"
    candidate = {
        "candidate_identity_sha256": identity,
        "source_content_sha256": content,
        "source": {
            "dataset": "unit",
            "revision": "rev",
            "row_id": identity,
            "license": "test",
            "source_type": "reference",
        },
        "text": f"source text {index}",
    }
    return {"candidate": candidate, "judgment": judgment(domain, bridge)}


def test_anchor_floor_rejects_unreliable_or_non_english_judgments() -> None:
    row = judgment("mathematics", "mathematics::computer_science")
    assert judgment_qualifies(row)
    row["source_language"] = "french"
    assert not judgment_qualifies(row)
    row["source_language"] = "english"
    row["risks"]["factual_unreliability"] = True
    assert not judgment_qualifies(row)


def test_pair_plan_is_deterministic_disjoint_and_non_result() -> None:
    rows = [
        anchor(1, "mathematics", "mathematics::computer_science"),
        anchor(2, "computer_science", "computer_science::mathematics"),
        anchor(3, "mathematics", "mathematics::computer_science"),
        anchor(4, "computer_science", "computer_science::mathematics"),
    ]
    first = build_pair_plan(rows, target_pairs=2, seed=7)
    second = build_pair_plan(rows, target_pairs=2, seed=7)
    assert first == second
    assert all(row["source_disjoint"] is True for row in first)
    assert all(row["proposal_verified"] is False for row in first)
    assert all(row["training_ready"] is False for row in first)
    assert all(
        row["anchor_a"]["source_content_sha256"]
        != row["anchor_b"]["source_content_sha256"]
        for row in first
    )
    assert len(QUALIFICATION_SHA256) == 64
