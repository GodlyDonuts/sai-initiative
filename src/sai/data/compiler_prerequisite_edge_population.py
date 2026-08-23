"""Build repeated-evidence prerequisite-edge proposals from compiler receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create, normalize_candidate
from sai.data.bounded_pilot_work_queue import _atomic_jsonl
from sai.data.data_compiler_labeling import RUBRIC_SHA256
from sai.data.data_yield_ledger import _load_receipt
from sai.data.nous_compiler_worker import SUMMARY_SCHEMA
from sai.data.nous_label_worker import DEFAULT_MODEL, _assigned
from sai.data.reservoir_audit_aggregate import _validate_compiler_receipt
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-compiler-prerequisite-edge-population-v1"
ROW_SCHEMA = "sai-compiler-prerequisite-edge-verification-candidate-v1"
DEFAULT_EDGES = 192
MINIMUM_SUPPORTING_DOCUMENTS = 2
MAXIMUM_SUPPORTING_ANCHORS = 3
MAXIMUM_EDGES_PER_LABEL = 12
SEED = 20260826
DISQUALIFYING_RISKS = (
    "seo_or_content_farm",
    "incoherent_or_corrupted",
    "factual_unreliability",
    "answer_farm_without_teaching",
    "personal_or_secret_data",
    "weak_source_grounding",
    "license_or_provenance_unclear",
)
QUALIFICATION = {
    "verdict_excluded": "reject",
    "source_language": "english",
    "minimum_confidence_ppm": 800_000,
    "minimum_source_reliability": 3,
    "minimum_educational_value": 3,
    "minimum_supporting_documents": MINIMUM_SUPPORTING_DOCUMENTS,
    "maximum_supporting_anchors": MAXIMUM_SUPPORTING_ANCHORS,
    "maximum_edges_per_prerequisite_or_concept_label": MAXIMUM_EDGES_PER_LABEL,
    "support_requires_distinct_candidate_and_content_identity": True,
    "compiler_prerequisite_cooccurrence_is_verified_edge": False,
}
QUALIFICATION_SHA256 = canonical_sha256(QUALIFICATION)


class CompilerPrerequisiteEdgePopulationError(RuntimeError):
    """Compiler custody, repeated support, or edge selection differs."""


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise CompilerPrerequisiteEdgePopulationError(
            "prerequisite candidate population is unsafe"
        )
    rows = []
    try:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(normalize_candidate(json.loads(line)))
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        raise CompilerPrerequisiteEdgePopulationError(
            "prerequisite candidate population differs"
        ) from error
    identities = [row["candidate_identity_sha256"] for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise CompilerPrerequisiteEdgePopulationError(
            "prerequisite candidate identities differ"
        )
    return rows


def _load_compiler_receipt(path: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
        return _validate_compiler_receipt(payload, candidate)
    except (OSError, UnicodeError, json.JSONDecodeError, RuntimeError) as error:
        raise CompilerPrerequisiteEdgePopulationError(
            "prerequisite compiler receipt differs"
        ) from error


def _validate_summaries(
    candidates: list[dict[str, Any]], judgments_root: Path, logical_shards: int
) -> list[str]:
    expected_paths = {
        judgments_root / f"shard_{index:05d}.summary.json"
        for index in range(logical_shards)
    }
    if set(judgments_root.glob("shard_*.summary.json")) != expected_paths:
        raise CompilerPrerequisiteEdgePopulationError(
            "prerequisite compiler shard population differs"
        )
    hashes = []
    for index in range(logical_shards):
        summary = _load_receipt(judgments_root / f"shard_{index:05d}.summary.json")
        expected = sum(
            _assigned(row["candidate_identity_sha256"], logical_shards, index)
            for row in candidates
        )
        created = summary.get("created_judgments")
        preexisting = summary.get("preexisting_judgments")
        if (
            summary.get("schema") != SUMMARY_SCHEMA
            or summary.get("status") != "complete"
            or summary.get("model") != DEFAULT_MODEL
            or summary.get("rubric_sha256") != RUBRIC_SHA256
            or summary.get("logical_shards") != logical_shards
            or summary.get("shard_index") != index
            or summary.get("candidate_rows") != expected
            or summary.get("expected_judgments") != expected
            or not isinstance(created, int)
            or isinstance(created, bool)
            or created < 0
            or not isinstance(preexisting, int)
            or isinstance(preexisting, bool)
            or preexisting < 0
            or created + preexisting != expected
            or summary.get("api_key_persisted") is not False
            or summary.get("training_ready") is not False
        ):
            raise CompilerPrerequisiteEdgePopulationError(
                "prerequisite compiler shard summary differs"
            )
        hashes.append(summary["receipt_sha256"])
    return hashes


def judgment_qualifies(judgment: dict[str, Any]) -> bool:
    """Apply the conservative source floor before edge co-occurrence."""

    scores = judgment.get("scores")
    risks = judgment.get("risks")
    return bool(
        judgment.get("verdict") != QUALIFICATION["verdict_excluded"]
        and judgment.get("source_language") == QUALIFICATION["source_language"]
        and isinstance(judgment.get("confidence_ppm"), int)
        and judgment["confidence_ppm"] >= QUALIFICATION["minimum_confidence_ppm"]
        and isinstance(scores, dict)
        and scores.get("source_reliability", -1)
        >= QUALIFICATION["minimum_source_reliability"]
        and scores.get("educational_value", -1)
        >= QUALIFICATION["minimum_educational_value"]
        and isinstance(risks, dict)
        and not any(risks.get(key) is True for key in DISQUALIFYING_RISKS)
        and isinstance(judgment.get("domains"), list)
        and judgment["domains"]
        and isinstance(judgment.get("concepts_taught"), list)
        and judgment["concepts_taught"]
        and isinstance(judgment.get("prerequisites_assumed"), list)
        and judgment["prerequisites_assumed"]
    )


def _anchor(candidate: dict[str, Any], judgment: dict[str, Any]) -> dict[str, Any]:
    if (
        hashlib.sha256(candidate["text"].encode()).hexdigest()
        != candidate["source_content_sha256"]
    ):
        raise CompilerPrerequisiteEdgePopulationError(
            "prerequisite source content binding differs"
        )
    return {
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "source_content_sha256": candidate["source_content_sha256"],
        "source": candidate["source"],
        "text": candidate["text"],
        "compiler_judgment_sha256": judgment["judgment_sha256"],
        "domains": judgment["domains"],
        "evidence_quotes": judgment["evidence_quotes"],
        "confidence_ppm": judgment["confidence_ppm"],
    }


def _primary_domain(support: list[dict[str, Any]]) -> str:
    counts = Counter(domain for row in support for domain in row["judgment"]["domains"])
    maximum = max(counts.values())
    return min(domain for domain, count in counts.items() if count == maximum)


def build_edge_plan(
    anchors: list[dict[str, Any]],
    *,
    target_edges: int = DEFAULT_EDGES,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Select repeated, source-disjoint co-occurrences as verification work."""

    if (
        isinstance(target_edges, bool)
        or not isinstance(target_edges, int)
        or target_edges <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise CompilerPrerequisiteEdgePopulationError(
            "prerequisite edge geometry differs"
        )
    by_edge: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        judgment = anchor["judgment"]
        for prerequisite in judgment["prerequisites_assumed"]:
            for concept in judgment["concepts_taught"]:
                if prerequisite != concept:
                    by_edge[(prerequisite, concept)].append(anchor)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (prerequisite, concept), raw_support in by_edge.items():
        support_by_content = {}
        candidate_identities = set()
        for anchor in raw_support:
            candidate = anchor["candidate"]
            identity = candidate["candidate_identity_sha256"]
            content = candidate["source_content_sha256"]
            if identity in candidate_identities:
                continue
            candidate_identities.add(identity)
            support_by_content.setdefault(content, anchor)
        support = list(support_by_content.values())
        if len(support) < MINIMUM_SUPPORTING_DOCUMENTS:
            continue
        edge_identity = canonical_sha256(
            {
                "prerequisite": prerequisite,
                "concept": concept,
                "qualification_sha256": QUALIFICATION_SHA256,
            }
        )
        support.sort(
            key=lambda row: hashlib.sha256(
                (
                    f"{seed}:{edge_identity}:"
                    f"{row['candidate']['candidate_identity_sha256']}"
                ).encode()
            ).hexdigest()
        )
        selected_support = support[:MAXIMUM_SUPPORTING_ANCHORS]
        domain = _primary_domain(support)
        row = {
            "schema": ROW_SCHEMA,
            "edge_identity_sha256": edge_identity,
            "candidate_identity_sha256": edge_identity,
            "prerequisite": prerequisite,
            "concept": concept,
            "primary_domain": domain,
            "supporting_documents": len(support),
            "supporting_anchors": [
                _anchor(item["candidate"], item["judgment"])
                for item in selected_support
            ],
            "supporting_anchor_selection": {
                "seed": seed,
                "available_distinct_documents": len(support),
                "selected_documents": len(selected_support),
                "ordered_candidate_identities_sha256": canonical_sha256(
                    [
                        item["candidate"]["candidate_identity_sha256"]
                        for item in selected_support
                    ]
                ),
            },
            "source_disjoint_support": True,
            "compiler_cooccurrence_only": True,
            "directional_prerequisite_verified": False,
            "acyclic_graph_construction_complete": False,
            "training_ready": False,
        }
        by_domain[domain].append(row)
    for domain, rows in by_domain.items():
        rows.sort(
            key=lambda row: (
                -row["supporting_documents"],
                hashlib.sha256(
                    f"{seed}:{domain}:{row['edge_identity_sha256']}".encode()
                ).hexdigest(),
            )
        )
    selected = []
    prerequisite_counts: Counter[str] = Counter()
    concept_counts: Counter[str] = Counter()
    domains = sorted(
        by_domain,
        key=lambda domain: hashlib.sha256(f"{seed}:{domain}".encode()).hexdigest(),
    )
    while len(selected) < target_edges:
        progress = False
        for domain in domains:
            rows = by_domain[domain]
            while rows:
                row = rows.pop(0)
                if (
                    prerequisite_counts[row["prerequisite"]] < MAXIMUM_EDGES_PER_LABEL
                    and concept_counts[row["concept"]] < MAXIMUM_EDGES_PER_LABEL
                ):
                    selected.append(row)
                    prerequisite_counts[row["prerequisite"]] += 1
                    concept_counts[row["concept"]] += 1
                    progress = True
                    break
            if len(selected) == target_edges:
                break
        if not progress:
            break
    if len(selected) != target_edges:
        raise CompilerPrerequisiteEdgePopulationError(
            f"prerequisite edge population underfilled: {len(selected)} of "
            f"{target_edges}"
        )
    return selected


def build_population(
    candidate_paths: list[Path],
    judgment_roots: list[Path],
    logical_shards: list[int],
    output_root: Path,
    *,
    target_edges: int = DEFAULT_EDGES,
    seed: int = SEED,
) -> dict[str, Any]:
    """Replay complete compiler populations and freeze edge-verification work."""

    if (
        not candidate_paths
        or len(candidate_paths) != len(judgment_roots)
        or len(candidate_paths) != len(logical_shards)
        or output_root.exists()
        or output_root.is_symlink()
    ):
        raise CompilerPrerequisiteEdgePopulationError(
            "prerequisite edge input geometry differs"
        )
    anchors = []
    inputs = []
    seen_identities = set()
    for order, (candidate_path, judgments_root, shards) in enumerate(
        zip(candidate_paths, judgment_roots, logical_shards, strict=True)
    ):
        if isinstance(shards, bool) or not isinstance(shards, int) or shards <= 0:
            raise CompilerPrerequisiteEdgePopulationError(
                "prerequisite logical shards differ"
            )
        candidates = _load_candidates(candidate_path)
        expected_receipts = {
            judgments_root / f"{row['candidate_identity_sha256']}.compiler.json"
            for row in candidates
        }
        if set(judgments_root.glob("*.compiler.json")) != expected_receipts:
            raise CompilerPrerequisiteEdgePopulationError(
                "prerequisite compiler population differs"
            )
        receipt_hashes = []
        qualified = 0
        for candidate in candidates:
            identity = candidate["candidate_identity_sha256"]
            if identity in seen_identities:
                raise CompilerPrerequisiteEdgePopulationError(
                    "prerequisite populations overlap"
                )
            seen_identities.add(identity)
            receipt = _load_compiler_receipt(
                judgments_root / f"{identity}.compiler.json", candidate
            )
            receipt_hashes.append(receipt["receipt_sha256"])
            judgment = receipt["judgment"]
            if judgment_qualifies(judgment):
                anchors.append({"candidate": candidate, "judgment": judgment})
                qualified += 1
        summary_hashes = _validate_summaries(candidates, judgments_root, shards)
        inputs.append(
            {
                "order": order,
                "candidate_path": str(candidate_path.resolve()),
                "candidate_rows": len(candidates),
                "candidate_bytes": candidate_path.stat().st_size,
                "candidate_sha256": sha256_file(candidate_path),
                "judgments_root": str(judgments_root.resolve()),
                "logical_shards": shards,
                "compiler_receipts": len(receipt_hashes),
                "qualified_anchors": qualified,
                "ordered_compiler_receipts_sha256": canonical_sha256(receipt_hashes),
                "ordered_shard_summaries_sha256": canonical_sha256(summary_hashes),
            }
        )
    rows = build_edge_plan(anchors, target_edges=target_edges, seed=seed)
    output_root.mkdir(parents=True)
    try:
        candidates_path = output_root / "candidates.jsonl"
        _atomic_jsonl(candidates_path, rows)
        domain_counts = Counter(row["primary_domain"] for row in rows)
        payload = {
            "schema": SCHEMA,
            "status": "complete_nontraining_prerequisite_edge_proposals",
            "inputs": inputs,
            "qualification": QUALIFICATION,
            "qualification_sha256": QUALIFICATION_SHA256,
            "selection": {
                "seed": seed,
                "target_edges": target_edges,
                "selected_edges": len(rows),
                "qualified_anchors": len(anchors),
                "domains": dict(sorted(domain_counts.items())),
                "maximum_edges_per_label": MAXIMUM_EDGES_PER_LABEL,
                "ordered_edge_identities_sha256": canonical_sha256(
                    [row["edge_identity_sha256"] for row in rows]
                ),
            },
            "candidates": {
                "path": candidates_path.name,
                "rows": len(rows),
                "bytes": candidates_path.stat().st_size,
                "sha256": sha256_file(candidates_path),
                "text_bytes": sum(
                    len(anchor["text"].encode())
                    for row in rows
                    for anchor in row["supporting_anchors"]
                ),
            },
            "all_compiler_populations_complete": True,
            "source_disjoint_support": True,
            "compiler_cooccurrence_is_verified_edge": False,
            "directional_prerequisite_verification_complete": False,
            "acyclic_graph_construction_complete": False,
            "training_ready": False,
            "four_b_training_authorized": False,
        }
        payload["receipt_sha256"] = canonical_sha256(payload)
        _atomic_create(output_root / "receipt.json", payload)
        return payload
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--judgments", type=Path, action="append", required=True)
    parser.add_argument("--logical-shards", type=int, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-edges", type=int, default=DEFAULT_EDGES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    result = build_population(
        args.candidates,
        args.judgments,
        args.logical_shards,
        args.output_root,
        target_edges=args.target_edges,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
