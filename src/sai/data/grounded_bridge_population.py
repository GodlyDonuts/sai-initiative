"""Build source-disjoint cross-domain bridge proposals from Hermes judgments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import normalize_candidate
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-grounded-cross-domain-bridge-population-v1"
ROW_SCHEMA = "sai-grounded-cross-domain-bridge-candidate-v1"
SEED = 20260825
MAXIMUM_PAIRS_PER_DIRECTED_BRIDGE = 8
MAXIMUM_PAIRS_PER_ANCHOR = 2
DEFAULT_PAIRS = 512
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
    "source_language": "english",
    "minimum_confidence_ppm": 800_000,
    "minimum_source_reliability": 3,
    "minimum_educational_value": 3,
    "preservation_policy_excluded": "discard",
    "disqualifying_risks": list(DISQUALIFYING_RISKS),
    "required_bridge_delimiter": "::",
    "pair_source_disjointness": "different_candidate_and_source_content_sha256",
    "proposal_is_not_verified_connection_data": True,
}
QUALIFICATION_SHA256 = canonical_sha256(QUALIFICATION)


class GroundedBridgePopulationError(RuntimeError):
    """A compiler receipt, source anchor, pair, or frozen output differs."""


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
        raise GroundedBridgePopulationError("candidate population is missing")
    rows = []
    try:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(normalize_candidate(json.loads(line)))
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        raise GroundedBridgePopulationError("candidate population differs") from error
    identities = [row["candidate_identity_sha256"] for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise GroundedBridgePopulationError("candidate identities differ")
    return rows


def _compiler_receipts(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir() or root.is_symlink():
        raise GroundedBridgePopulationError("compiler receipt root is missing")
    rows = []
    for path in sorted(root.rglob("*.compiler.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise GroundedBridgePopulationError(
                "compiler receipt is unreadable"
            ) from error
        unsigned = {
            key: value for key, value in payload.items() if key != "receipt_sha256"
        }
        if (
            payload.get("schema") != "sai-nous-data-compiler-receipt-v2"
            or payload.get("status") != "complete"
            or payload.get("receipt_sha256") != canonical_sha256(unsigned)
            or payload.get("training_ready") is not False
            or not isinstance(payload.get("judgment"), dict)
            or payload["judgment"].get("candidate_identity_sha256")
            != payload.get("candidate_identity_sha256")
        ):
            raise GroundedBridgePopulationError("compiler receipt differs")
        rows.append(payload)
    if not rows:
        raise GroundedBridgePopulationError("compiler receipt root is empty")
    return rows


def judgment_qualifies(judgment: dict[str, Any]) -> bool:
    """Apply the prospective anchor floor to one normalized compiler judgment."""

    scores = judgment.get("scores")
    risks = judgment.get("risks")
    bridges = judgment.get("cross_domain_bridges")
    return bool(
        judgment.get("source_language") == QUALIFICATION["source_language"]
        and isinstance(judgment.get("confidence_ppm"), int)
        and judgment["confidence_ppm"] >= QUALIFICATION["minimum_confidence_ppm"]
        and isinstance(scores, dict)
        and scores.get("source_reliability", -1)
        >= QUALIFICATION["minimum_source_reliability"]
        and scores.get("educational_value", -1)
        >= QUALIFICATION["minimum_educational_value"]
        and judgment.get("preservation_policy")
        != QUALIFICATION["preservation_policy_excluded"]
        and isinstance(risks, dict)
        and not any(risks.get(key) is True for key in DISQUALIFYING_RISKS)
        and isinstance(bridges, list)
        and bridges
        and isinstance(judgment.get("domains"), list)
        and judgment["domains"]
        and isinstance(judgment.get("concepts_taught"), list)
        and judgment["concepts_taught"]
    )


def _bridge(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str) or value.count("::") != 1:
        return None
    left, right = value.split("::")
    if not left or not right or left == right:
        return None
    return left, right


def _anchor(candidate: dict[str, Any], judgment: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_identity_sha256": candidate["candidate_identity_sha256"],
        "source_content_sha256": candidate["source_content_sha256"],
        "source": candidate["source"],
        "text": candidate["text"],
        "compiler": {
            "judgment_sha256": judgment["judgment_sha256"],
            "domains": judgment["domains"],
            "subdomains": judgment["subdomains"],
            "concepts_taught": judgment["concepts_taught"],
            "prerequisites_assumed": judgment["prerequisites_assumed"],
            "evidence_quotes": judgment["evidence_quotes"],
            "scores": judgment["scores"],
            "confidence_ppm": judgment["confidence_ppm"],
        },
    }


def build_pair_plan(
    anchors: list[dict[str, Any]], *, target_pairs: int, seed: int = SEED
) -> list[dict[str, Any]]:
    """Select a coverage-first deterministic bridge plan without generating prose."""

    if (
        isinstance(target_pairs, bool)
        or not isinstance(target_pairs, int)
        or target_pairs <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise GroundedBridgePopulationError("bridge selection geometry differs")
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        for domain in anchor["judgment"]["domains"]:
            by_domain[domain].append(anchor)
    proposals: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proposal_identities = set()
    for source in anchors:
        source_domains = set(source["judgment"]["domains"])
        for label in source["judgment"]["cross_domain_bridges"]:
            endpoints = _bridge(label)
            if endpoints is None:
                continue
            left, right = endpoints
            target_domain = (
                right
                if left in source_domains
                else left if right in source_domains else None
            )
            if target_domain is None:
                continue
            candidates = []
            for partner in by_domain.get(target_domain, []):
                if (
                    source["candidate"]["candidate_identity_sha256"]
                    == partner["candidate"]["candidate_identity_sha256"]
                    or source["candidate"]["source_content_sha256"]
                    == partner["candidate"]["source_content_sha256"]
                ):
                    continue
                selection_key = hashlib.sha256(
                    (
                        f"{seed}:{label}:"
                        f"{source['candidate']['candidate_identity_sha256']}:"
                        f"{partner['candidate']['candidate_identity_sha256']}"
                    ).encode()
                ).hexdigest()
                candidates.append((selection_key, partner))
            if not candidates:
                continue
            selection_key, partner = min(candidates, key=lambda item: item[0])
            identity_payload = {
                "bridge_label": label,
                "anchor_a": source["candidate"]["candidate_identity_sha256"],
                "anchor_b": partner["candidate"]["candidate_identity_sha256"],
                "qualification_sha256": QUALIFICATION_SHA256,
            }
            pair_identity = canonical_sha256(identity_payload)
            if pair_identity in proposal_identities:
                continue
            proposal_identities.add(pair_identity)
            proposals[label].append(
                {
                    "schema": ROW_SCHEMA,
                    "bridge_label": label,
                    "bridge_endpoints": [left, right],
                    "anchor_a": _anchor(source["candidate"], source["judgment"]),
                    "anchor_b": _anchor(partner["candidate"], partner["judgment"]),
                    "selection_key": selection_key,
                    "pair_identity_sha256": pair_identity,
                    "source_disjoint": True,
                    "proposal_verified": False,
                    "training_ready": False,
                }
            )
    for rows in proposals.values():
        rows.sort(key=lambda row: (row["selection_key"], row["pair_identity_sha256"]))
    selected = []
    label_counts: Counter[str] = Counter()
    anchor_counts: Counter[str] = Counter()
    labels = sorted(
        proposals,
        key=lambda label: hashlib.sha256(f"{seed}:{label}".encode()).hexdigest(),
    )
    while len(selected) < target_pairs:
        progress = False
        for label in labels:
            if label_counts[label] >= MAXIMUM_PAIRS_PER_DIRECTED_BRIDGE:
                continue
            while proposals[label]:
                row = proposals[label].pop(0)
                identities = (
                    row["anchor_a"]["candidate_identity_sha256"],
                    row["anchor_b"]["candidate_identity_sha256"],
                )
                if all(
                    anchor_counts[value] < MAXIMUM_PAIRS_PER_ANCHOR
                    for value in identities
                ):
                    selected.append(row)
                    label_counts[label] += 1
                    anchor_counts.update(identities)
                    progress = True
                    break
            if len(selected) == target_pairs:
                break
        if not progress:
            break
    if len(selected) != target_pairs:
        raise GroundedBridgePopulationError(
            f"bridge population underfilled: {len(selected)} of {target_pairs}"
        )
    return selected


def build_population(
    candidate_paths: list[Path],
    judgment_roots: list[Path],
    output_path: Path,
    receipt_path: Path,
    *,
    target_pairs: int = DEFAULT_PAIRS,
    seed: int = SEED,
) -> dict[str, Any]:
    """Replay partial compiler evidence and freeze a non-result bridge population."""

    if (
        len(candidate_paths) != len(judgment_roots)
        or not candidate_paths
        or output_path.exists()
        or output_path.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise GroundedBridgePopulationError("bridge input or output geometry differs")
    input_receipts = []
    anchors = []
    seen_candidates = set()
    for order, (candidate_path, judgment_root) in enumerate(
        zip(candidate_paths, judgment_roots, strict=True)
    ):
        candidates = {
            row["candidate_identity_sha256"]: row for row in _load_jsonl(candidate_path)
        }
        receipts = _compiler_receipts(judgment_root)
        qualified = 0
        for receipt in receipts:
            identity = receipt["candidate_identity_sha256"]
            candidate = candidates.get(identity)
            judgment = receipt["judgment"]
            if candidate is None:
                raise GroundedBridgePopulationError("judgment candidate is absent")
            if identity in seen_candidates:
                raise GroundedBridgePopulationError("candidate populations overlap")
            seen_candidates.add(identity)
            if judgment_qualifies(judgment):
                anchors.append({"candidate": candidate, "judgment": judgment})
                qualified += 1
        input_receipts.append(
            {
                "order": order,
                "candidate_path": str(candidate_path.resolve()),
                "candidate_bytes": candidate_path.stat().st_size,
                "candidate_sha256": sha256_file(candidate_path),
                "candidate_rows": len(candidates),
                "judgment_root": str(judgment_root.resolve()),
                "completed_judgments": len(receipts),
                "qualified_anchors": qualified,
                "ordered_judgment_receipts_sha256": canonical_sha256(
                    [receipt["receipt_sha256"] for receipt in receipts]
                ),
            }
        )
    rows = build_pair_plan(anchors, target_pairs=target_pairs, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    label_counts = Counter(row["bridge_label"] for row in rows)
    payload = {
        "schema": SCHEMA,
        "status": "complete_proposal_population_not_connection_data",
        "inputs": input_receipts,
        "qualification": QUALIFICATION,
        "qualification_sha256": QUALIFICATION_SHA256,
        "selection": {
            "seed": seed,
            "target_pairs": target_pairs,
            "selected_pairs": len(rows),
            "qualified_anchors": len(anchors),
            "directed_bridge_labels": len(label_counts),
            "maximum_pairs_per_directed_bridge": MAXIMUM_PAIRS_PER_DIRECTED_BRIDGE,
            "maximum_pairs_per_anchor": MAXIMUM_PAIRS_PER_ANCHOR,
            "label_counts": dict(sorted(label_counts.items())),
            "ordered_pair_identity_sha256": canonical_sha256(
                [row["pair_identity_sha256"] for row in rows]
            ),
        },
        "population": {
            "path": str(output_path.resolve()),
            "bytes": output_path.stat().st_size,
            "rows": len(rows),
            "sha256": sha256_file(output_path),
        },
        "source_disjoint_pairs": True,
        "all_compiler_populations_complete": False,
        "grounded_synthesis_complete": False,
        "independent_verification_complete": False,
        "benchmark_decontamination_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    temporary = receipt_path.parent / f".{receipt_path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, receipt_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--judgments", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--target-pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    payload = build_population(
        args.candidates,
        args.judgments,
        args.output,
        args.receipt,
        target_pairs=args.target_pairs,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "pairs": payload["selection"]["selected_pairs"],
                "receipt_sha256": payload["receipt_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
