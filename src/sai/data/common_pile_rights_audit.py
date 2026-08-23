"""Seal exact Common Pile card and per-row license-declaration evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.license_policy import POLICY as LICENSE_POLICY
from sai.data.license_policy import classify_declared_license
from sai.data.reservoir_audit_aggregate import load_population
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-common-pile-rights-declaration-audit-v1"


class CommonPileRightsAuditError(RuntimeError):
    """A population, pinned card, row declaration, or output differs."""


def summarize_rights(lineage: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize exact declaration labels without retaining document text."""

    if not lineage:
        raise CommonPileRightsAuditError("rights audit lineage is empty")
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    declaration_counts: Counter[str] = Counter()
    canonical_counts: Counter[str] = Counter()
    classification_digest = []
    for row in lineage:
        if not isinstance(row, dict):
            raise CommonPileRightsAuditError("rights audit lineage differs")
        source_id = row.get("source_id")
        declaration = row.get("declared_license")
        if not isinstance(source_id, str) or not source_id:
            raise CommonPileRightsAuditError("rights audit source differs")
        classification = classify_declared_license(declaration)
        declaration_counts[classification["declared_license"]] += 1
        canonical = classification["canonical_license"] or "<rights_hold>"
        canonical_counts[canonical] += 1
        counter = by_source[source_id]
        counter["rows"] += 1
        counter["recognized_declaration_rows"] += classification[
            "declaration_recognized"
        ]
        counter["rights_hold_rows"] += classification["rights_hold"]
        counter["attribution_required_rows"] += (
            classification["attribution_required"] is True
        )
        counter["share_alike_required_rows"] += (
            classification["share_alike_required"] is True
        )
        counter[f"canonical_license:{canonical}"] += 1
        classification_digest.append(classification["classification_sha256"])
    return {
        "rows": len(lineage),
        "sources": len(by_source),
        "by_source": {
            source_id: dict(sorted(counter.items()))
            for source_id, counter in sorted(by_source.items())
        },
        "declared_license_counts": dict(sorted(declaration_counts.items())),
        "canonical_license_counts": dict(sorted(canonical_counts.items())),
        "ordered_classifications_sha256": canonical_sha256(classification_digest),
        "source_text_persisted": False,
        "source_wide_rights_clearance_established": False,
        "legal_clearance_established": False,
    }


def acquire_card(
    repository: str, revision: str, token: str, local_dir: Path
) -> dict[str, Any]:
    """Resolve and hash one exact dataset README without retaining its text."""

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise CommonPileRightsAuditError("huggingface_hub is required") from error
    try:
        api = HfApi(token=token or None)
        info = api.dataset_info(repository, revision=revision, files_metadata=False)
        readme = Path(
            hf_hub_download(
                repository,
                "README.md",
                repo_type="dataset",
                revision=revision,
                token=token or None,
                local_dir=local_dir,
            )
        )
    except Exception as error:
        raise CommonPileRightsAuditError(
            "pinned dataset card acquisition failed"
        ) from error
    card_data = info.card_data
    if card_data is None:
        metadata: dict[str, Any] = {}
    elif hasattr(card_data, "to_dict"):
        metadata = card_data.to_dict()
    else:
        metadata = dict(card_data)
    if info.sha != revision or not readme.is_file() or readme.is_symlink():
        raise CommonPileRightsAuditError("pinned dataset card identity differs")
    return {
        "repository": repository,
        "revision": revision,
        "readme_bytes": readme.stat().st_size,
        "readme_sha256": sha256_file(readme),
        "top_level_card_license": metadata.get("license"),
        "card_text_persisted": False,
    }


def build_audit(
    population_root: Path,
    output_path: Path,
    *,
    token: str,
    acquire_card_function: Callable[
        [str, str, str, Path], dict[str, Any]
    ] = acquire_card,
) -> dict[str, Any]:
    """Replay one population and seal card plus declaration evidence."""

    if output_path.exists() or output_path.is_symlink():
        raise CommonPileRightsAuditError("rights audit output already exists")
    _candidates, lineage, population_receipt = load_population(population_root)
    identities: dict[str, tuple[str, str]] = {}
    for row in lineage:
        source_id = row.get("source_id")
        repository = row.get("repository")
        revision = row.get("revision")
        if (
            not isinstance(source_id, str)
            or not isinstance(repository, str)
            or not isinstance(revision, str)
            or len(revision) != 40
        ):
            raise CommonPileRightsAuditError("rights card lineage differs")
        identity = (repository, revision)
        prior = identities.setdefault(source_id, identity)
        if prior != identity:
            raise CommonPileRightsAuditError("one source maps to multiple cards")
    cards = []
    with tempfile.TemporaryDirectory(prefix="sai-common-rights-cards-") as temporary:
        for source_id, (repository, revision) in sorted(identities.items()):
            card = acquire_card_function(repository, revision, token, Path(temporary))
            if (
                card.get("repository") != repository
                or card.get("revision") != revision
                or not isinstance(card.get("readme_bytes"), int)
                or card["readme_bytes"] <= 0
                or not isinstance(card.get("readme_sha256"), str)
                or len(card["readme_sha256"]) != 64
                or card.get("card_text_persisted") is not False
            ):
                raise CommonPileRightsAuditError("pinned card receipt differs")
            cards.append({"source_id": source_id, **card})
    summary = summarize_rights(lineage)
    payload = {
        "schema": SCHEMA,
        "status": "complete_declaration_audit_not_legal_clearance",
        "population": {
            "root_name": population_root.name,
            "receipt_sha256": population_receipt["receipt_sha256"],
            "population_sha256": sha256_file(population_root / "candidates.jsonl"),
            "lineage_sha256": sha256_file(population_root / "lineage.jsonl"),
        },
        "license_policy": LICENSE_POLICY,
        "license_policy_sha256": canonical_sha256(LICENSE_POLICY),
        "cards": cards,
        "cards_sha256": canonical_sha256(cards),
        "summary": summary,
        "unknown_or_ambiguous_declarations_fail_closed": True,
        "source_provenance_verification_complete": False,
        "source_wide_rights_clearance_established": False,
        "legal_clearance_established": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    import os

    result = build_audit(
        args.population_root,
        args.output,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
