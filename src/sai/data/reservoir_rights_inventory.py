"""Inventory exact reservoir cards and declared source-license evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.license_policy import POLICY, classify_declared_license
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-reservoir-rights-inventory-v1"
SUPPORTED_RECEIPT_SCHEMAS = {
    "sai-source-reservoir-receipt-v1",
    "sai-frontier-source-reservoir-receipt-v1",
}


class ReservoirRightsInventoryError(RuntimeError):
    """A reservoir receipt, manifest source, pinned card, or output differs."""


def acquire_card_or_absence(
    repository: str, revision: str, token: str, local_dir: Path
) -> dict[str, Any]:
    """Hash an exact card or record that the pinned tree has no README."""

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise ReservoirRightsInventoryError("huggingface_hub is required") from error
    try:
        info = HfApi(token=token or None).dataset_info(
            repository, revision=revision, files_metadata=False
        )
    except Exception as error:
        raise ReservoirRightsInventoryError(
            "pinned dataset identity acquisition failed"
        ) from error
    if info.sha != revision:
        raise ReservoirRightsInventoryError("pinned dataset identity differs")
    card_data = info.card_data
    if card_data is None:
        metadata: dict[str, Any] = {}
    elif hasattr(card_data, "to_dict"):
        metadata = card_data.to_dict()
    else:
        metadata = dict(card_data)
    files = {sibling.rfilename for sibling in (info.siblings or [])}
    if "README.md" not in files:
        return {
            "repository": repository,
            "revision": revision,
            "readme_present": False,
            "readme_bytes": 0,
            "readme_sha256": None,
            "top_level_card_license": metadata.get("license"),
            "card_text_persisted": False,
        }
    try:
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
        raise ReservoirRightsInventoryError(
            "pinned dataset card acquisition failed"
        ) from error
    if not readme.is_file() or readme.is_symlink():
        raise ReservoirRightsInventoryError("pinned dataset card differs")
    return {
        "repository": repository,
        "revision": revision,
        "readme_present": True,
        "readme_bytes": readme.stat().st_size,
        "readme_sha256": sha256_file(readme),
        "top_level_card_license": metadata.get("license"),
        "card_text_persisted": False,
    }


def _load_receipt(root: Path) -> tuple[dict[str, Any], Path]:
    receipt_path = root / "receipt.json"
    if (
        not receipt_path.is_file()
        or receipt_path.is_symlink()
        or receipt_path.stat().st_nlink != 1
    ):
        raise ReservoirRightsInventoryError("rights reservoir receipt is unsafe")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReservoirRightsInventoryError(
            "rights reservoir receipt is unreadable"
        ) from error
    receipt_sha256 = receipt.get("receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    manifest = receipt.get("manifest")
    if (
        receipt.get("schema") not in SUPPORTED_RECEIPT_SCHEMAS
        or receipt_sha256 != canonical_sha256(unsigned)
        or not isinstance(manifest, dict)
        or not isinstance(manifest.get("path"), str)
    ):
        raise ReservoirRightsInventoryError("rights reservoir receipt differs")
    manifest_path = root / manifest["path"]
    try:
        manifest_path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ReservoirRightsInventoryError(
            "rights reservoir manifest escapes its root"
        ) from error
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path.stat().st_nlink != 1
        or manifest.get("bytes") != manifest_path.stat().st_size
        or manifest.get("sha256") != sha256_file(manifest_path)
    ):
        raise ReservoirRightsInventoryError("rights reservoir manifest differs")
    return receipt, manifest_path


def _sources(
    reservoir_roots: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: dict[str, dict[str, Any]] = {}
    reservoirs = []
    for root in reservoir_roots:
        receipt, manifest_path = _load_receipt(root)
        reservoirs.append(
            {
                "root": str(root.resolve()),
                "receipt_sha256": receipt["receipt_sha256"],
                "manifest_sha256": sha256_file(manifest_path),
                "receipt_schema": receipt["schema"],
            }
        )
        with manifest_path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ReservoirRightsInventoryError(
                        f"rights manifest row {line_number} is malformed"
                    ) from error
                source_id = row.get("source_id")
                repository = row.get("repository")
                revision = row.get("revision")
                declared_license = row.get("license")
                access = row.get("access")
                physical_bytes = row.get("physical_bytes", row.get("bytes"))
                if (
                    not isinstance(source_id, str)
                    or not source_id
                    or not isinstance(repository, str)
                    or not repository
                    or not isinstance(revision, str)
                    or len(revision) != 40
                    or not isinstance(declared_license, str)
                    or not declared_license
                    or not isinstance(access, str)
                    or not access
                    or not isinstance(physical_bytes, int)
                    or physical_bytes <= 0
                ):
                    raise ReservoirRightsInventoryError(
                        f"rights manifest row {line_number} differs"
                    )
                identity = {
                    "source_id": source_id,
                    "repository": repository,
                    "revision": revision,
                    "declared_license": declared_license,
                    "access": access,
                }
                prior = sources.setdefault(
                    source_id, {**identity, "files": 0, "bytes": 0}
                )
                if any(prior[key] != value for key, value in identity.items()):
                    raise ReservoirRightsInventoryError(
                        "one source maps to multiple rights identities"
                    )
                prior["files"] += 1
                prior["bytes"] += physical_bytes
    if not sources:
        raise ReservoirRightsInventoryError("rights reservoir inventory is empty")
    return sorted(sources.values(), key=lambda row: row["source_id"]), reservoirs


def _card_licenses(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if isinstance(value, list) and all(
        isinstance(item, str) and item.strip() for item in value
    ):
        return sorted({item.strip() for item in value})
    return []


def build_inventory(
    reservoir_roots: list[Path],
    output_path: Path,
    *,
    token: str,
    acquire_card_function: Callable[
        [str, str, str, Path], dict[str, Any]
    ] = acquire_card_or_absence,
) -> dict[str, Any]:
    """Hash-bind all source cards and fail closed on non-exact declarations."""

    if output_path.exists() or output_path.is_symlink() or not reservoir_roots:
        raise ReservoirRightsInventoryError("rights inventory output differs")
    sources, reservoirs = _sources(reservoir_roots)
    identities = sorted({(row["repository"], row["revision"]) for row in sources})
    cards_by_identity = {}
    with tempfile.TemporaryDirectory(prefix="sai-reservoir-rights-") as temporary:
        for repository, revision in identities:
            card = acquire_card_function(repository, revision, token, Path(temporary))
            readme_present = card.get("readme_present", True)
            if (
                card.get("repository") != repository
                or card.get("revision") != revision
                or not isinstance(readme_present, bool)
                or not isinstance(card.get("readme_bytes"), int)
                or (readme_present and card["readme_bytes"] <= 0)
                or (not readme_present and card["readme_bytes"] != 0)
                or (
                    readme_present
                    and (
                        not isinstance(card.get("readme_sha256"), str)
                        or len(card["readme_sha256"]) != 64
                    )
                )
                or (not readme_present and card.get("readme_sha256") is not None)
                or card.get("card_text_persisted") is not False
            ):
                raise ReservoirRightsInventoryError("rights card receipt differs")
            cards_by_identity[(repository, revision)] = card
    source_rows = []
    route_counts: Counter[str] = Counter()
    for source in sources:
        card = cards_by_identity[(source["repository"], source["revision"])]
        manifest_classification = classify_declared_license(
            source["declared_license"]
        )
        card_declarations = _card_licenses(card.get("top_level_card_license"))
        card_classifications = [
            classify_declared_license(value) for value in card_declarations
        ]
        exact_manifest_recognized = manifest_classification[
            "declaration_recognized"
        ]
        exact_card_recognized = bool(card_classifications) and all(
            row["declaration_recognized"] for row in card_classifications
        )
        if card.get("readme_present", True) is False:
            route = "source_terms_resolution_required"
        elif source["declared_license"].startswith(
            ("common_pile_", "source_specific_")
        ):
            route = "per_row_license_evidence_required"
        elif exact_manifest_recognized or exact_card_recognized:
            route = "recognized_declaration_obligations_required"
        else:
            route = "source_terms_resolution_required"
        route_counts[route] += 1
        row = {
            **source,
            "manifest_declaration_classification": manifest_classification,
            "card": card,
            "card_license_declarations": card_declarations,
            "card_license_classifications": card_classifications,
            "rights_work_route": route,
            "source_text_persisted": False,
            "source_wide_rights_clearance_established": False,
            "legal_clearance_established": False,
            "training_ready": False,
        }
        row["source_row_sha256"] = canonical_sha256(row)
        source_rows.append(row)
    payload = {
        "schema": SCHEMA,
        "status": "complete_inventory_not_legal_clearance",
        "reservoirs": reservoirs,
        "reservoirs_sha256": canonical_sha256(reservoirs),
        "license_policy": POLICY,
        "license_policy_sha256": canonical_sha256(POLICY),
        "source_rows": source_rows,
        "source_rows_sha256": canonical_sha256(source_rows),
        "summary": {
            "sources": len(source_rows),
            "repositories": len(identities),
            "files": sum(row["files"] for row in source_rows),
            "physical_candidate_bytes": sum(row["bytes"] for row in source_rows),
            "rights_work_routes": dict(sorted(route_counts.items())),
        },
        "source_text_persisted": False,
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
    parser.add_argument("--reservoir-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()
    import os

    result = build_inventory(
        args.reservoir_root,
        args.output,
        token=os.environ.get(args.token_env, ""),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
