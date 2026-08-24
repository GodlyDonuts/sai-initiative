"""Seal the matched book+PleIAs tokenizer tournament and aggregate lineage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_production_materializer import _load_signed
from sai.data.token_stream import canonical_sha256, sha256_file, sha256_tree
from sai.data.transient_tokenizer_sample_aggregate import (
    SCHEMA as SAMPLE_AGGREGATE_SCHEMA,
)
from sai.data.transient_tokenizer_sample_aggregate import (
    STATUS as SAMPLE_AGGREGATE_STATUS,
)
from sai.tokenizer.build import SCHEMA as BUILD_SCHEMA
from sai.tokenizer.qualification import (
    CANDIDATE_SIZES,
)
from sai.tokenizer.qualification import (
    RECEIPT_SCHEMA as SELECTED_SCHEMA,
)
from sai.tokenizer.qualification import SCHEMA as QUALIFICATION_SCHEMA

SCHEMA = "sai-combined-tokenizer-tournament-custody-v1"
STATUS = "complete_nontraining_combined_tokenizer_tournament"


class TokenizerTournamentCustodyError(RuntimeError):
    """Sample aggregates, candidate builds, or matched qualification differ."""


def _load_manifest(path: Path, schema: str, identity_field: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
        raise TokenizerTournamentCustodyError("tournament manifest is unsafe")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise TokenizerTournamentCustodyError(
            "tournament manifest is invalid"
        ) from error
    unsigned = {key: item for key, item in value.items() if key != identity_field}
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get(identity_field) != canonical_sha256(unsigned)
    ):
        raise TokenizerTournamentCustodyError("tournament manifest differs")
    return value


def seal(
    pleias_samples_root: Path,
    book_samples_root: Path,
    tournament_root: Path,
    protected_suite: Path,
    output: Path,
) -> dict[str, Any]:
    """Verify identical corpora and bind all tournament artifacts atomically."""

    if output.exists() or output.is_symlink():
        raise TokenizerTournamentCustodyError("tournament custody output exists")
    pleias = _load_signed(
        pleias_samples_root / "aggregate.json", SAMPLE_AGGREGATE_SCHEMA
    )
    books = _load_signed(book_samples_root / "aggregate.json", SAMPLE_AGGREGATE_SCHEMA)
    if (
        pleias.get("status") != SAMPLE_AGGREGATE_STATUS
        or books.get("status") != SAMPLE_AGGREGATE_STATUS
        or pleias.get("shards", {}).get("logical_shards") != 128
        or books.get("shards", {}).get("logical_shards") != 64
        or pleias.get("exact_document_identity_unique") is not True
        or books.get("exact_document_identity_unique") is not True
        or pleias.get("development_partition_excluded") is not True
        or books.get("development_partition_excluded") is not True
    ):
        raise TokenizerTournamentCustodyError("sample aggregate differs")
    expected_paths = [
        *[
            book_samples_root / "samples" / f"shard_{index:05d}" / "sample.jsonl"
            for index in range(64)
        ],
        *[
            pleias_samples_root / "samples" / f"shard_{index:05d}" / "sample.jsonl"
            for index in range(128)
        ],
    ]
    expected_receipts = [
        {
            "order": order,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for order, path in enumerate(expected_paths)
    ]
    if sum(row["bytes"] for row in expected_receipts[:64]) != books.get(
        "totals", {}
    ).get("jsonl_bytes") or sum(
        row["bytes"] for row in expected_receipts[64:]
    ) != pleias.get("totals", {}).get("jsonl_bytes"):
        raise TokenizerTournamentCustodyError("sample source bytes differ")
    builds = {}
    source_identity = canonical_sha256(expected_receipts)
    for name, size in CANDIDATE_SIZES.items():
        root = tournament_root / "candidate-builds" / name
        manifest = _load_manifest(
            root / "manifest.json", BUILD_SCHEMA, "manifest_sha256"
        )
        candidate = manifest.get("candidates", {}).get(name)
        candidate_root = root / name
        if (
            manifest.get("status") != "complete"
            or manifest.get("training_authorized") is not False
            or manifest.get("source_receipts") != expected_receipts
            or manifest.get("source_identity_sha256") != source_identity
            or set(manifest.get("candidates", {})) != {name}
            or not isinstance(candidate, dict)
            or candidate.get("vocab_size") != size
            or candidate.get("root") != name
            or candidate.get("tree_sha256") != sha256_tree(candidate_root)
        ):
            raise TokenizerTournamentCustodyError("candidate build differs")
        builds[name] = {
            "manifest_sha256": manifest["manifest_sha256"],
            "tokenizer_tree_sha256": candidate["tree_sha256"],
        }
    qualification = _load_manifest(
        tournament_root / "qualification.json",
        QUALIFICATION_SCHEMA,
        "report_sha256",
    )
    selected = _load_manifest(
        tournament_root / "selected-48k.json", SELECTED_SCHEMA, "receipt_sha256"
    )
    if (
        qualification.get("status") != "all_candidates_qualified"
        or qualification.get("training_authorized") is not False
        or qualification.get("source_receipts") != expected_receipts
        or set(qualification.get("candidates", {})) != set(CANDIDATE_SIZES)
        or any(
            qualification["candidates"][name].get("qualified") is not True
            or qualification["candidates"][name].get("tokenizer_identity_sha256")
            != builds[name]["tokenizer_tree_sha256"]
            for name in CANDIDATE_SIZES
        )
        or selected.get("status") != "qualified"
        or selected.get("training_authorized") is not False
        or selected.get("vocab_size") != 48_000
        or selected.get("tokenizer_identity_sha256")
        != builds["48k"]["tokenizer_tree_sha256"]
        or selected.get("tournament_report_sha256") != qualification["report_sha256"]
        or qualification.get("protected_suite_receipt", {}).get("path")
        != str(protected_suite.resolve())
        or qualification.get("protected_suite_receipt", {}).get("sha256")
        != sha256_file(protected_suite)
    ):
        raise TokenizerTournamentCustodyError("qualification custody differs")
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "source": {
            "pleias_sample_aggregate_receipt_sha256": pleias["receipt_sha256"],
            "book_sample_aggregate_receipt_sha256": books["receipt_sha256"],
            "source_identity_sha256": source_identity,
            "source_files": len(expected_receipts),
            "source_jsonl_bytes": sum(row["bytes"] for row in expected_receipts),
        },
        "candidate_builds": builds,
        "qualification_report_sha256": qualification["report_sha256"],
        "selected_48k_receipt_sha256": selected["receipt_sha256"],
        "matched_source_receipts_across_all_candidates": True,
        "all_candidates_lossless_and_qualified": True,
        "development_partition_excluded": True,
        "production_tokenizer_selected": False,
        "mechanically_selected_candidate": "48k",
        "mechanically_selected_tokenizer_identity_sha256": builds["48k"][
            "tokenizer_tree_sha256"
        ],
        "capability_selection_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pleias-samples-root", type=Path, required=True)
    parser.add_argument("--book-samples-root", type=Path, required=True)
    parser.add_argument("--tournament-root", type=Path, required=True)
    parser.add_argument("--protected-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = seal(
        args.pleias_samples_root,
        args.book_samples_root,
        args.tournament_root,
        args.protected_suite,
        args.output,
    )
    print(
        json.dumps(
            {"status": result["status"], "receipt_sha256": result["receipt_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
