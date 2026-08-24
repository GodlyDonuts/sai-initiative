import json
from pathlib import Path

import pytest

from sai.data.token_stream import canonical_sha256, sha256_file, sha256_tree
from sai.data.transient_tokenizer_sample_aggregate import (
    SCHEMA as SAMPLE_SCHEMA,
)
from sai.data.transient_tokenizer_sample_aggregate import (
    STATUS as SAMPLE_STATUS,
)
from sai.tokenizer.build import SCHEMA as BUILD_SCHEMA
from sai.tokenizer.qualification import (
    CANDIDATE_SIZES,
)
from sai.tokenizer.qualification import (
    RECEIPT_SCHEMA as SELECTED_SCHEMA,
)
from sai.tokenizer.qualification import SCHEMA as QUALIFICATION_SCHEMA
from sai.tokenizer.tournament_custody import (
    TokenizerTournamentCustodyError,
    seal,
)


def _signed(path: Path, payload: dict, field: str = "receipt_sha256") -> dict:
    payload[field] = canonical_sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def _samples(root: Path, shards: int) -> dict:
    total = 0
    for index in range(shards):
        path = root / "samples" / f"shard_{index:05d}" / "sample.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(f'{{"shard":{index}}}\n')
        total += path.stat().st_size
    return _signed(
        root / "aggregate.json",
        {
            "schema": SAMPLE_SCHEMA,
            "status": SAMPLE_STATUS,
            "shards": {"logical_shards": shards},
            "totals": {"jsonl_bytes": total},
            "exact_document_identity_unique": True,
            "development_partition_excluded": True,
            "training_ready": False,
        },
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    books = tmp_path / "books"
    pleias = tmp_path / "pleias"
    tournament = tmp_path / "tournament"
    protected = tmp_path / "protected.jsonl"
    protected.write_text('{"protected":true}\n')
    _samples(books, 64)
    _samples(pleias, 128)
    paths = [
        *[
            books / "samples" / f"shard_{index:05d}" / "sample.jsonl"
            for index in range(64)
        ],
        *[
            pleias / "samples" / f"shard_{index:05d}" / "sample.jsonl"
            for index in range(128)
        ],
    ]
    sources = [
        {
            "order": order,
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for order, path in enumerate(paths)
    ]
    candidate_identities = {}
    for name, size in CANDIDATE_SIZES.items():
        root = tournament / "candidate-builds" / name
        tree = root / name
        tree.mkdir(parents=True)
        (tree / "tokenizer.json").write_text(json.dumps({"name": name}))
        identity = sha256_tree(tree)
        candidate_identities[name] = identity
        _signed(
            root / "manifest.json",
            {
                "schema": BUILD_SCHEMA,
                "status": "complete",
                "training_authorized": False,
                "source_receipts": sources,
                "source_identity_sha256": canonical_sha256(sources),
                "candidates": {
                    name: {
                        "vocab_size": size,
                        "root": name,
                        "tree_sha256": identity,
                    }
                },
            },
            "manifest_sha256",
        )
    qualification = _signed(
        tournament / "qualification.json",
        {
            "schema": QUALIFICATION_SCHEMA,
            "status": "all_candidates_qualified",
            "training_authorized": False,
            "source_receipts": sources,
            "protected_suite_receipt": {
                "path": str(protected.resolve()),
                "sha256": sha256_file(protected),
            },
            "candidates": {
                name: {
                    "qualified": True,
                    "tokenizer_identity_sha256": identity,
                }
                for name, identity in candidate_identities.items()
            },
        },
        "report_sha256",
    )
    _signed(
        tournament / "selected-48k.json",
        {
            "schema": SELECTED_SCHEMA,
            "status": "qualified",
            "training_authorized": False,
            "vocab_size": 48_000,
            "tokenizer_identity_sha256": candidate_identities["48k"],
            "tournament_report_sha256": qualification["report_sha256"],
        },
    )
    return pleias, books, tournament, protected


def test_custody_binds_both_aggregates_and_all_candidates(tmp_path: Path) -> None:
    pleias, books, tournament, protected = _fixture(tmp_path)
    result = seal(
        pleias,
        books,
        tournament,
        protected,
        tournament / "custody.json",
    )
    assert result["source"]["source_files"] == 192
    assert result["matched_source_receipts_across_all_candidates"] is True
    assert result["all_candidates_lossless_and_qualified"] is True
    assert result["production_tokenizer_selected"] is False
    assert result["mechanically_selected_candidate"] == "48k"
    assert result["capability_selection_complete"] is False


def test_custody_rejects_candidate_source_mismatch(tmp_path: Path) -> None:
    pleias, books, tournament, protected = _fixture(tmp_path)
    manifest_path = tournament / "candidate-builds" / "32k" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_receipts"][0]["sha256"] = "f" * 64
    manifest.pop("manifest_sha256")
    _signed(manifest_path, manifest, "manifest_sha256")
    with pytest.raises(
        TokenizerTournamentCustodyError, match="candidate build differs"
    ):
        seal(
            pleias,
            books,
            tournament,
            protected,
            tournament / "custody.json",
        )
