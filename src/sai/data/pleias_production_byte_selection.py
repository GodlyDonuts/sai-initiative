"""Select a deterministic, diverse PleIAs candidate core under a byte ceiling."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.pleias_production_near_dedup import SCHEMA as NEAR_SCHEMA
from sai.data.pleias_production_normalized_exact_dedup import SCHEMA as EXACT_SCHEMA
from sai.data.pleias_production_normalized_exact_dedup import _load_signed
from sai.data.token_stream import canonical_sha256, sha256_file

SCHEMA = "sai-pleias-production-byte-selection-v1"
MAXIMUM_SINGLE_STRATUM_PPM = 200_000


class PleiasProductionByteSelectionError(RuntimeError):
    """Dedup custody, byte accounting, or deterministic selection differs."""


def _database(root: Path, receipt: dict[str, Any], key: str) -> Path:
    descriptor = receipt.get(key)
    path = root / descriptor.get("path", "") if isinstance(descriptor, dict) else root
    if (
        not isinstance(descriptor, dict)
        or not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.stat().st_size != descriptor.get("bytes")
        or sha256_file(path) != descriptor.get("sha256")
    ):
        raise PleiasProductionByteSelectionError("selection input database differs")
    return path


def choose_rows(
    rows: list[tuple[str, str, int]], maximum_bytes: int
) -> tuple[set[str], dict[str, int], int]:
    """Protect broad strata first, then deterministically refill spare capacity."""

    if maximum_bytes <= 0:
        raise PleiasProductionByteSelectionError("maximum bytes differs")
    normalized = sorted(rows, key=lambda row: row[0])
    if any(
        not isinstance(identity, str)
        or not identity
        or not isinstance(stratum, str)
        or not stratum
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        for identity, stratum, size in normalized
    ):
        raise PleiasProductionByteSelectionError("candidate row differs")
    if len({row[0] for row in normalized}) != len(normalized):
        raise PleiasProductionByteSelectionError("candidate identities overlap")
    total = sum(row[2] for row in normalized)
    if total <= maximum_bytes:
        by_stratum: Counter[str] = Counter()
        for _identity, stratum, size in normalized:
            by_stratum[stratum] += size
        return {row[0] for row in normalized}, dict(by_stratum), total
    stratum_cap = maximum_bytes * MAXIMUM_SINGLE_STRATUM_PPM // 1_000_000
    selected: set[str] = set()
    by_stratum = Counter()
    selected_bytes = 0
    for identity, stratum, size in normalized:
        if (
            selected_bytes + size <= maximum_bytes
            and by_stratum[stratum] + size <= stratum_cap
        ):
            selected.add(identity)
            by_stratum[stratum] += size
            selected_bytes += size
    for identity, stratum, size in normalized:
        if identity in selected or selected_bytes + size > maximum_bytes:
            continue
        selected.add(identity)
        by_stratum[stratum] += size
        selected_bytes += size
    return selected, dict(by_stratum), selected_bytes


def build_selection(
    exact_root: Path,
    near_root: Path,
    output_root: Path,
    maximum_bytes: int,
) -> dict[str, Any]:
    """Write a text-free exact-row selection under the frozen maximum."""

    if output_root.exists() or output_root.is_symlink() or maximum_bytes <= 0:
        raise PleiasProductionByteSelectionError("selection arguments differ")
    exact = _load_signed(exact_root / "receipt.json", EXACT_SCHEMA)
    near = _load_signed(near_root / "receipt.json", NEAR_SCHEMA)
    exact_database = _database(exact_root, exact, "keep_database")
    near_database = _database(near_root, near, "drop_database")
    if (
        exact.get("normalized_exact_deduplication_complete") is not True
        or near.get("high_precision_near_duplicate_pass_complete") is not True
        or near.get("source", {}).get("normalized_exact_receipt_sha256")
        != exact.get("receipt_sha256")
        or exact.get("decision_contains_source_text") is not False
        or near.get("decision_contains_source_text") is not False
    ):
        raise PleiasProductionByteSelectionError("selection lineage differs")
    output_root.mkdir(parents=True)
    database_path = output_root / "selected_rows.sqlite3"
    temporary = output_root / f".selection.partial.{uuid.uuid4().hex}.sqlite3"
    output = sqlite3.connect(temporary)
    try:
        output.execute("PRAGMA journal_mode=DELETE")
        output.execute("PRAGMA synchronous=FULL")
        output.execute("PRAGMA temp_store=FILE")
        output.execute("ATTACH DATABASE ? AS exact", (str(exact_database.resolve()),))
        output.execute("ATTACH DATABASE ? AS near", (str(near_database.resolve()),))
        candidate_rows, candidate_bytes = output.execute(
            "SELECT COUNT(*), COALESCE(SUM(k.text_utf8_bytes), 0) "
            "FROM exact.keep k LEFT JOIN near.drops d ON "
            "d.source_row_identity_sha256=k.source_row_identity_sha256 "
            "WHERE d.source_row_identity_sha256 IS NULL"
        ).fetchone()
        output.execute(
            "CREATE TABLE chosen ("
            "identity TEXT PRIMARY KEY, stratum TEXT NOT NULL, bytes INTEGER NOT NULL"
            ") WITHOUT ROWID"
        )
        selected_bytes = 0
        by_stratum: Counter[str] = Counter()
        candidates_sql = (
            "SELECT k.source_row_identity_sha256, k.stratum, k.text_utf8_bytes "
            "FROM exact.keep k LEFT JOIN near.drops d ON "
            "d.source_row_identity_sha256=k.source_row_identity_sha256 "
            "WHERE d.source_row_identity_sha256 IS NULL "
            "ORDER BY k.stratum_quality_floor_milli DESC, "
            "k.stratum_quality_mean_milli DESC, k.source_row_identity_sha256"
        )
        if candidate_bytes <= maximum_bytes:
            output.execute(
                "INSERT INTO chosen "
                "SELECT k.source_row_identity_sha256, k.stratum, k.text_utf8_bytes "
                "FROM exact.keep k LEFT JOIN near.drops d ON "
                "d.source_row_identity_sha256=k.source_row_identity_sha256 "
                "WHERE d.source_row_identity_sha256 IS NULL"
            )
            selected_bytes = candidate_bytes
        else:
            stratum_cap = maximum_bytes * MAXIMUM_SINGLE_STRATUM_PPM // 1_000_000
            for identity, stratum, size in output.execute(candidates_sql):
                if (
                    selected_bytes + size <= maximum_bytes
                    and by_stratum[stratum] + size <= stratum_cap
                ):
                    output.execute(
                        "INSERT INTO chosen VALUES (?, ?, ?)",
                        (identity, stratum, size),
                    )
                    selected_bytes += size
                    by_stratum[stratum] += size
            refill_sql = (
                "SELECT k.source_row_identity_sha256, k.stratum, "
                "k.text_utf8_bytes FROM exact.keep k "
                "LEFT JOIN near.drops d ON "
                "d.source_row_identity_sha256=k.source_row_identity_sha256 "
                "LEFT JOIN chosen c ON "
                "c.identity=k.source_row_identity_sha256 "
                "WHERE d.source_row_identity_sha256 IS NULL AND c.identity IS NULL "
                "ORDER BY k.stratum_quality_floor_milli DESC, "
                "k.stratum_quality_mean_milli DESC, k.source_row_identity_sha256"
            )
            for identity, stratum, size in output.execute(refill_sql):
                if selected_bytes + size > maximum_bytes:
                    continue
                output.execute(
                    "INSERT INTO chosen VALUES (?, ?, ?)",
                    (identity, stratum, size),
                )
                selected_bytes += size
                by_stratum[stratum] += size
        output.execute(
            "CREATE TABLE selected ("
            "source_row_identity_sha256 TEXT PRIMARY KEY, "
            "source_path TEXT NOT NULL, source_parent_sha256 TEXT NOT NULL, "
            "source_row_index INTEGER NOT NULL, content_sha256 TEXT NOT NULL, "
            "stratum TEXT NOT NULL, text_utf8_bytes INTEGER NOT NULL, "
            "token_count INTEGER NOT NULL, "
            "stratum_quality_floor_milli INTEGER NOT NULL, "
            "stratum_quality_mean_milli INTEGER NOT NULL"
            ") WITHOUT ROWID"
        )
        output.execute(
            "INSERT INTO selected SELECT k.source_row_identity_sha256, "
            "k.source_path, k.source_parent_sha256, k.source_row_index, "
            "k.content_sha256, k.stratum, k.text_utf8_bytes, k.token_count, "
            "k.stratum_quality_floor_milli, k.stratum_quality_mean_milli "
            "FROM exact.keep k JOIN chosen c ON "
            "c.identity=k.source_row_identity_sha256"
        )
        selected_rows, replay_bytes, selected_tokens = output.execute(
            "SELECT COUNT(*), COALESCE(SUM(text_utf8_bytes), 0), "
            "COALESCE(SUM(token_count), 0) FROM selected"
        ).fetchone()
        if replay_bytes != selected_bytes:
            raise PleiasProductionByteSelectionError("selected byte replay differs")
        by_stratum = Counter(
            dict(
                output.execute(
                    "SELECT stratum, SUM(text_utf8_bytes) FROM selected "
                    "GROUP BY stratum"
                )
            )
        )
        output.execute("DROP TABLE chosen")
        output.commit()
        output.execute(
            "CREATE INDEX selected_path ON selected(source_path, source_row_index)"
        )
        output.execute("CREATE INDEX selected_stratum ON selected(stratum)")
        output.commit()
        output.execute("DETACH DATABASE near")
        output.execute("DETACH DATABASE exact")
        if (
            sha256_file(exact_database) != exact["keep_database"]["sha256"]
            or sha256_file(near_database) != near["drop_database"]["sha256"]
        ):
            raise PleiasProductionByteSelectionError("selection input mutated")
        output.execute("VACUUM")
        output.close()
        os.replace(temporary, database_path)
    except BaseException:
        output.close()
        temporary.unlink(missing_ok=True)
        raise
    payload = {
        "schema": SCHEMA,
        "status": "complete_nontraining_pleias_production_byte_selection",
        "source": {
            "normalized_exact_receipt_sha256": exact["receipt_sha256"],
            "high_precision_near_receipt_sha256": near["receipt_sha256"],
        },
        "policy": {
            "maximum_text_utf8_bytes": maximum_bytes,
            "maximum_is_padding_floor": False,
            "selection_rank": (
                "stratum_quality_floor_desc_then_stratum_quality_mean_desc_then_"
                "source_row_identity_sha256_ascending"
            ),
            "first_pass_maximum_single_stratum_ppm": MAXIMUM_SINGLE_STRATUM_PPM,
            "second_pass_refills_unused_capacity_across_all_strata": True,
            "oversized_last_document_is_skipped": True,
        },
        "counts": {
            "post_near_candidate_rows": candidate_rows,
            "post_near_candidate_text_utf8_bytes": candidate_bytes,
            "selected_rows": selected_rows,
            "selected_text_utf8_bytes": selected_bytes,
            "selected_tokens": selected_tokens,
            "held_over_byte_ceiling_rows": candidate_rows - selected_rows,
            "held_over_byte_ceiling_text_utf8_bytes": candidate_bytes - selected_bytes,
        },
        "selected_text_utf8_bytes_by_stratum": dict(sorted(by_stratum.items())),
        "selection_database": {
            "path": database_path.name,
            "bytes": database_path.stat().st_size,
            "sha256": sha256_file(database_path),
            "rows": selected_rows,
        },
        "selection_contains_source_text": False,
        "byte_ceiling_respected": selected_bytes <= maximum_bytes,
        "padding_performed": False,
        "benchmark_decontamination_complete": False,
        "cross_source_near_deduplication_complete": False,
        "production_materialization_complete": False,
        "training_ready": False,
        "four_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    _atomic_create(output_root / "receipt.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-root", type=Path, required=True)
    parser.add_argument("--near-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--maximum-bytes", type=int, required=True)
    args = parser.parse_args()
    result = build_selection(
        args.exact_root,
        args.near_root,
        args.output_root,
        args.maximum_bytes,
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
